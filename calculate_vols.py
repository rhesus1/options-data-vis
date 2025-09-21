import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import json
from scipy.optimize import brentq, least_squares, minimize
from scipy.optimize import NonlinearConstraint, Bounds
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Treasury yield functions
FALLBACK_YIELDS = {
    '1 Mo': 0.0518, '2 Mo': 0.0522, '3 Mo': 0.0506, '6 Mo': 0.0468,
    '1 Yr': 0.0409, '2 Yr': 0.0364, '3 Yr': 0.0347, '5 Yr': 0.0347,
    '7 Yr': 0.0357, '10 Yr': 0.0368, '20 Yr': 0.0407, '30 Yr': 0.04
}
def get_dividend_yield(ticker):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        dividend_yield = info.get('dividendYield', 0.0)
        if dividend_yield is None or pd.isna(dividend_yield):
            return 0.0
        return np.float64(dividend_yield)
    except Exception:
        return 0.0
def fetch_treasury_yields(date_str=None, use_latest_if_missing=True):
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    target_date = pd.to_datetime(date_str)
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(start=(target_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
                          end=(target_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        if hist.empty:
            return FALLBACK_YIELDS
        hist.index = pd.to_datetime(hist.index)
        if target_date in hist.index:
            row = hist.loc[target_date]
        elif use_latest_if_missing:
            prior = hist[hist.index <= target_date]
            if prior.empty:
                return FALLBACK_YIELDS
            row = prior.iloc[-1]
        else:
            return FALLBACK_YIELDS
        tnx_yield = row['Close'] / 100
        yields = {}
        base_yields = FALLBACK_YIELDS
        base_10yr = base_yields['10 Yr']
        for maturity, base_yield in base_yields.items():
            yields[maturity] = np.float64(tnx_yield * (base_yield / base_10yr))
        return yields
    except Exception:
        return FALLBACK_YIELDS
def interpolate_r(T, yields_dict):
    if not yields_dict:
        return np.float64(0.05)
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30], dtype=np.float64)
    rates = np.array([yields_dict.get(k, np.nan) for k in FALLBACK_YIELDS.keys()], dtype=np.float64)
    valid = ~np.isnan(rates)
    maturities = maturities[valid]
    rates = rates[valid]
    if len(maturities) == 0:
        return np.float64(0.05)
    if T <= maturities[0]:
        return rates[0]
    if T >= maturities[-1]:
        return rates[-1]
    idx = np.searchsorted(maturities, T)
    t1, t2 = maturities[idx-1], maturities[idx]
    r1, r2 = rates[idx-1], rates[idx]
    return np.float64(r1 + (r2 - r1) * (T - t1) / (t2 - t1))
# Black-Scholes and IV functions
def black_scholes_price(S, K, T, r, sigma, option_type='call', q=0):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
def implied_vol_bsm(S, K, T, r, market_price, option_type='call', q=0):
    def objective(sigma):
        price = black_scholes_price(S, K, T, r, sigma, option_type, q)
        return price - market_price if not np.isnan(price) else np.inf
    m = S / K
    initial_guess = 0.5 if m > 0.8 and m < 1.2 else 1.0
    try:
        iv = brentq(objective, 0.0001, 50.0)
        model_price = black_scholes_price(S, K, T, r, iv, option_type, q)
        if abs(model_price - market_price) < 0.001 and iv > 0:
            return iv, 'verified'
        return np.nan, 'inaccurate'
    except Exception:
        return np.nan, 'convergence'
def _compute_iv_for_row(args):
    row, yields_dict, default_r, q = args
    if not isinstance(row, dict):
        return np.nan, default_r, np.nan, np.nan
    T = np.float64(row.get('Years_to_Expiry', 0))
    r = interpolate_r(T, yields_dict)
    bid = row.get('Bid', 0)
    ask = row.get('Ask', 0)
    market_price = np.float64((bid + ask) / 2) if bid > 0 and ask > 0 else np.nan
    iv_bs = np.nan
    if pd.isna(market_price) or market_price <= 0.01 or T <= 0:
        pass
    else:
        S = row.get('Last Stock Price', 0)
        K = row.get('Strike', 0)
        if S <= 0 or K <= 0:
            pass
        else:
            type_ = row.get('Type', 'call').lower()
            if type_ in ['call', 'put']:
                intrinsic = np.float64(max(S - K, 0)) if type_ == 'call' else np.float64(max(K - S, 0))
                market_price = np.maximum(market_price, intrinsic)
                iv_bs, _ = implied_vol_bsm(S, K, T, r, market_price, type_, q)
    return T, r, market_price, iv_bs
def calculate_iv_binomial(options_df, yields_dict, q=0, default_r=0.05, max_workers=None):
    options_df = options_df.copy()
    if len(options_df) == 0:
        return options_df
    rows = [row.to_dict() for _, row in options_df.iterrows()]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_compute_iv_for_row, (row, yields_dict, default_r, q)): idx
                         for idx, row in enumerate(rows)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                T, r, market_price, iv_bs = future.result()
                results.append((idx, T, r, market_price, iv_bs))
            except Exception:
                results.append((idx, np.nan, default_r, np.nan, np.nan))
    results.sort(key=lambda x: x[0])
    for idx, T, r, market_price, iv_bs in results:
        options_df.at[idx, 'T'] = T
        options_df.at[idx, 'r'] = r
        options_df.at[idx, 'market_price'] = market_price
        options_df.at[idx, 'IV_mid'] = iv_bs
    return options_df
# Volatility surface fitting
def global_vol_model_hyp(x, a0, a1, b0, b1, m0, m1, rho0, rho1, sigma0, sigma1, c):
    moneyness, tau = x
    k = np.log(moneyness)
    a_T = a0 + a1 / (1 + c * tau)
    b_T = b0 + b1 / (1 + c * tau)
    m_T = m0 + m1 / (1 + c * tau)
    rho_T = rho0 + rho1 / (1 + c * tau)
    sigma_T = sigma0 + sigma1 / (1 + c * tau)
    return a_T + b_T * (rho_T * (k - m_T) + np.sqrt((k - m_T)**2 + sigma_T**2))
MODEL_CONFIG = {
    'hyp': {
        'func': global_vol_model_hyp,
        'p0': [0.2, 0.1, 0.1, 0.05, 0.0, 0.0, -0.3, -0.1, 0.1, 0.05, 0.1],
        'bounds': ([0, -np.inf, 0, -np.inf, -np.inf, -np.inf, -1, -1, 0.01, -np.inf, 0],
                   [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf])
    }
}
def fit_vol_surface(df, ticker=None, exp_min=0.3, exp_max=np.inf, mon_min=0.3, mon_max=np.inf, extrap_tau=None, model='hyp', option_type='Call', p0=None, do_plot=True):
    df = df.copy()
    if model not in MODEL_CONFIG:
        raise ValueError(f"Model must be one of {list(MODEL_CONFIG.keys())}")
    df = df[(df['Type'] == option_type) & (df['Years_to_Expiry'] > exp_min) & (df['Years_to_Expiry'] < exp_max) &
            (df['Moneyness'] > mon_min) & (df['Moneyness'] < mon_max) & (df['IV_mid'].notna())]
    if ticker:
        df_ticker = df[df['Ticker'] == ticker]
        if df_ticker.empty:
            return None, None
        result = fit_single_ticker(df_ticker, model, p0=p0)
        if result[0] is not None and do_plot:
            plot_vol_surface_3d(df_ticker, result[0], ticker, exp_min, exp_max, mon_min, mon_max, extrap_tau, model, result[1], option_type)
        return result
    else:
        results = {}
        for tick, group in df.groupby('Ticker'):
            if group.empty:
                results[tick] = (None, None)
                continue
            result = fit_single_ticker(group, model, p0=p0)
            results[tick] = result
            if result[0] is not None and do_plot:
                plot_vol_surface_3d(group, result[0], tick, exp_min, exp_max, mon_min, mon_max, extrap_tau, model, result[1], option_type)
        return results
def fit_single_ticker(df, model, p0=None):
    df = df.copy()
    if len(df) < 4:
        return None, None
    df.loc[:, 'SMI'] = 100 * (df['Ask'] - df['Bid']).clip(lower=0) / (df['Bid'] + df['Ask']).clip(lower=1e-6) / 2
    df.loc[:, 'weight'] = np.log(1 + df['Open Interest']) / df['SMI'].clip(lower=1e-6)
    sigma_m = 0.6
    sigma_t = 0.6 * (df['Years_to_Expiry'].max() - df['Years_to_Expiry'].min()) if df['Years_to_Expiry'].max() > df['Years_to_Expiry'].min() else 1.0
    moneyness_dist = (df['Moneyness'] - 1.0)**2 / (2 * sigma_m**2)
    expiry_dist = (df['Years_to_Expiry'] - df['Years_to_Expiry'].median())**2 / (2 * sigma_t**2)
    df.loc[:, 'dist_weight'] = np.exp(-(moneyness_dist + expiry_dist))
    df.loc[:, 'weight'] *= df['dist_weight']
    outlier_weight = 0.01
    df['is_outlier'] = False
    df = df[df['IV_mid'] <= 5.0]
    for _, group in df.groupby('Years_to_Expiry'):
        if len(group) < 3:
            continue
        Q1 = group['IV_mid'].quantile(0.25)
        Q3 = group['IV_mid'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = group[(group['IV_mid'] < lower_bound) | (group['IV_mid'] > upper_bound)].index
        df.loc[outliers, 'is_outlier'] = True
    df.loc[df['is_outlier'], 'weight'] *= outlier_weight
    df = df[
        (df['IV_mid'] > 0) &
        (df['weight'] > 0) &
        (df['Bid'] >= 0) &
        (df['Ask'] >= df['Bid']) &
        (df['SMI'] > 0)
    ]
    if len(df) < 4:
        return None, None
    df = df.sort_values(['Moneyness', 'Years_to_Expiry'])
    df = df.drop_duplicates(subset=['Moneyness', 'Years_to_Expiry'], keep='first')
    M = df['Moneyness'].values
    tau = df['Years_to_Expiry'].values
    IV = df['IV_mid'].values
    w = df['weight'].values
    xdata = np.vstack((M, tau))
    IV_all = IV
    w_all = w
    model_config = MODEL_CONFIG[model]
    model_func = model_config['func']
    if p0 is None:
        p0 = model_config['p0']
    bounds = ([0, -np.inf, 0, -np.inf, -np.inf, -np.inf, -0.95, -0.95, 0.05, -np.inf, 0],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.95, 0.95, np.inf, np.inf, np.inf])
    max_nfev = 50000
    max_iterations = 5
    tolerance = 0.001
    try:
        def residuals(params):
            return np.sqrt(w_all) * (model_func(xdata, *params) - IV_all)
        result = least_squares(residuals, p0, bounds=bounds, method='trf', jac='2-point', max_nfev=max_nfev)
        popt = result.x
        current_residuals = np.sum(w * (IV - model_func(xdata, *popt))**2)
        prev_residuals = current_residuals
        for iteration in range(max_iterations):
            res = model_func(xdata, *popt) - IV_all
            Q1 = np.percentile(np.abs(res), 25)
            Q3 = np.percentile(np.abs(res), 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = np.abs(res) > upper_bound
            w_all[outlier_mask] *= 0.01
            median_res = np.median(np.abs(res))
            high_res_mask = (np.abs(res) > 1.5 * median_res) & (~outlier_mask)
            w_all[high_res_mask] *= 2.0
            result = least_squares(residuals, p0, bounds=bounds, method='trf', jac='2-point', max_nfev=max_nfev)
            popt = result.x
            new_residuals = np.sum(w * (IV - model_func(xdata, *popt))**2)
            if abs(new_residuals - prev_residuals) < tolerance:
                break
            prev_residuals = new_residuals
        return popt, current_residuals
    except Exception as e:
        return None, None
def calculate_smoothed_iv(df, params_calls, params_puts, model='hyp'):
    df = df.copy()
    df['Smoothed_IV'] = np.nan
    if params_calls is None and params_puts is None:
        return df
    model_func = MODEL_CONFIG['hyp']['func']
    valid_mask = (df['Moneyness'].notna()) & (df['Years_to_Expiry'] > 0)
    call_mask = (df['Type'] == 'Call') & valid_mask
    put_mask = (df['Type'] == 'Put') & valid_mask
    if params_calls is not None and call_mask.any():
        x_calls = np.vstack((df.loc[call_mask, 'Moneyness'].values, df.loc[call_mask, 'Years_to_Expiry'].values))
        smoothed_calls = model_func(x_calls, *params_calls)
        df.loc[call_mask, 'Smoothed_IV'] = np.clip(smoothed_calls, 0, np.inf)
    if params_puts is not None and put_mask.any():
        x_puts = np.vstack((df.loc[put_mask, 'Moneyness'].values, df.loc[put_mask, 'Years_to_Expiry'].values))
        smoothed_puts = model_func(x_puts, *params_puts)
        df.loc[put_mask, 'Smoothed_IV'] = np.clip(smoothed_puts, 0, np.inf)
    return df
def main():
    timestamp = '20250902_2137'
    data_dir = f'data/{timestamp}/cleaned_yfinance/'
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return
    files = [f for f in os.listdir(data_dir) if f.startswith('cleaned_yfinance_') and f.endswith('.csv')]
    if not files:
        print("No data files found.")
        return
    tickers = [f.replace('cleaned_yfinance_', '').replace('.csv', '') for f in files]
    exp_min_short = 0
    exp_max_long = np.inf
    exp_min_full = 0
    exp_max_full = np.inf
    mon_min = 0
    mon_max = np.inf
    extrap_tau = None
    model = 'hyp'
    all_metrics = []
    timestamp_dt = datetime.strptime(timestamp, '%Y%m%d_%H%M')
    timestamp_date = timestamp_dt.strftime('%Y-%m-%d')
    yields_dict = fetch_treasury_yields(timestamp_date)
    for ticker in tickers:
        data_file = os.path.join(data_dir, f'cleaned_yfinance_{ticker}.csv')
        if not os.path.exists(data_file):
            print(f"File {data_file} does not exist.")
            continue
        df = pd.read_csv(data_file, parse_dates=['Expiry'])
        if df.empty:
            print(f"Empty dataframe for {ticker}.")
            continue
        df['Ticker'] = ticker # Ensure Ticker column is set
        q = get_dividend_yield(ticker)
        df['Expiry_dt'] = pd.to_datetime(df['Expiry'])
        df['Years_to_Expiry'] = (df['Expiry_dt'] - timestamp_dt).dt.days / 365.25
        S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2 if 'Bid Stock' in df and 'Ask Stock' in df else df['Last Stock Price'].iloc[0]
        df['Last Stock Price'] = S
        df = calculate_iv_binomial(df, yields_dict, q=q, default_r=0.05, max_workers=None)
        if df.empty:
            print(f"Empty after IV calculation for {ticker}.")
            continue
        df['Forward'] = df['Last Stock Price'] * np.exp((df['r'] - q) * df['Years_to_Expiry'])
        df['Moneyness'] = df['Strike'] / df['Forward']
        params_calls = None
        params_puts = None
        best_exp_max_short_calls = None
        best_exp_min_long_calls = None
        best_exp_max_short_puts = None
        best_exp_min_long_puts = None
        for option_type in ['Call', 'Put']:
            df_type = df[df['Type'] == option_type].copy()
            if len(df_type) < 10:
                # Fallback to original
                exp_max_short = max(df_type['Years_to_Expiry'].min() + 0.3, 0.3)
                exp_min_long = min(df_type['Years_to_Expiry'].max() - 0.3, 1.8)
                print(f"{ticker} {option_type} fallback: exp_max_short = {exp_max_short}, exp_min_long = {exp_min_long}")
                # Stage 1: Short term fit to get initial parameters
                params_short, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_short, exp_max=exp_max_short,
                                                  mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                  model=model, option_type=option_type, p0=None, do_plot=False)
                p0_long = params_short if params_short is not None else None
                # Stage 2: Long term fit using short-term params as initial guess
                params_long, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_long, exp_max=exp_max_long,
                                                 mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                 model=model, option_type=option_type, p0=p0_long, do_plot=False)
                p0_full = params_long if params_long is not None else p0_long
                # Stage 3: Full range fit using long-term params as initial guess
                params, residuals = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_full, exp_max=exp_max_full,
                                                    mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                    model=model, option_type=option_type, p0=p0_full, do_plot=False)
                if option_type == 'Call':
                    best_exp_max_short_calls = exp_max_short
                    best_exp_min_long_calls = exp_min_long
                    params_calls = params
                else:
                    best_exp_max_short_puts = exp_max_short
                    best_exp_min_long_puts = exp_min_long
                    params_puts = params
                continue
            expiry_min = df_type['Years_to_Expiry'].min()
            expiry_max = df_type['Years_to_Expiry'].max()
            if expiry_max - expiry_min < 0.1:
                # Fallback if range too small
                exp_max_short = max(expiry_min + 0.3, 0.3)
                exp_min_long = min(expiry_max - 0.3, 1.8)
                print(f"{ticker} {option_type} fallback: exp_max_short = {exp_max_short}, exp_min_long = {exp_min_long}")
                params_short, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_short, exp_max=exp_max_short,
                                                  mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                  model=model, option_type=option_type, p0=None, do_plot=False)
                p0_long = params_short if params_short is not None else None
                params_long, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_long, exp_max=exp_max_long,
                                                 mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                 model=model, option_type=option_type, p0=p0_long, do_plot=False)
                p0_full = params_long if params_long is not None else p0_long
                params, residuals = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_full, exp_max=exp_max_full,
                                                    mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                    model=model, option_type=option_type, p0=p0_full, do_plot=False)
                if option_type == 'Call':
                    best_exp_max_short_calls = exp_max_short
                    best_exp_min_long_calls = exp_min_long
                    params_calls = params
                else:
                    best_exp_max_short_puts = exp_max_short
                    best_exp_min_long_puts = exp_min_long
                    params_puts = params
                continue
            median_expiry = df_type['Years_to_Expiry'].median()
            initial_guess = [df_type['Years_to_Expiry'].quantile(0.25), df_type['Years_to_Expiry'].quantile(0.75)]
            lb = [expiry_min + 0.01, median_expiry + 0.05]
            ub = [median_expiry - 0.05, expiry_max - 0.01]
            bounds_obj = Bounds(lb, ub)
            con = NonlinearConstraint(lambda x: x[1] - x[0], 0.05, np.inf)
            def objective(x):
                exp_max_s, exp_min_l = x
                p90 = compute_p90(exp_max_s, exp_min_l, df, df_type, ticker, option_type, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau, model)
                return p90
            res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds_obj, constraints=[con], options={'maxiter': 10})
            if res.success and res.fun < np.inf:
                best_exp_max_short = res.x[0]
                best_exp_min_long = res.x[1]
            else:
                # Fallback if optimization fails
                best_exp_max_short = max(df_type['Years_to_Expiry'].min() + 0.3, 0.3)
                best_exp_min_long = min(df_type['Years_to_Expiry'].max() - 0.3, 1.8)
            # Refit with best values
            params_short, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_short, exp_max=best_exp_max_short,
                                              mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                              model=model, option_type=option_type, p0=None, do_plot=False)
            p0_long = params_short if params_short is not None else None
            params_long, _ = fit_vol_surface(df, ticker=ticker, exp_min=best_exp_min_long, exp_max=exp_max_long,
                                             mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                             model=model, option_type=option_type, p0=p0_long, do_plot=False)
            p0_full = params_long if params_long is not None else p0_long
            params, residuals = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_full, exp_max=exp_max_full,
                                                mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                model=model, option_type=option_type, p0=p0_full, do_plot=False)
            if option_type == 'Call':
                best_exp_max_short_calls = best_exp_max_short
                best_exp_min_long_calls = best_exp_min_long
                params_calls = params
            else:
                best_exp_max_short_puts = best_exp_max_short
                best_exp_min_long_puts = best_exp_min_long
                params_puts = params
    df = calculate_smoothed_iv(df, params_calls, params_puts, model)
    # Save best params
    params_dir = f'data/{timestamp}/params_yfinance'
    os.makedirs(params_dir, exist_ok=True)
    if params_calls is not None:
        np.save(os.path.join(params_dir, f'params_yfinance_{ticker}_calls.npy'), params_calls)
    if params_puts is not None:
        np.save(os.path.join(params_dir, f'params_yfinance_{ticker}_puts.npy'), params_puts)
    # Compute ATM-normalized relative errors
    valid_mask = (df['IV_mid'].notna() & df['Smoothed_IV'].notna() &
                  (df['IV_mid'] > 0) & (df['Smoothed_IV'] > 0))
    ticker_metrics = []
    for opt_type in ['Call', 'Put']:
        type_mask = df['Type'] == opt_type
        df_type = df[valid_mask & type_mask].copy()
        params = params_calls if opt_type == 'Call' else params_puts
        best_short = best_exp_max_short_calls if opt_type == 'Call' else best_exp_max_short_puts
        best_long = best_exp_min_long_calls if opt_type == 'Call' else best_exp_min_long_puts
        atm_iv = np.nan
        if params is not None:
            try:
                atm_iv = global_vol_model_hyp(np.array([[1.0], [1.0]]), *params)
                atm_iv = float(atm_iv.item()) if isinstance(atm_iv, np.ndarray) else float(atm_iv)
                if np.isnan(atm_iv) or atm_iv <= 0.01:
                    atm_iv = np.nan
            except Exception:
                atm_iv = np.nan
        # Compute relative errors for valid rows in df_type
        df_type['rel_error_atm_pct'] = np.where((df_type['IV_mid'].notna()) &
                                                (df_type['Smoothed_IV'].notna()) &
                                                (atm_iv > 0),
                                                abs((df_type['IV_mid'] - df_type['Smoothed_IV']) / atm_iv) * 100,
                                                np.nan)
        atm_candidates = df_type[
            (abs(df_type['Years_to_Expiry'] - 1) <= 0.5) & # Increased to 0.5
            (abs(df_type['Moneyness'] - 1) <= 0.1) # Increased to 0.1
        ].copy()
        one_yr_atm_residual = np.nan
        atm_details = " (no close match)"
        atm_dist_t = np.nan
        atm_dist_m = np.nan
        if not atm_candidates.empty:
            atm_candidates['dist_to_atm'] = (atm_candidates['Years_to_Expiry'] - 1)**2 + (atm_candidates['Moneyness'] - 1)**2
            closest_idx = atm_candidates['dist_to_atm'].idxmin()
            one_yr_atm_residual = df_type.at[closest_idx, 'rel_error_atm_pct']
            atm_dist_t = abs(df_type.at[closest_idx, 'Years_to_Expiry'] - 1)
            atm_dist_m = abs(df_type.at[closest_idx, 'Moneyness'] - 1)
            atm_details = f" (closest: T={df_type.at[closest_idx, 'Years_to_Expiry']:.2f}, " \
                          f"M={df_type.at[closest_idx, 'Moneyness']:.2f})"
        else:
            # Use the closest if no candidates
            df_type['dist_to_atm'] = (df_type['Years_to_Expiry'] - 1)**2 + (df_type['Moneyness'] - 1)**2
            closest_idx = df_type['dist_to_atm'].idxmin()
            one_yr_atm_residual = df_type.at[closest_idx, 'rel_error_atm_pct']
            atm_dist_t = abs(df_type.at[closest_idx, 'Years_to_Expiry'] - 1)
            atm_dist_m = abs(df_type.at[closest_idx, 'Moneyness'] - 1)
            atm_details = f" (closest: T={df_type.at[closest_idx, 'Years_to_Expiry']:.2f}, " \
                          f"M={df_type.at[closest_idx, 'Moneyness']:.2f})"
        p90_rel_error = df_type['rel_error_atm_pct'].quantile(0.9) if not df_type.empty else np.nan
        if not np.isnan(p90_rel_error) and p90_rel_error > 1000:
            print(f"Warning: High P90 rel error ({p90_rel_error:.2f}%) for {ticker} ({opt_type}), setting to NaN")
            p90_rel_error = np.nan
        print(f"{ticker} ({opt_type}): 1yr ATM rel error = {one_yr_atm_residual:.2f}%{atm_details}, "
              f"P90 rel error (ATM-norm) = {p90_rel_error:.2f}% (n_valid={len(df_type)})")
        metrics_df = pd.DataFrame({
            'Ticker': [ticker],
            'Timestamp': [timestamp],
            'Option_Type': [opt_type],
            'One_Yr_ATM_Rel_Error_Pct': [one_yr_atm_residual],
            'P90_Rel_Error_Pct': [p90_rel_error],
            'N_Valid_Options': [len(df_type)],
            'ATM_Dist_T': [atm_dist_t],
            'ATM_Dist_M': [atm_dist_m],
            'ATM_IV': [atm_iv]
        })
        ticker_metrics.append(metrics_df)
    df = calculate_smoothed_iv(df, params_calls, params_puts, model)
    # Compute ATM-normalized relative errors for calls and puts
    valid_mask = (df['IV_mid'].notna() & df['Smoothed_IV'].notna() &
                  (df['IV_mid'] > 0) & (df['Smoothed_IV'] > 0))
    for opt_type in ['Call', 'Put']:
        type_mask = df['Type'] == opt_type
        df_type = df[valid_mask & type_mask].copy()
        params = params_calls if opt_type == 'Call' else params_puts
        atm_iv = np.nan
        if params is not None:
            try:
                atm_iv = global_vol_model_hyp(np.array([[1.0], [1.0]]), *params)
                atm_iv = float(atm_iv.item()) if isinstance(atm_iv, np.ndarray) else float(atm_iv)
                if np.isnan(atm_iv) or atm_iv <= 0.01:
                    atm_iv = np.nan
            except Exception:
                atm_iv = np.nan
        # Compute relative errors for valid rows in df_type
        df_type['rel_error_atm_pct'] = np.where((df_type['IV_mid'].notna()) &
                                                (df_type['Smoothed_IV'].notna()) &
                                                (atm_iv > 0),
                                                abs((df_type['IV_mid'] - df_type['Smoothed_IV']) / atm_iv) * 100,
                                                np.nan)
        atm_candidates = df_type[
            (abs(df_type['Years_to_Expiry'] - 1) <= 0.25) & # Within ~3 months
            (abs(df_type['Moneyness'] - 1) <= 0.05) # Within 5% moneyness
        ].copy()
        one_yr_atm_residual = np.nan
        atm_details = " (no close match)"
        atm_dist_t = np.nan
        atm_dist_m = np.nan
        if not atm_candidates.empty:
            atm_candidates['dist_to_atm'] = (abs(atm_candidates['Years_to_Expiry'] - 1) +
                                            abs(atm_candidates['Moneyness'] - 1))
            closest_idx = atm_candidates['dist_to_atm'].idxmin()
            one_yr_atm_residual = df_type.at[closest_idx, 'rel_error_atm_pct']
            atm_dist_t = abs(df_type.at[closest_idx, 'Years_to_Expiry'] - 1)
            atm_dist_m = abs(df_type.at[closest_idx, 'Moneyness'] - 1)
            atm_details = f" (closest: T={df_type.at[closest_idx, 'Years_to_Expiry']:.2f}, " \
                          f"M={df_type.at[closest_idx, 'Moneyness']:.2f})"
              
        p90_rel_error = df_type['rel_error_atm_pct'].quantile(0.9) if not df_type.empty else np.nan
        if not np.isnan(p90_rel_error) and p90_rel_error > 1000: # Cap extreme values
            p90_rel_error = np.nan
        metrics_df = pd.DataFrame({
            'Ticker': [ticker],
            'Timestamp': [timestamp],
            'Option_Type': [opt_type],
            'One_Yr_ATM_Rel_Error_Pct': [one_yr_atm_residual],
            'P90_Rel_Error_Pct': [p90_rel_error],
            'N_Valid_Options': [len(df_type)],
            'ATM_Dist_T': [atm_dist_t],
            'ATM_Dist_M': [atm_dist_m],
            'ATM_IV': [atm_iv]
        })
        all_metrics.append(metrics_df)
    df['TotalVariance'] = df['Smoothed_IV']**2 * df['Years_to_Expiry']
    df['TotalVariance'] = df['TotalVariance'].fillna(np.nan)
    r = df['r'].iloc[0]
    output_columns = [
        'Ticker', 'Contract Name', 'Type', 'Expiry', 'Strike', 'Moneyness', 'Bid', 'Ask', 'Volume', 'Open Interest',
        'Bid Stock', 'Ask Stock', 'Last Stock Price', 'Implied Volatility', 'Expiry_dt', 'Years_to_Expiry', 'Forward',
        'LogMoneyness', 'IV_bid', 'IV_ask', 'IV_spread', 'Delta', 'Ivol/Rvol100d Ratio', 'Smoothed_IV',
        'TotalVariance', 'Call Local Vol', 'Put Local Vol', 'Realised Vol 100d', 'IV_mid_binomial'
    ]
    df['Implied Volatility'] = df['IV_mid']
    df['IV_bid'] = np.nan
    df['IV_ask'] = np.nan
    df['IV_spread'] = np.nan
    df['Delta'] = np.nan
    df['Call Local Vol'] = np.nan
    df['Put Local Vol'] = np.nan
    df = df[output_columns]
    output_file = os.path.join(processed_dir, f'processed_yfinance_{ticker}.csv')
    df.to_csv(output_file, index=False)
    df.to_json(os.path.join(processed_dir, f'processed_yfinance_{ticker}.json'), orient='records', date_format='iso')
    param_names = ['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']
    vol_surf_file = os.path.join(vol_surf_dir, 'vol_surf.csv')
    param_dfs = []
    if params_calls is not None:
        param_df_calls = pd.DataFrame([params_calls], columns=param_names)
        param_df_calls['Ticker'] = ticker
        param_df_calls['Model'] = model
        param_df_calls['Residuals'] = residuals
        param_df_calls['Timestamp'] = timestamp
        param_df_calls['Option_Type'] = 'Call'
        param_dfs.append(param_df_calls)
    if params_puts is not None:
        param_df_puts = pd.DataFrame([params_puts], columns=param_names)
        param_df_puts['Ticker'] = ticker
        param_df_puts['Model'] = model
        param_df_puts['Residuals'] = residuals
        param_df_puts['Timestamp'] = timestamp
        param_df_puts['Option_Type'] = 'Put'
        param_dfs.append(param_df_puts)
    if param_dfs:
        param_df = pd.concat(param_dfs, ignore_index=True)
        if os.path.exists(vol_surf_file):
            try:
                existing_params = pd.read_csv(vol_surf_file)
                param_df = pd.concat([existing_params, param_df], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing {vol_surf_file}: {e}")
        param_df.to_csv(vol_surf_file, index=False)
        print(f"Volatility surface parameters saved to {vol_surf_file}")
    skew_metrics_df, slope_metrics_df = calculate_skew_slope_metrics(df, ticker, timestamp, r, q= q)
if __name__ == '__main__':
    main()
