import pandas as pd
import numpy as np
from datetime import datetime
import os
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
    target_date = pd.to_datetime(date_str).tz_localize(None)
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(start=(target_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
                          end=(target_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        if hist is None or hist.empty:
            return FALLBACK_YIELDS
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
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
    initial_guess = 0.5 if 0.8 < m < 1.2 else 1.0
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
        return np.nan, default_r, np.nan, np.nan, 'invalid_row'
    T = np.float64(row.get('Years_to_Expiry', 0))
    r = interpolate_r(T, yields_dict)
    bid = row.get('Bid', 0)
    ask = row.get('Ask', 0)
    market_price = np.float64((bid + ask) / 2) if bid > 0 and ask > 0 else np.nan
    iv_bs = np.nan
    status = 'valid'
    if pd.isna(market_price) or market_price <= 0.01 or T <= 0:
        status = 'invalid_market_price_or_expiry'
    else:
        S = row.get('Last Stock Price', 0)
        K = row.get('Strike', 0)
        if S <= 0 or K <= 0:
            status = 'invalid_stock_or_strike'
        else:
            type_ = row.get('Type', 'call').lower()
            if type_ not in ['call', 'put']:
                status = 'invalid_option_type'
            else:
                intrinsic = np.float64(max(S - K, 0)) if type_ == 'call' else np.float64(max(K - S, 0))
                market_price = np.maximum(market_price, intrinsic)
                iv_bs, status = implied_vol_bsm(S, K, T, r, market_price, type_, q)
    return T, r, market_price, iv_bs, status

def calculate_iv_binomial(options_df, yields_dict, q=0, default_r=0.05, max_workers=None):
    if options_df is None or len(options_df) == 0:
        return pd.DataFrame()
    options_df = options_df.copy()
    rows = [row.to_dict() for _, row in options_df.iterrows()]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_compute_iv_for_row, (row, yields_dict, default_r, q)): idx
                         for idx, row in enumerate(rows)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                T, r, market_price, iv_bs, status = future.result()
                results.append((idx, T, r, market_price, iv_bs, status))
            except Exception as e:
                results.append((idx, np.nan, default_r, np.nan, np.nan, f'error: {str(e)}'))
    results.sort(key=lambda x: x[0])
    for idx, T, r, market_price, iv_bs, status in results:
        options_df.at[idx, 'T'] = T
        options_df.at[idx, 'r'] = r
        options_df.at[idx, 'market_price'] = market_price
        options_df.at[idx, 'IV_mid'] = iv_bs
        options_df.at[idx, 'IV_status'] = status
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

def compute_p90(exp_max_s, exp_min_l, df, df_type, ticker, option_type, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau, model):
    if df is None or df_type is None:
        return np.inf
    if exp_max_s + 0.1 >= exp_min_l:
        return np.inf
    params_short, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_short, exp_max=exp_max_s,
                                      mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                      model=model, option_type=option_type, p0=None, do_plot=False, max_nfev=10000, max_iterations=2)
    if params_short is None:
        return np.inf
    p0_long = params_short
    params_long, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_l, exp_max=exp_max_long,
                                     mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                     model=model, option_type=option_type, p0=p0_long, do_plot=False, max_nfev=10000, max_iterations=2)
    p0_full = params_long if params_long is not None else p0_long
    params_full, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_full, exp_max=exp_max_full,
                                     mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                     model=model, option_type=option_type, p0=p0_full, do_plot=False, max_nfev=20000, max_iterations=3)
    if params_full is None:
        return np.inf
    valid_temp_mask = (df_type['Moneyness'].notna()) & (df_type['Years_to_Expiry'] > 0) & (df_type['IV_mid'].notna()) & (df_type['IV_mid'] > 0)
    valid_temp = df_type[valid_temp_mask]
    if len(valid_temp) == 0:
        return np.inf
    x_temp = np.vstack((valid_temp['Moneyness'].values, valid_temp['Years_to_Expiry'].values))
    smoothed_temp = global_vol_model_hyp(x_temp, *params_full)
    smoothed_temp = np.clip(smoothed_temp, 0.01, 5.0)
    try:
        atm_point = np.array([[1.0], [1.0]])
        atm_iv_temp = global_vol_model_hyp(atm_point, *params_full)
        atm_iv_temp = float(atm_iv_temp) if isinstance(atm_iv_temp, (np.ndarray, float)) else np.nan
    except Exception:
        atm_iv_temp = np.nan
    if np.isnan(atm_iv_temp) or atm_iv_temp <= 0:
        atm_iv_temp = valid_temp['IV_mid'].median()
    if np.isnan(atm_iv_temp) or atm_iv_temp <= 0:
        return np.inf
    rel_errors = np.abs((valid_temp['IV_mid'].values - smoothed_temp) / atm_iv_temp) * 100
    valid_rel = rel_errors[~np.isnan(rel_errors)]
    if len(valid_rel) == 0:
        return np.inf
    return np.percentile(valid_rel, 90)

def fit_vol_surface(df, ticker=None, exp_min=0.0, exp_max=np.inf, mon_min=0.0, mon_max=np.inf, extrap_tau=None, model='hyp', option_type='Call', p0=None, do_plot=False, max_nfev=20000, max_iterations=3):
    if df is None:
        return None, None
    df = df.copy()
    if model not in MODEL_CONFIG:
        return None, None
    df = df[(df['Type'] == option_type) & (df['Years_to_Expiry'] > exp_min) & (df['Years_to_Expiry'] < exp_max) &
            (df['Moneyness'] > mon_min) & (df['Moneyness'] < mon_max) & (df['IV_mid'].notna())]
    if ticker:
        df_ticker = df[df['Ticker'] == ticker]
        if df_ticker.empty:
            return None, None
        return fit_single_ticker(df_ticker, model, p0, max_nfev, max_iterations, ticker, option_type)
    else:
        results = {}
        for tick, group in df.groupby('Ticker'):
            if group.empty:
                results[tick] = (None, None)
                continue
            results[tick] = fit_single_ticker(group, model, p0, max_nfev, max_iterations, tick, option_type)
        return results

def fit_single_ticker(df, model, p0=None, max_nfev=20000, max_iterations=3, ticker=None, option_type='Call'):
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
            if abs(new_residuals - prev_residuals) < 0.001:
                break
            prev_residuals = new_residuals
        return popt, current_residuals
    except Exception:
        return None, None

def calculate_smoothed_iv(df, params_calls, params_puts, model='hyp', ticker=None):
    if df is None:
        return None
    df = df.copy()
    df['Smoothed_IV'] = np.nan
    if params_calls is None and params_puts is None:
        return df
    model_func = MODEL_CONFIG['hyp']['func']
    valid_mask = (df['Moneyness'].notna()) & (df['Years_to_Expiry'] > 0)
    call_mask = (df['Type'] == 'Call') & valid_mask
    put_mask = (df['Type'] == 'Put') & valid_mask
    try:
        if params_calls is not None and call_mask.any():
            x_calls = np.vstack((df.loc[call_mask, 'Moneyness'].values, df.loc[call_mask, 'Years_to_Expiry'].values))
            smoothed_calls = model_func(x_calls, *params_calls)
            df.loc[call_mask, 'Smoothed_IV'] = np.clip(smoothed_calls, 0, np.inf)
        if params_puts is not None and put_mask.any():
            x_puts = np.vstack((df.loc[put_mask, 'Moneyness'].values, df.loc[put_mask, 'Years_to_Expiry'].values))
            smoothed_puts = model_func(x_puts, *params_puts)
            df.loc[put_mask, 'Smoothed_IV'] = np.clip(smoothed_puts, 0, np.inf)
    except Exception:
        return df
    return df

def calculate_skew_slope_metrics(df, ticker, timestamp, r, q=0.0):
    if df is None:
        return pd.DataFrame(), pd.DataFrame()
    vol_surf_file = f'data/{timestamp}/vol_surf/vol_surf.csv'
    if not os.path.exists(vol_surf_file):
        return pd.DataFrame(), pd.DataFrame()
    try:
        vol_surf_df = pd.read_csv(vol_surf_file)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    vol_surf_df = vol_surf_df[vol_surf_df['Ticker'] == ticker]
    if vol_surf_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    call_params = vol_surf_df[vol_surf_df['Option_Type'] == 'Call']
    put_params = vol_surf_df[vol_surf_df['Option_Type'] == 'Put']
    if call_params.empty or put_params.empty:
        return pd.DataFrame(), pd.DataFrame()
    call_params = call_params.iloc[0][['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']].values
    put_params = put_params.iloc[0][['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']].values
    S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2 if 'Bid Stock' in df and 'Ask Stock' in df else df['Last Stock Price'].iloc[0]
    if S <= 0:
        return pd.DataFrame(), pd.DataFrame()
    moneyness_levels = [0.7, 0.8, 0.9]
    T_values = np.arange(0.1, 5.1, 0.1)
    skew_data = []
    slope_data = []
    for T in T_values:
        atm_iv_call = global_vol_model_hyp((1.0, T), *call_params)
        if np.isnan(atm_iv_call) or atm_iv_call <= 0:
            continue
        skew_moneyness = {}
        skew_moneyness_vol = {}
        for m in moneyness_levels:
            call_strike = S * m
            call_iv = global_vol_model_hyp((m, T), *call_params)
            call_mid = black_scholes_price(S, call_strike, T, r, call_iv, 'call', q)
            if np.isnan(call_mid) or call_mid <= 0:
                skew_moneyness[f'Skew_{int(m*100)}_Moneyness'] = np.nan
                skew_moneyness_vol[f'Skew_{int(m*100)}_Moneyness_Vol'] = np.nan
                continue
            def price_diff(K):
                moneyness_put = K / (S * np.exp((r - q) * T))
                sigma = global_vol_model_hyp((moneyness_put, T), *put_params)
                return black_scholes_price(S, K, T, r, sigma, 'put', q) - call_mid
            try:
                put_strike = brentq(price_diff, S * 0.01, S * 10.0, xtol=0.001, maxiter=200)
                put_moneyness = put_strike / (S * np.exp((r - q) * T))
                put_iv = global_vol_model_hyp((put_moneyness, T), *put_params)
                call_iv = global_vol_model_hyp((m, T), *call_params)
                skew_moneyness[f'Skew_{int(m*100)}_Moneyness'] = put_strike / S if put_strike > 0 else np.nan
                skew_moneyness_vol[f'Skew_{int(m*100)}_Moneyness_Vol'] = put_iv / call_iv if not np.isnan(put_iv) and not np.isnan(call_iv) and call_iv > 0 else np.nan
            except ValueError:
                skew_moneyness[f'Skew_{int(m*100)}_Moneyness'] = np.nan
                skew_moneyness_vol[f'Skew_{int(m*100)}_Moneyness_Vol'] = np.nan
        atm_iv_3m = global_vol_model_hyp((1.0, 0.25), *call_params)
        atm_iv_12m = global_vol_model_hyp((1.0, 1.0), *call_params)
        atm_ratio = atm_iv_12m / atm_iv_3m if not np.isnan(atm_iv_3m) and not np.isnan(atm_iv_12m) and atm_iv_3m > 0 else np.nan
        skew_data.append({
            'T': T,
            'Skew_90_Moneyness': skew_moneyness.get('Skew_90_Moneyness', np.nan),
            'Skew_80_Moneyness': skew_moneyness.get('Skew_80_Moneyness', np.nan),
            'Skew_70_Moneyness': skew_moneyness.get('Skew_70_Moneyness', np.nan),
            'Skew_90_Moneyness_Vol': skew_moneyness_vol.get('Skew_90_Moneyness_Vol', np.nan),
            'Skew_80_Moneyness_Vol': skew_moneyness_vol.get('Skew_80_Moneyness_Vol', np.nan),
            'Skew_70_Moneyness_Vol': skew_moneyness_vol.get('Skew_70_Moneyness_Vol', np.nan),
            'ATM_12m_3m_Ratio': atm_ratio,
            'Ticker': ticker
        })
        for m in moneyness_levels:
            for opt_type in ['Call', 'Put']:
                params = call_params if opt_type == 'Call' else put_params
                h_t = max(0.01 * T, 1e-4)
                iv_t = global_vol_model_hyp((m, T), *params)
                iv_t_plus = global_vol_model_hyp((m, T + h_t), *params)
                iv_t_minus = global_vol_model_hyp((m, max(T - h_t, 1e-6)), *params)
                if np.isnan(iv_t) or np.isnan(iv_t_plus) or np.isnan(iv_t_minus):
                    slope = np.nan
                else:
                    slope = (iv_t_plus - iv_t_minus) / (2 * h_t)
                slope_data.append({
                    'T': T,
                    'Moneyness': m,
                    'Type': opt_type,
                    'IV_Slope': slope,
                    'Ticker': ticker
                })
    skew_metrics_df = pd.DataFrame(skew_data)
    slope_metrics_df = pd.DataFrame(slope_data)
    skew_dir = f'data/{timestamp}/skew_metrics_yfinance'
    slope_dir = f'data/{timestamp}/slope_metrics_yfinance'
    os.makedirs(skew_dir, exist_ok=True)
    os.makedirs(slope_dir, exist_ok=True)
    if not skew_metrics_df.empty:
        skew_file = os.path.join(skew_dir, f'skew_metrics_yfinance_{ticker}.csv')
        skew_metrics_df.to_csv(skew_file, index=False)
    if not slope_metrics_df.empty:
        slope_file = os.path.join(slope_dir, f'skew_metrics_yfinance_{ticker}.csv')
        slope_metrics_df.to_csv(slope_file, index=False)
    return skew_metrics_df, slope_metrics_df

def process_ticker(ticker, timestamp, yields_dict, model, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau):
    data_file = f'data/{timestamp}/cleaned_yfinance/cleaned_yfinance_{ticker}.csv'
    raw_file = f'data/{timestamp}/raw_yfinance/raw_yfinance_{ticker}.csv'
    historic_file = f'data/{timestamp}/historic/historic_{ticker}.csv'
    processed_dir = f'data/{timestamp}/processed_yfinance'
    vol_surf_dir = f'data/{timestamp}/vol_surf'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(vol_surf_dir, exist_ok=True)
    if not os.path.exists(data_file) or not os.path.exists(raw_file):
        print(f"{ticker} (Call): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: missing data files)")
        print(f"{ticker} (Put): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: missing data files)")
        return None, None
    try:
        df = pd.read_csv(data_file, parse_dates=['Expiry'])
    except Exception as e:
        print(f"{ticker} (Call): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: failed to read data file)")
        print(f"{ticker} (Put): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: failed to read data file)")
        return None, None
    if df is None or df.empty:
        print(f"{ticker} (Call): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: empty data file)")
        print(f"{ticker} (Put): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: empty data file)")
        return None, None
    df['Ticker'] = ticker
    q = get_dividend_yield(ticker)
    rvol100d = np.nan
    if os.path.exists(historic_file):
        try:
            historic_df = pd.read_csv(historic_file, parse_dates=['Date'])
            if not historic_df.empty and 'Realised_Vol_Close_100' in historic_df.columns:
                rvol100d = historic_df['Realised_Vol_Close_100'].iloc[-1] / 100
        except Exception:
            pass
    timestamp_dt = datetime.strptime(timestamp, '%Y%m%d_%H%M')
    df['Expiry_dt'] = pd.to_datetime(df['Expiry'])
    df['Years_to_Expiry'] = (df['Expiry_dt'] - timestamp_dt).dt.days / 365.25
    S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2 if 'Bid Stock' in df and 'Ask Stock' in df else df['Last Stock Price'].iloc[0] if 'Last Stock Price' in df else np.nan
    if pd.isna(S) or S <= 0:
        print(f"{ticker} (Call): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: invalid stock price)")
        print(f"{ticker} (Put): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: invalid stock price)")
        return None, None
    df['Last Stock Price'] = S
    df = calculate_iv_binomial(df, yields_dict, q=q, default_r=0.05, max_workers=None)
    if df is None or df.empty:
        print(f"{ticker} (Call): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: IV calculation failed)")
        print(f"{ticker} (Put): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: IV calculation failed)")
        return None, None
    valid_iv_mask = df['IV_mid'].notna() & (df['IV_mid'] > 0)
    if not valid_iv_mask.any():
        print(f"{ticker} (Call): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: no valid IVs)")
        print(f"{ticker} (Put): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: no valid IVs)")
        return None, None
    df['Forward'] = df['Last Stock Price'] * np.exp((df['r'] - q) * df['Years_to_Expiry'])
    df['Moneyness'] = df['Strike'] / df['Forward']
    df['LogMoneyness'] = np.log(df['Moneyness'].where(df['Moneyness'] > 0, np.nan))
    df['Realised Vol 100d'] = rvol100d
    df['Ivol/Rvol100d Ratio'] = df['IV_mid'] / rvol100d if not np.isnan(rvol100d) else np.nan
    params_calls = None
    params_puts = None
    residuals_calls = None
    residuals_puts = None
    best_exp_max_short_calls = None
    best_exp_min_long_calls = None
    best_exp_max_short_puts = None
    best_exp_min_long_puts = None
    ticker_metrics = []
    for option_type in ['Call', 'Put']:
        df_type = df[df['Type'] == option_type].copy()
        if df_type.empty:
            print(f"{ticker} ({option_type}): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: no {option_type.lower()} options)")
            continue
        valid_type_mask = (df_type['Moneyness'].notna()) & (df_type['Years_to_Expiry'] > 0) & (df_type['IV_mid'].notna()) & (df_type['IV_mid'] > 0)
        valid_type_df = df_type[valid_type_mask]
        if len(valid_type_df) < 4:
            print(f"{ticker} ({option_type}): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid={len(valid_type_df)}, reason: insufficient valid data)")
            continue
        if len(df_type) < 10:
            exp_max_short = max(df_type['Years_to_Expiry'].min() + 0.3, 0.3)
            exp_min_long = min(df_type['Years_to_Expiry'].max() - 0.3, 1.8)
            params_short, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_short, exp_max=exp_max_short,
                                              mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                              model=model, option_type=option_type, p0=None, do_plot=False, max_nfev=10000, max_iterations=2)
            p0_long = params_short if params_short is not None else None
            params_long, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_long, exp_max=exp_max_long,
                                             mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                             model=model, option_type=option_type, p0=p0_long, do_plot=False, max_nfev=10000, max_iterations=2)
            p0_full = params_long if params_long is not None else p0_long
            params, residuals = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_full, exp_max=exp_max_full,
                                                mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                model=model, option_type=option_type, p0=p0_full, do_plot=False, max_nfev=20000, max_iterations=3)
        else:
            expiry_min = df_type['Years_to_Expiry'].min()
            expiry_max = df_type['Years_to_Expiry'].max()
            if expiry_max - expiry_min < 0.1:
                exp_max_short = max(expiry_min + 0.3, 0.3)
                exp_min_long = min(expiry_max - 0.3, 1.8)
                params_short, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_short, exp_max=exp_max_short,
                                                  mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                  model=model, option_type=option_type, p0=None, do_plot=False, max_nfev=10000, max_iterations=2)
                p0_long = params_short if params_short is not None else None
                params_long, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_long, exp_max=exp_max_long,
                                                 mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                 model=model, option_type=option_type, p0=p0_long, do_plot=False, max_nfev=10000, max_iterations=2)
                p0_full = params_long if params_long is not None else p0_long
                params, residuals = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_full, exp_max=exp_max_full,
                                                    mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                    model=model, option_type=option_type, p0=p0_full, do_plot=False, max_nfev=20000, max_iterations=3)
            else:
                median_expiry = df_type['Years_to_Expiry'].median()
                initial_guess = [df_type['Years_to_Expiry'].quantile(0.25), df_type['Years_to_Expiry'].quantile(0.75)]
                lb = [expiry_min + 0.02, median_expiry + 0.1]
                ub = [median_expiry - 0.1, expiry_max - 0.02]
                bounds_obj = Bounds(lb, ub)
                con = NonlinearConstraint(lambda x: x[1] - x[0], 0.1, np.inf)
                def objective(x):
                    exp_max_s, exp_min_l = x
                    return compute_p90(exp_max_s, exp_min_l, df, df_type, ticker, option_type, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau, model)
                try:
                    res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds_obj, constraints=[con], options={'maxiter': 5, 'disp': False})
                    if res.success and res.fun < np.inf:
                        best_exp_max_short = res.x[0]
                        best_exp_min_long = res.x[1]
                    else:
                        best_exp_max_short = max(df_type['Years_to_Expiry'].min() + 0.3, 0.3)
                        best_exp_min_long = min(df_type['Years_to_Expiry'].max() - 0.3, 1.8)
                except Exception:
                    best_exp_max_short = max(df_type['Years_to_Expiry'].min() + 0.3, 0.3)
                    best_exp_min_long = min(df_type['Years_to_Expiry'].max() - 0.3, 1.8)
                params_short, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_short, exp_max=best_exp_max_short,
                                                  mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                  model=model, option_type=option_type, p0=None, do_plot=False, max_nfev=10000, max_iterations=2)
                p0_long = params_short if params_short is not None else None
                params_long, _ = fit_vol_surface(df, ticker=ticker, exp_min=best_exp_min_long, exp_max=exp_max_long,
                                                 mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                 model=model, option_type=option_type, p0=p0_long, do_plot=False, max_nfev=10000, max_iterations=2)
                p0_full = params_long if params_long is not None else p0_long
                params, residuals = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_full, exp_max=exp_max_full,
                                                    mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                    model=model, option_type=option_type, p0=p0_full, do_plot=False, max_nfev=20000, max_iterations=3)
        if option_type == 'Call':
            params_calls = params
            residuals_calls = residuals
            best_exp_max_short_calls = best_exp_max_short
            best_exp_min_long_calls = best_exp_min_long
        else:
            params_puts = params
            residuals_puts = residuals
            best_exp_max_short_puts = best_exp_max_short
            best_exp_min_long_puts = best_exp_min_long
        valid_mask = (df['IV_mid'].notna() & df['Smoothed_IV'].notna() &
                      (df['IV_mid'] > 0) & (df['Smoothed_IV'] > 0))
        type_mask = df['Type'] == option_type
        df_type = df[valid_mask & type_mask].copy()
        atm_iv = np.nan
        if params is not None:
            try:
                atm_iv = global_vol_model_hyp(np.array([[1.0], [1.0]]), *params)
                atm_iv = float(atm_iv) if isinstance(atm_iv, (np.ndarray, float)) else np.nan
                if np.isnan(atm_iv) or atm_iv <= 0.01:
                    atm_iv = np.nan
            except Exception:
                atm_iv = np.nan
        df_type['rel_error_atm_pct'] = np.where((df_type['IV_mid'].notna()) &
                                                (df_type['Smoothed_IV'].notna()) &
                                                (atm_iv > 0),
                                                abs((df_type['IV_mid'] - df_type['Smoothed_IV']) / atm_iv) * 100,
                                                np.nan)
        atm_candidates = df_type[
            (abs(df_type['Years_to_Expiry'] - 1) <= 0.5) &
            (abs(df_type['Moneyness'] - 1) <= 0.1)
        ].copy()
        one_yr_atm_residual = np.nan
        atm_dist_t = np.nan
        atm_dist_m = np.nan
        reason = ""
        if not atm_candidates.empty:
            atm_candidates['dist_to_atm'] = (atm_candidates['Years_to_Expiry'] - 1)**2 + (atm_candidates['Moneyness'] - 1)**2
            closest_idx = atm_candidates['dist_to_atm'].idxmin()
            one_yr_atm_residual = df_type.at[closest_idx, 'rel_error_atm_pct']
            atm_dist_t = df_type.at[closest_idx, 'Years_to_Expiry']
            atm_dist_m = df_type.at[closest_idx, 'Moneyness']
        else:
            if not df_type.empty:
                df_type['dist_to_atm'] = (df_type['Years_to_Expiry'] - 1)**2 + (df_type['Moneyness'] - 1)**2
                closest_idx = df_type['dist_to_atm'].idxmin()
                one_yr_atm_residual = df_type.at[closest_idx, 'rel_error_atm_pct']
                atm_dist_t = df_type.at[closest_idx, 'Years_to_Expiry']
                atm_dist_m = df_type.at[closest_idx, 'Moneyness']
            else:
                reason = "no valid options after filtering"
        p90_rel_error = df_type['rel_error_atm_pct'].quantile(0.9) if not df_type.empty else np.nan
        if not np.isnan(p90_rel_error) and p90_rel_error > 1000:
            p90_rel_error = np.nan
            reason = "excessive P90 error"
        if pd.isna(one_yr_atm_residual) or pd.isna(p90_rel_error):
            if not reason:
                reason = "invalid IV or smoothed IV" if df_type.empty else "fitting failed"
            print(f"{ticker} ({option_type}): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid={len(df_type)}, reason: {reason})")
        else:
            print(f"{ticker} ({option_type}): 1yr ATM rel error = {one_yr_atm_residual:.2f}% (closest: T={atm_dist_t:.2f}, M={atm_dist_m:.2f}), "
                  f"P90 rel error (ATM-norm) = {p90_rel_error:.2f}% (n_valid={len(df_type)})")
        metrics_df = pd.DataFrame({
            'Ticker': [ticker],
            'Timestamp': [timestamp],
            'Option_Type': [option_type],
            'One_Yr_ATM_Rel_Error_Pct': [one_yr_atm_residual],
            'P90_Rel_Error_Pct': [p90_rel_error],
            'N_Valid_Options': [len(df_type)],
            'ATM_Dist_T': [atm_dist_t],
            'ATM_Dist_M': [atm_dist_m],
            'ATM_IV': [atm_iv],
            'Best_Exp_Max_Short': [best_exp_max_short_calls if option_type == 'Call' else best_exp_max_short_puts],
            'Best_Exp_Min_Long': [best_exp_min_long_calls if option_type == 'Call' else best_exp_min_long_puts]
        })
        ticker_metrics.append(metrics_df)
    df = calculate_smoothed_iv(df, params_calls, params_puts, model, ticker=ticker)
    if df is None:
        print(f"{ticker} (Call): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: smoothed IV calculation failed)")
        print(f"{ticker} (Put): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: smoothed IV calculation failed)")
        return None, None
    df['TotalVariance'] = df['Smoothed_IV']**2 * df['Years_to_Expiry']
    df['TotalVariance'] = df['TotalVariance'].fillna(np.nan)
    r = df['r'].iloc[0] if 'r' in df else 0.05
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
    df = df.reindex(columns=output_columns, fill_value=np.nan)
    output_file = f'{processed_dir}/processed_yfinance_{ticker}.csv'
    df.to_csv(output_file, index=False)
    df.to_json(f'{processed_dir}/processed_yfinance_{ticker}.json', orient='records', date_format='iso')
    param_names = ['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']
    param_dfs = []
    if params_calls is not None:
        param_df_calls = pd.DataFrame([params_calls], columns=param_names)
        param_df_calls['Ticker'] = ticker
        param_df_calls['Model'] = model
        param_df_calls['Residuals'] = residuals_calls if residuals_calls is not None else np.nan
        param_df_calls['Timestamp'] = timestamp
        param_df_calls['Option_Type'] = 'Call'
        param_dfs.append(param_df_calls)
    if params_puts is not None:
        param_df_puts = pd.DataFrame([params_puts], columns=param_names)
        param_df_puts['Ticker'] = ticker
        param_df_puts['Model'] = model
        param_df_puts['Residuals'] = residuals_puts if residuals_puts is not None else np.nan
        param_df_puts['Timestamp'] = timestamp
        param_df_puts['Option_Type'] = 'Put'
        param_dfs.append(param_df_puts)
    param_df = pd.concat(param_dfs, ignore_index=True) if param_dfs else pd.DataFrame()
    skew_metrics_df, slope_metrics_df = calculate_skew_slope_metrics(df, ticker, timestamp, r, q)
    skew_file = f'{processed_dir}/skew_metrics_yfinance_{ticker}.csv'
    slope_file = f'{processed_dir}/slope_metrics_yfinance_{ticker}.csv'
    skew_metrics_df.to_csv(skew_file, index=False)
    slope_metrics_df.to_csv(slope_file, index=False)
    return pd.concat(ticker_metrics, ignore_index=True), param_df

def process_volumes(timestamp):
    print(f"Starting process_volumes for timestamp {timestamp}")
    if not os.path.exists('tickers.txt'):
        print("tickers.txt not found, skipping processing")
        return
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    timestamp_dt = datetime.strptime(timestamp, '%Y%m%d_%H%M')
    timestamp_date = timestamp_dt.strftime('%Y-%m-%d')
    yields_dict = fetch_treasury_yields(timestamp_date)
    model = 'hyp'
    exp_min_short = 0
    exp_max_long = np.inf
    exp_min_full = 0
    exp_max_full = np.inf
    mon_min = 0
    mon_max = np.inf
    extrap_tau = None
    all_metrics = []
    all_params = []
    max_workers = 2
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_ticker, ticker, timestamp, yields_dict, model, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau): ticker for ticker in tickers}
        for future in as_completed(futures):  # Removed timeout
            ticker = futures[future]
            try:
                metrics_df, param_df = future.result()
                if metrics_df is not None and not metrics_df.empty:
                    all_metrics.append(metrics_df)
                if not param_df.empty:
                    all_params.append(param_df)
            except Exception as e:
                print(f"{ticker} (Call): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: processing error: {str(e)})")
                print(f"{ticker} (Put): 1yr ATM rel error = nan% (closest: T=nan, M=nan), P90 rel error (ATM-norm) = nan% (n_valid=0, reason: processing error: {str(e)})")
    if all_metrics:
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        metrics_file = f'data/{timestamp}/processed_yfinance/fit_metrics_yfinance_all.csv'
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        all_metrics_df.to_csv(metrics_file, index=False)
    if all_params:
        all_params_df = pd.concat(all_params, ignore_index=True)
        vol_surf_file = f'data/{timestamp}/vol_surf/vol_surf.csv'
        os.makedirs(os.path.dirname(vol_surf_file), exist_ok=True)
        all_params_df.to_csv(vol_surf_file, index=False)
    dates_file = 'data/dates.json'
    if os.path.exists(dates_file):
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    else:
        dates = []
    if timestamp not in dates:
        dates.append(timestamp)
        dates.sort(reverse=True)
    os.makedirs(os.path.dirname(dates_file), exist_ok=True)
    with open(dates_file, 'w') as f:
        json.dump(dates, f)

if __name__ == '__main__':
    import sys
    timestamp = sys.argv[1] if len(sys.argv) > 1 else '20250919_2126'
    process_volumes(timestamp)
