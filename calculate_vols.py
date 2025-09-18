import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import json
from scipy.optimize import brentq, least_squares
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
        hist = tnx.history(start=(target_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d'), end=(target_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
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

def binomial_tree_price(S, K, T, r, sigma, option_type='call', q=0, N=100):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0 or sigma > 4.0 or T > 10.0:
        return np.nan
    try:
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        if u - d < 1e-8:
            return np.nan
        p = (np.exp((r - q) * dt) - d) / (u - d)
        if p <= 0 or p >= 1 or np.isnan(p):
            return np.nan
        discount = np.exp(-r * dt)
        stock = np.zeros((N + 1, N + 1))
        stock[0, 0] = S
        for i in range(1, N + 1):
            stock[i, 0] = stock[i-1, 0] * u
            if stock[i, 0] > 1e308 or np.isinf(stock[i, 0]):
                return np.nan
            for j in range(1, i + 1):
                stock[i, j] = stock[i-1, j-1] * d
        option = np.zeros((N + 1, N + 1))
        for j in range(N + 1):
            if option_type == 'call':
                option[N, j] = max(0, stock[N, j] - K)
            else:
                option[N, j] = max(0, K - stock[N, j])
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                option[i, j] = discount * (p * option[i + 1, j] + (1 - p) * option[i + 1, j + 1])
                if np.isinf(option[i, j]) or np.isnan(option[i, j]):
                    return np.nan
        return option[0, 0]
    except Exception:
        return np.nan

def implied_vol_binomial(S, K, T, r, market_price, option_type='call', q=0, N=100, iv_bs=None):
    if market_price <= 0 or np.isnan(market_price):
        return np.nan, 'invalid_market_price'
    price_low = binomial_tree_price(S, K, T, r, 0.05, option_type, q, N)
    price_high = binomial_tree_price(S, K, T, r, 4.0, option_type, q, N)
    if np.isnan(price_low) or np.isnan(price_high):
        return np.nan, 'invalid_binomial_price'
    if market_price < price_low or market_price > price_high:
        return np.nan, 'price_out_of_range'
    initial_guesses = [0.2, 0.5, 0.8, 1.2, 1.8, 2.5, 3.0, 4.0]
    if iv_bs is not None and not np.isnan(iv_bs):
        initial_guesses.insert(0, iv_bs)
    for guess in initial_guesses:
        def objective(sigma):
            price = binomial_tree_price(S, K, T, r, sigma, option_type, q, N)
            return price - market_price if not np.isnan(price) else np.inf
        try:
            iv = brentq(objective, 0.05, 4.0, maxiter=200)
            model_price = binomial_tree_price(S, K, T, r, iv, option_type, q, N)
            if abs(model_price - market_price) < 0.001 and iv > 0:
                return iv, 'verified'
            return np.nan, 'inaccurate'
        except Exception:
            continue
    return np.nan, 'convergence_failed'

def _compute_iv_for_row(args):
    row, yields_dict, default_r, q, N = args
    if not isinstance(row, dict):
        return np.nan, default_r, np.nan, np.nan, np.nan
    T = np.float64(row.get('Years_to_Expiry', 0))
    r = interpolate_r(T, yields_dict)
    bid = row.get('Bid', 0)
    ask = row.get('Ask', 0)
    market_price = np.float64((bid + ask) / 2) if bid > 0 and ask > 0 else np.nan
    iv_bs = np.nan
    iv_binomial = np.nan
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
                iv_binomial = iv_bs#, _ = iv_bs#implied_vol_binomial(S, K, T, r, market_price, type_, q, N, iv_bs)
    return T, r, market_price, iv_bs, iv_binomial

def calculate_iv_binomial(options_df, yields_dict, q=0, default_r=0.05, max_workers=None, binomial_steps=100):
    options_df = options_df.copy()
    if len(options_df) == 0:
        return options_df
    rows = [row.to_dict() for _, row in options_df.iterrows()]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_compute_iv_for_row, (row, yields_dict, default_r, q, binomial_steps)): idx
                         for idx, row in enumerate(rows)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                T, r, market_price, iv_bs, iv_binomial = future.result()
                results.append((idx, T, r, market_price, iv_bs, iv_binomial))
            except Exception as e:
                results.append((idx, np.nan, default_r, np.nan, np.nan, np.nan))
    results.sort(key=lambda x: x[0])
    for idx, T, r, market_price, iv_bs, iv_binomial in results:
        options_df.at[idx, 'T'] = T
        options_df.at[idx, 'r'] = r
        options_df.at[idx, 'market_price'] = market_price
        options_df.at[idx, 'IV_mid'] = iv_bs
        options_df.at[idx, 'IV_mid_binomial'] = iv_binomial
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

def fit_vol_surface(df, ticker=None, exp_min=0.2, exp_max=np.inf, mon_min=0, mon_max=np.inf, extrap_tau=None, model='hyp', option_type='Call'):
    df = df.copy()
    if model not in MODEL_CONFIG:
        raise ValueError(f"Model must be one of {list(MODEL_CONFIG.keys())}")
    df = df[(df['Type'] == option_type) & (df['Years_to_Expiry'] > exp_min) & (df['Years_to_Expiry'] < exp_max) &
            (df['Moneyness'] > mon_min) & (df['Moneyness'] < mon_max) & (df['IV_mid'].notna())]
    if ticker:
        df_ticker = df[df['Ticker'] == ticker]
        if df_ticker.empty:
            print(f"No valid data points for {ticker} ({option_type}) after filtering.")
            return None, None
        print(f"Data points for {ticker} ({option_type}): {len(df_ticker)}, Moneyness range: {df_ticker['Moneyness'].min():.2f}–{df_ticker['Moneyness'].max():.2f}, "
              f"Expiry range: {df_ticker['Years_to_Expiry'].min():.2f}–{df_ticker['Years_to_Expiry'].max():.2f}, "
              f"IV range: {df_ticker['IV_mid'].min():.2f}–{df_ticker['IV_mid'].max():.2f}")
        return fit_single_ticker(df_ticker, model)
    else:
        results = {}
        for tick, group in df.groupby('Ticker'):
            if group.empty:
                print(f"No valid data points for {tick} ({option_type}) after filtering.")
                results[tick] = (None, None)
                continue
            print(f"Data points for {tick} ({option_type}): {len(group)}, Moneyness range: {group['Moneyness'].min():.2f}–{group['Moneyness'].max():.2f}, "
                  f"Expiry range: {group['Years_to_Expiry'].min():.2f}–{group['Years_to_Expiry'].max():.2f}, "
                  f"IV range: {group['IV_mid'].min():.2f}–{group['IV_mid'].max():.2f}")
            results[tick] = fit_single_ticker(group, model)
        return results

def fit_single_ticker(df, model):
    df = df.copy()
    if len(df) < 4:
        print(f"Insufficient data points ({len(df)}) for ticker {df['Ticker'].iloc[0] if not df.empty else 'unknown'}.")
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
        print(f"Insufficient data points after filtering ({len(df)}) for ticker {df['Ticker'].iloc[0] if not df.empty else 'unknown'}.")
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
            print(f"Iteration {iteration + 1}: Residuals {new_residuals:.6f} (change {new_residuals - prev_residuals:.6f})")
            if abs(new_residuals - prev_residuals) < tolerance:
                print(f"Converged after {iteration + 1} iterations.")
                break
            prev_residuals = new_residuals
        ref_point = np.array([[1.0], [1.0]])
        IV_ref = model_func(ref_point, *popt)
        if isinstance(IV_ref, np.ndarray):
            IV_ref = float(IV_ref.item()) if IV_ref.size == 1 else float(IV_ref[0, 0])
        if IV_ref <= 0 or np.isnan(IV_ref):
            IV_ref = 1.0
        res = (IV - model_func(xdata, *popt)) / IV_ref * 100
        residuals_sum = np.sum(w * res**2)
        return popt, residuals_sum
    except Exception as e:
        print(f"Fit failed for ticker {df['Ticker'].iloc[0] if not df.empty else 'unknown'} with model {model}: {e}")
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

def calculate_skew_slope_metrics(df, ticker, timestamp, r, q=0.0):
    vol_surf_file = f'data/{timestamp}/vol_surf/vol_surf.csv'
    if not os.path.exists(vol_surf_file):
        print(f"No volatility surface parameters found for {ticker} at {vol_surf_file}")
        return pd.DataFrame(), pd.DataFrame()
    try:
        vol_surf_df = pd.read_csv(vol_surf_file)
    except Exception as e:
        print(f"Error reading {vol_surf_file}: {e}")
        return pd.DataFrame(), pd.DataFrame()
    vol_surf_df = vol_surf_df[vol_surf_df['Ticker'] == ticker]
    if vol_surf_df.empty:
        print(f"No volatility surface parameters for {ticker}")
        return pd.DataFrame(), pd.DataFrame()
    call_params = vol_surf_df[vol_surf_df['Option_Type'] == 'Call']
    put_params = vol_surf_df[vol_surf_df['Option_Type'] == 'Put']
    if call_params.empty or put_params.empty:
        print(f"Missing parameters for {ticker}: Calls {'missing' if call_params.empty else 'present'}, Puts {'missing' if put_params.empty else 'present'}")
        return pd.DataFrame(), pd.DataFrame()
    call_params = call_params.iloc[0][['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']].values
    put_params = put_params.iloc[0][['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']].values
    S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2
    if S <= 0:
        print(f"Invalid stock price {S} for {ticker}")
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
        print(f"Skew metrics for {ticker} saved to {skew_file}")
    if not slope_metrics_df.empty:
        slope_file = os.path.join(slope_dir, f'slope_metrics_yfinance_{ticker}.csv')
        slope_metrics_df.to_csv(slope_file, index=False)
        print(f"Slope metrics for {ticker} saved to {slope_file}")
    return skew_metrics_df, slope_metrics_df

def process_volumes(timestamp):
    if not os.path.exists('tickers.txt'):
        print("tickers.txt not found")
        return
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
   
    clean_dir = f'data/{timestamp}/cleaned_yfinance'
    raw_dir = f'data/{timestamp}/raw_yfinance'
    processed_dir = f'data/{timestamp}/processed_yfinance'
    skew_dir = f'data/{timestamp}/skew_metrics_yfinance'
    slope_dir = f'data/{timestamp}/slope_metrics_yfinance'
    historic_dir = f'data/{timestamp}/historic'
    vol_surf_dir = f'data/{timestamp}/vol_surf'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(skew_dir, exist_ok=True)
    os.makedirs(slope_dir, exist_ok=True)
    os.makedirs(historic_dir, exist_ok=True)
    os.makedirs(vol_surf_dir, exist_ok=True)
   
    timestamp_dt = datetime.strptime(timestamp, '%Y%m%d_%H%M')
    timestamp_date = timestamp_dt.strftime('%Y-%m-%d')
    yields_dict = fetch_treasury_yields(timestamp_date)
    print(f"Using treasury yields for {timestamp_date}")
   
    model = 'hyp'
    exp_min = 0.3
    exp_max = np.inf
    mon_min = 0
    mon_max = np.inf
    extrap_tau = None
   
    # Initialize metrics collection for all tickers
    all_metrics = []
   
    for ticker in tickers:
        data_file = os.path.join(clean_dir, f'cleaned_yfinance_{ticker}.csv')
        raw_file = os.path.join(raw_dir, f'raw_yfinance_{ticker}.csv')
        historic_file = os.path.join(historic_dir, f'historic_{ticker}.csv')
       
        if not os.path.exists(data_file):
            print(f"No cleaned file for {ticker} in {clean_dir}")
            continue
        if not os.path.exists(raw_file):
            print(f"No raw file for {ticker} in {raw_dir}")
            continue
       
        df = pd.read_csv(data_file, parse_dates=['Expiry'])
        if df.empty:
            print(f"No data for ticker {ticker}")
            continue
       
        q = get_dividend_yield(ticker)
        print(f"Dividend yield for {ticker}: {q}")
       
        rvol100d = np.nan
        if os.path.exists(historic_file):
            try:
                historic_df = pd.read_csv(historic_file, parse_dates=['Date'])
                if not historic_df.empty and 'Realised_Vol_Close_100' in historic_df.columns:
                    rvol100d = historic_df['Realised_Vol_Close_100'].iloc[-1] / 100
            except Exception as e:
                print(f"Error reading historic file for {ticker}: {e}")
       
        df['Expiry_dt'] = pd.to_datetime(df['Expiry'])
        df['Years_to_Expiry'] = (df['Expiry_dt'] - timestamp_dt).dt.days / 365.25
        S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2
        df['Last Stock Price'] = S
       
        df = calculate_iv_binomial(df, yields_dict, q=q, default_r=0.05, max_workers=None, binomial_steps=100)
        if df.empty:
            print(f"Failed to calculate IV for {ticker}")
            continue
       
        df['Forward'] = df['Last Stock Price'] * np.exp((df['r'] - q) * df['Years_to_Expiry'])
        df['Moneyness'] = df['Strike'] / df['Forward']
        df['LogMoneyness'] = np.log(df['Moneyness'].where(df['Moneyness'] > 0, np.nan))
        df['Realised Vol 100d'] = rvol100d
        df['Ivol/Rvol100d Ratio'] = df['IV_mid'] / rvol100d if not np.isnan(rvol100d) else np.nan
       
        params_calls, residuals_calls = fit_vol_surface(df, ticker=ticker, exp_min=exp_min, exp_max=exp_max, mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau, model=model, option_type='Call')
        print(f"Parameters for {ticker} (Calls, {model} model, {exp_min} < Years_to_Expiry < {exp_max}, {mon_min} < Moneyness < {mon_max}):",
              params_calls,
              f"Residuals: {residuals_calls:.4e}" if residuals_calls is not None else "N/A")
       
        params_puts, residuals_puts = fit_vol_surface(df, ticker=ticker, exp_min=exp_min, exp_max=exp_max, mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau, model=model, option_type='Put')
        print(f"Parameters for {ticker} (Puts, {model} model, {exp_min} < Years_to_Expiry < {exp_max}, {mon_min} < Moneyness < {mon_max}):",
              params_puts,
              f"Residuals: {residuals_puts:.4e}" if residuals_puts is not None else "N/A")
       
        df = calculate_smoothed_iv(df, params_calls, params_puts, model)
       
        # Compute ATM-normalized relative errors for calls and puts
        valid_mask = (df['IV_mid'].notna() & df['Smoothed_IV'].notna() &
                      (df['IV_mid'] > 0) & (df['Smoothed_IV'] > 0))
        for opt_type in ['Call', 'Put']:
            type_mask = (df['Type'] == opt_type) & valid_mask
            df_type = df[type_mask].copy()
            params = params_calls if opt_type == 'Call' else params_puts
            atm_iv = np.nan
            if params is not None:
                try:
                    atm_iv = global_vol_model_hyp(np.array([[1.0], [1.0]]), *params)
                    atm_iv = float(atm_iv.item()) if isinstance(atm_iv, np.ndarray) else float(atm_iv)
                    if np.isnan(atm_iv) or atm_iv <= 0:
                        atm_iv = np.nan
                except Exception:
                    atm_iv = np.nan
            df_type['rel_error_atm_pct'] = np.where(type_mask & (atm_iv > 0),
                                                    abs((df_type['IV_mid'] - df_type['Smoothed_IV']) / atm_iv) * 100,
                                                    np.nan)
            atm_candidates = df_type[
                (abs(df_type['Years_to_Expiry'] - 1) <= 0.25) &  # Within ~3 months
                (abs(df_type['Moneyness'] - 1) <= 0.05)          # Within 5% moneyness
            ].copy()
            one_yr_atm_residual = np.nan
            atm_details = " (no close match)"
            if not atm_candidates.empty:
                atm_candidates['dist_to_atm'] = (abs(atm_candidates['Years_to_Expiry'] - 1) +
                                                abs(atm_candidates['Moneyness'] - 1))
                closest_idx = atm_candidates['dist_to_atm'].idxmin()
                one_yr_atm_residual = df_type.at[closest_idx, 'rel_error_atm_pct']
                atm_details = f" (closest: T={df_type.at[closest_idx, 'Years_to_Expiry']:.2f}, " \
                              f"M={df_type.at[closest_idx, 'Moneyness']:.2f})"
               
            p90_rel_error = df_type['rel_error_atm_pct'].quantile(0.9) if type_mask.sum() > 0 else np.nan
            print(f"{ticker} ({opt_type}): 1yr ATM rel error = {one_yr_atm_residual:.2f}%{atm_details}, "
                  f"P90 rel error (ATM-norm) = {p90_rel_error:.2f}% (n_valid={type_mask.sum()})")
            metrics_df = pd.DataFrame({
                'Ticker': [ticker],
                'Timestamp': [timestamp],
                'Option_Type': [opt_type],
                'One_Yr_ATM_Rel_Error_Pct': [one_yr_atm_residual],
                'P90_Rel_Error_Pct': [p90_rel_error],
                'N_Valid_Options': [type_mask.sum()],
                'ATM_Dist_T': [abs(df_type.at[closest_idx, 'Years_to_Expiry'] - 1)
                               if 'closest_idx' in locals() else np.nan],
                'ATM_Dist_M': [abs(df_type.at[closest_idx, 'Moneyness'] - 1)
                               if 'closest_idx' in locals() else np.nan],
                'ATM_IV': [atm_iv]
            })
            all_metrics.append(metrics_df)
       
        df['TotalVariance'] = df['Smoothed_IV']**2 * df['Years_to_Expiry']
        df['TotalVariance'] = df['TotalVariance'].fillna(np.nan)
        r = df['r'].iloc[0]
       
        output_columns = [
            'Ticker', 'Contract Name', 'Type', 'Expiry', 'Strike', 'Moneyness', 'Bid', 'Ask', 'Volume', 'Open Interest',
            'Bid Stock', 'Ask Stock', 'Last Stock Price', 'Implied Volatility', 'Expiry_dt', 'Years_to_Expiry', 'Forward',
            'LogMoneyness', 'IV_bid', 'IV_ask', 'IV_mid', 'IV_spread', 'Delta', 'Ivol/Rvol100d Ratio', 'Smoothed_IV',
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
        print(f"Processed data for {ticker} saved to {processed_dir}")
       
        param_names = ['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']
        vol_surf_file = os.path.join(vol_surf_dir, 'vol_surf.csv')
        param_dfs = []
        if params_calls is not None:
            param_df_calls = pd.DataFrame([params_calls], columns=param_names)
            param_df_calls['Ticker'] = ticker
            param_df_calls['Model'] = model
            param_df_calls['Residuals'] = residuals_calls
            param_df_calls['Timestamp'] = timestamp
            param_df_calls['Option_Type'] = 'Call'
            param_dfs.append(param_df_calls)
        if params_puts is not None:
            param_df_puts = pd.DataFrame([params_puts], columns=param_names)
            param_df_puts['Ticker'] = ticker
            param_df_puts['Model'] = model
            param_df_puts['Residuals'] = residuals_puts
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
        skew_metrics_df, slope_metrics_df = calculate_skew_slope_metrics(df, ticker, timestamp, r, q=q)
   
    # Save all metrics after processing all tickers
    if all_metrics:
        metrics_file = os.path.join(processed_dir, 'fit_metrics_yfinance_all.csv')
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        if os.path.exists(metrics_file):
            try:
                existing_metrics = pd.read_csv(metrics_file)
                all_metrics_df = pd.concat([existing_metrics, all_metrics_df], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing {metrics_file}: {e}")
        all_metrics_df.to_csv(metrics_file, index=False)
        print(f"All fit metrics saved to {metrics_file}")
   
    dates_file = 'data/dates.json'
    if os.path.exists(dates_file):
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    else:
        dates = []
    if timestamp not in dates:
        dates.append(timestamp)
        dates.sort(reverse=True)
    with open(dates_file, 'w') as f:
        json.dump(dates, f)
    print(f"Updated dates list in {dates_file}")
