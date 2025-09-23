DEBUG = False  # Set to True to save plots
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from scipy.optimize import brentq, least_squares
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import warnings
import optuna
import optuna.logging
import plotly.graph_objects as go
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    except Exception as e:
        print(f"DEBUG: Failed to fetch dividend yield for {ticker}: {e}")
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
        if hist.empty:
            print(f"DEBUG: No treasury yield data for {date_str}, using fallback yields")
            return FALLBACK_YIELDS
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        if target_date in hist.index:
            row = hist.loc[target_date]
        elif use_latest_if_missing:
            prior = hist[hist.index <= target_date]
            if prior.empty:
                print(f"DEBUG: No prior treasury yield data for {date_str}, using fallback yields")
                return FALLBACK_YIELDS
            row = prior.iloc[-1]
        else:
            print(f"DEBUG: No treasury yield data for {date_str}, using fallback yields")
            return FALLBACK_YIELDS
        tnx_yield = row['Close'] / 100
        yields = {}
        base_yields = FALLBACK_YIELDS
        base_10yr = base_yields['10 Yr']
        for maturity, base_yield in base_yields.items():
            yields[maturity] = np.float64(tnx_yield * (base_yield / base_10yr))
        return yields
    except Exception as e:
        print(f"DEBUG: Error fetching treasury yields for {date_str}: {e}")
        return FALLBACK_YIELDS
def interpolate_r(T, yields_dict):
    if not yields_dict:
        print(f"DEBUG: No treasury yields provided, using default rate 0.05")
        return np.float64(0.05)
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30], dtype=np.float64)
    rates = np.array([yields_dict.get(k, np.nan) for k in FALLBACK_YIELDS.keys()], dtype=np.float64)
    valid = ~np.isnan(rates)
    maturities = maturities[valid]
    rates = rates[valid]
    if len(maturities) == 0:
        print(f"DEBUG: No valid treasury rates, using default rate 0.05")
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
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    except Exception as e:
        print(f"DEBUG: Black-Scholes price calculation failed: {e}")
        return np.nan
def implied_vol_bsm(S, K, T, r, market_price, option_type='call', q=0):
    def objective(sigma):
        price = black_scholes_price(S, K, T, r, sigma, option_type, q)
        return price - market_price if not np.isnan(price) else np.inf
    if np.isnan(market_price) or market_price <= 0:
        return np.nan, 'invalid_market_price'
    m = S / K
    if S <= 0 or K <= 0 or T <= 0.001 or r < 0 or m < 0.1 or m > 5.0:
        return np.nan, 'invalid_inputs'
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if market_price < intrinsic + 0.01:
        return np.nan, 'market_price_below_intrinsic'
    try:
        iv = brentq(objective, 0.0001, 10.0, xtol=1e-6, maxiter=30)
        model_price = black_scholes_price(S, K, T, r, iv, option_type, q)
        if abs(model_price - market_price) < 0.01 and iv > 0:
            return iv, 'verified'
        return np.nan, 'inaccurate'
    except Exception:
        return np.nan, 'convergence'
def _compute_iv_for_row(args):
    row, yields_dict, default_r, q = args
    if not isinstance(row, dict):
        return np.nan, default_r, np.nan, np.nan, 'invalid_row'
    T = np.float64(row.get('Years_to_Expiry', 0))
    if T <= 0.001:
        return T, default_r, np.nan, np.nan, 'invalid_expiry'
    r = interpolate_r(T, yields_dict)
    bid = row.get('Bid', 0)
    ask = row.get('Ask', 0)
    market_price = np.float64((bid + ask) / 2) if bid > 0 and ask > 0 else np.nan
    iv_bs = np.nan
    status = 'valid'
    if pd.isna(market_price) or market_price <= 0.01:
        status = 'invalid_market_price'
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
                market_price = np.maximum(market_price, intrinsic + 0.01)
                iv_bs, status = implied_vol_bsm(S, K, T, r, market_price, type_, q)
    return T, r, market_price, iv_bs, status
def calculate_iv_binomial(options_df, yields_dict, q=0, default_r=0.05, max_workers=4):
    options_df = options_df.copy()
    if len(options_df) == 0:
        return options_df
    valid_mask = (options_df['Bid'] > 0) & (options_df['Ask'] > 0) & (options_df['Years_to_Expiry'] > 0.001) & \
                 (options_df['Last Stock Price'] > 0) & (options_df['Strike'] > 0) & \
                 (options_df['Strike'] / options_df['Last Stock Price']).between(0.1, 5.0)
    options_df = options_df[valid_mask]
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
                T, r, market_price, iv_bs, status = future.result()
                results.append((idx, T, r, market_price, iv_bs, status))
            except Exception:
                results.append((idx, np.nan, default_r, np.nan, np.nan, 'error'))
    results.sort(key=lambda x: x[0])
    for idx, T, r, market_price, iv_bs, status in results:
        options_df.at[idx, 'T'] = T
        options_df.at[idx, 'r'] = r
        options_df.at[idx, 'market_price'] = market_price
        options_df.at[idx, 'IV_mid'] = iv_bs
        options_df.at[idx, 'IV_status'] = status
    return options_df
def calculate_smoothed_iv(df, params_calls, params_puts, model='hyp'):
    df = df.copy()
    if 'Smoothed_IV' not in df.columns:
        df['Smoothed_IV'] = np.nan
    if params_calls is None and params_puts is None:
        print(f"DEBUG: Both params_calls and params_puts are None in calculate_smoothed_iv")
        return df
    model_func = MODEL_CONFIG['hyp']['func']
    valid_mask = (df['Moneyness'].notna()) & (df['Years_to_Expiry'] > 0.001)
    call_mask = (df['Type'] == 'Call') & valid_mask
    put_mask = (df['Type'] == 'Put') & valid_mask
    try:
        if params_calls is not None and call_mask.any():
            x_calls = np.vstack((df.loc[call_mask, 'Moneyness'].values, df.loc[call_mask, 'Years_to_Expiry'].values))
            smoothed_calls = model_func(x_calls, *params_calls)
            if not np.all(np.isnan(smoothed_calls)):
                df.loc[call_mask, 'Smoothed_IV'] = np.clip(smoothed_calls, 0.01, 5.0)
            else:
                print(f"DEBUG: Smoothed calls are all NaN")
        if params_puts is not None and put_mask.any():
            x_puts = np.vstack((df.loc[put_mask, 'Moneyness'].values, df.loc[put_mask, 'Years_to_Expiry'].values))
            smoothed_puts = model_func(x_puts, *params_puts)
            if not np.all(np.isnan(smoothed_puts)):
                df.loc[put_mask, 'Smoothed_IV'] = np.clip(smoothed_puts, 0.01, 5.0)
            else:
                print(f"DEBUG: Smoothed puts are all NaN")
    except Exception as e:
        print(f"DEBUG: Error in calculate_smoothed_iv: {e}")
    return df
def fit_single_ticker(df, model, p0=None, max_nfev=10000, max_iterations=3, ticker=None, option_type='Call'):
    df = df.copy()
    if len(df) < 4:
        print(f"DEBUG: Insufficient data points ({len(df)}) for {ticker} ({option_type}) in fit_single_ticker")
        return None, None
    df.loc[:, 'SMI'] = 100 * (df['Ask'] - df['Bid']).clip(lower=0) / (df['Bid'] + df['Ask']).clip(lower=1e-6) / 2
    df.loc[:, 'weight'] = np.log(1 + df['Open Interest'].clip(lower=0)) / df['SMI'].clip(lower=1e-6)
    sigma_m = 0.6
    sigma_t = 0.6 * (df['Years_to_Expiry'].max() - df['Years_to_Expiry'].min()) if df['Years_to_Expiry'].max() > df['Years_to_Expiry'].min() else 1.0
    moneyness_dist = (df['Moneyness'] - 1.0)**2 / (2 * sigma_m**2)
    expiry_dist = (df['Years_to_Expiry'] - df['Years_to_Expiry'].median())**2 / (2 * sigma_t**2)
    df.loc[:, 'dist_weight'] = np.exp(-(moneyness_dist + expiry_dist))
    df.loc[:, 'weight'] *= df['dist_weight']
    df = df[
        (df['IV_mid'] > 0) & (df['IV_mid'] <= 5.0) &
        (df['weight'] > 0) & (df['Bid'] >= 0) & (df['Ask'] >= df['Bid']) &
        (df['SMI'] > 0)
    ]
    if len(df) < 4:
        print(f"DEBUG: Insufficient valid data points ({len(df)}) after filtering for {ticker} ({option_type})")
        return None, None
    df = df.sort_values(['Moneyness', 'Years_to_Expiry']).drop_duplicates(subset=['Moneyness', 'Years_to_Expiry'], keep='first')
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
        result = least_squares(residuals, p0, bounds=bounds, method='trf', max_nfev=max_nfev)
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
            result = least_squares(residuals, p0, bounds=bounds, method='trf', max_nfev=max_nfev)
            popt = result.x
            new_residuals = np.sum(w * (IV - model_func(xdata, *popt))**2)
            if abs(new_residuals - prev_residuals) < 0.001:
                break
            prev_residuals = new_residuals
        return popt, current_residuals
    except Exception as e:
        print(f"DEBUG: Fitting failed for {ticker} ({option_type}): {e}")
        return None, None
def compute_p90_restricted(df, params, ticker, option_type):
    valid_mask = (df['Moneyness'].notna()) & (df['Years_to_Expiry'].between(0.5, 2.0)) & \
                 (df['Moneyness'].between(0.5, 2.0)) & (df['IV_mid'].notna()) & \
                 (df['IV_mid'] > 0) & (df['Type'] == option_type)
    valid_temp = df[valid_mask]
    if len(valid_temp) < 4 or params is None:
        print(f"DEBUG: Insufficient data ({len(valid_temp)}) or no params for {ticker} ({option_type}) in compute_p90_restricted")
        return np.nan
    x_temp = np.vstack((valid_temp['Moneyness'].values, valid_temp['Years_to_Expiry'].values))
    smoothed_temp = global_vol_model_hyp(x_temp, *params)
    smoothed_temp = np.clip(smoothed_temp, 0.01, 5.0)
    try:
        atm_point = np.array([[1.0], [1.0]])
        atm_iv_temp = global_vol_model_hyp(atm_point, *params)
        atm_iv_temp = float(atm_iv_temp.item()) if isinstance(atm_iv_temp, np.ndarray) else float(atm_iv_temp)
        if np.isnan(atm_iv_temp) or atm_iv_temp <= 0:
            raise ValueError
    except Exception:
        atm_iv_temp = valid_temp['IV_mid'].median()
    if np.isnan(atm_iv_temp) or atm_iv_temp <= 0:
        print(f"DEBUG: Invalid ATM IV for {ticker} ({option_type}) in compute_p90_restricted")
        return np.nan
    rel_errors = np.abs((valid_temp['IV_mid'].values - smoothed_temp) / atm_iv_temp) * 100
    valid_rel = rel_errors[~np.isnan(rel_errors)]
    if len(valid_rel) == 0:
        print(f"DEBUG: No valid relative errors for {ticker} ({option_type}) in compute_p90_restricted")
        return np.nan
    return np.percentile(valid_rel, 90)
def compute_p90(exp_max_s, exp_min_l, df, df_type, ticker, option_type, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau, model):
    if exp_max_s + 0.1 >= exp_min_l:
        print(f"DEBUG: Invalid expiry bounds for {ticker} ({option_type})")
        return np.inf
    valid_temp_mask = (df_type['Moneyness'].notna()) & (df_type['Years_to_Expiry'] > 0.001) & \
                      (df_type['IV_mid'].notna()) & (df_type['IV_mid'] > 0)
    valid_temp = df_type[valid_temp_mask]
    if len(valid_temp) < 4:
        print(f"DEBUG: Insufficient data ({len(valid_temp)}) for {ticker} ({option_type}) in compute_p90")
        return np.inf
    params_short, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_short, exp_max=exp_max_s,
                                      mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                      model=model, option_type=option_type, p0=None, max_nfev=10000, max_iterations=3)
    if params_short is None:
        print(f"DEBUG: Short-term fit failed for {ticker} ({option_type})")
        return np.inf
    p0_long = params_short
    params_long, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_l, exp_max=exp_max_long,
                                     mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                     model=model, option_type=option_type, p0=p0_long, max_nfev=10000, max_iterations=3)
    if params_long is None:
        print(f"DEBUG: Long-term fit failed for {ticker} ({option_type})")
        return np.inf
    p0_full = params_long
    params_full, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_full, exp_max=exp_max_full,
                                     mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                     model=model, option_type=option_type, p0=p0_full, max_nfev=10000, max_iterations=3)
    if params_full is None:
        print(f"DEBUG: Full-term fit failed for {ticker} ({option_type})")
        return np.inf
    x_temp = np.vstack((valid_temp['Moneyness'].values, valid_temp['Years_to_Expiry'].values))
    smoothed_temp = global_vol_model_hyp(x_temp, *params_full)
    smoothed_temp = np.clip(smoothed_temp, 0.01, 5.0)
    try:
        atm_point = np.array([[1.0], [1.0]])
        atm_iv_temp = global_vol_model_hyp(atm_point, *params_full)
        atm_iv_temp = float(atm_iv_temp.item()) if isinstance(atm_iv_temp, np.ndarray) else float(atm_iv_temp)
        if np.isnan(atm_iv_temp) or atm_iv_temp <= 0:
            raise ValueError
    except Exception:
        atm_iv_temp = valid_temp['IV_mid'].median()
    if np.isnan(atm_iv_temp) or atm_iv_temp <= 0:
        print(f"DEBUG: Invalid ATM IV for {ticker} ({option_type}) in compute_p90")
        return np.inf
    rel_errors = np.abs((valid_temp['IV_mid'].values - smoothed_temp) / atm_iv_temp) * 100
    valid_rel = rel_errors[~np.isnan(rel_errors)]
    if len(valid_rel) == 0:
        print(f"DEBUG: No valid relative errors for {ticker} ({option_type}) in compute_p90")
        return np.inf
    return np.percentile(valid_rel, 90)
def optimize_p90(df, df_type, ticker, option_type, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau, model):
    expiry_min = df_type['Years_to_Expiry'].min()
    expiry_max = df_type['Years_to_Expiry'].max()
    median_expiry = df_type['Years_to_Expiry'].median()
    def objective(trial):
        exp_max_s = trial.suggest_float('exp_max_s', expiry_min + 0.05, median_expiry - 0.05)
        exp_min_l = trial.suggest_float('exp_min_l', median_expiry + 0.1, expiry_max - 0.05)
        if exp_min_l - exp_max_s < 0.1:
            return np.inf
        p90 = compute_p90(exp_max_s, exp_min_l, df, df_type, ticker, option_type, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau, model)
        penalty = 0.01 * (exp_max_s + exp_min_l)
        return p90 + penalty if np.isfinite(p90) else np.inf
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=8, timeout=60)
    if study.best_value < np.inf:
        best_exp_max_short = study.best_params['exp_max_s']
        best_exp_min_long = study.best_params['exp_min_l']
        best_p90 = study.best_value - 0.01 * (best_exp_max_short + best_exp_min_long)
        print(f"{ticker} ({option_type}): p90={best_p90:.2f}%, short_max={best_exp_max_short:.3f}, long_min={best_exp_min_long:.3f}")
        return best_exp_max_short, best_exp_min_long
    best_exp_max_short = max(df_type['Years_to_Expiry'].min() + 0.3, 0.3)
    best_exp_min_long = min(df_type['Years_to_Expiry'].max() - 0.3, 1.8)
    print(f"{ticker} ({option_type}): p90=nan%, short_max={best_exp_max_short:.3f}, long_min={best_exp_min_long:.3f}")
    return best_exp_max_short, best_exp_min_long
# Volatility surface fitting
def global_vol_model_hyp(x, a0, a1, b0, b1, m0, m1, rho0, rho1, sigma0, sigma1, c):
    moneyness, tau = x
    k = np.log(moneyness)
    try:
        a_T = a0 + a1 / (1 + c * tau)
        b_T = b0 + b1 / (1 + c * tau)
        m_T = m0 + m1 / (1 + c * tau)
        rho_T = rho0 + rho1 / (1 + c * tau)
        sigma_T = sigma0 + sigma1 / (1 + c * tau)
        return a_T + b_T * (rho_T * (k - m_T) + np.sqrt((k - m_T)**2 + sigma_T**2))
    except Exception:
        return np.nan
MODEL_CONFIG = {
    'hyp': {
        'func': global_vol_model_hyp,
        'p0': [0.2, 0.1, 0.1, 0.05, 0.0, 0.0, -0.3, -0.1, 0.1, 0.05, 0.1],
        'bounds': ([0, -np.inf, 0, -np.inf, -np.inf, -np.inf, -1, -1, 0.01, -np.inf, 0],
                   [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf])
    }
}
def fit_vol_surface(df, ticker=None, exp_min=0.0, exp_max=np.inf, mon_min=0.0, mon_max=np.inf, extrap_tau=None, model='hyp', option_type='Call', p0=None, max_nfev=10000, max_iterations=3):
    df = df.copy()
    if model not in MODEL_CONFIG:
        return None, None
    df = df[(df['Type'] == option_type) & (df['Years_to_Expiry'] > exp_min) & (df['Years_to_Expiry'] < exp_max) &
            (df['Moneyness'] > mon_min) & (df['Moneyness'] < mon_max) & (df['IV_mid'].notna())]
    if ticker:
        df_ticker = df[df['Ticker'] == ticker]
        if df_ticker.empty:
            print(f"DEBUG: No data for {ticker} ({option_type}) in fit_vol_surface")
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
def plot_vol_surface(df, params, ticker, option_type, timestamp):
    plots_dir = f'data/{timestamp}/plots'
    os.makedirs(plots_dir, exist_ok=True)
    valid_mask = (df['Moneyness'].notna()) & (df['Years_to_Expiry'] > 0.001) & \
                 (df['IV_mid'].notna()) & (df['IV_mid'] > 0) & (df['Type'] == option_type)
    df_plot = df[valid_mask].copy()
    if df_plot.empty or params is None:
        print(f"DEBUG: No valid data or parameters for {ticker} ({option_type}) plot")
        return
    m_min, m_max = df_plot['Moneyness'].min(), df_plot['Moneyness'].max()
    t_min, t_max = df_plot['Years_to_Expiry'].min(), df_plot['Years_to_Expiry'].max()
    m_grid = np.linspace(max(0.1, m_min), min(5.0, m_max), 50)
    t_grid = np.linspace(max(0.001, t_min), min(t_max, 5.0), 50)
    M, T = np.meshgrid(m_grid, t_grid)
    X = np.vstack((M.ravel(), T.ravel()))
    Z = global_vol_model_hyp(X, *params)
    Z = np.clip(Z, 0.01, 5.0).reshape(M.shape)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=M, y=T, z=Z,
        colorscale='Viridis',
        opacity=0.7,
        name='Fitted Surface',
        showscale=True
    ))
    fig.add_trace(go.Scatter3d(
        x=df_plot['Moneyness'],
        y=df_plot['Years_to_Expiry'],
        z=df_plot['IV_mid'],
        mode='markers',
        marker=dict(size=5, color='red', symbol='circle'),
        name='Data Points'
    ))
    fig.update_layout(
        title=f'Volatility Surface - {ticker} ({option_type})',
        scene=dict(
            xaxis_title='Moneyness (K/F)',
            yaxis_title='Years to Expiry',
            zaxis_title='Implied Volatility'
        ),
        showlegend=True,
        width=800,
        height=600
    )
    plot_file = os.path.join(plots_dir, f'vol_surface_{ticker}_{option_type.lower()}.html')
    fig.write_html(plot_file)
    print(f"Saved volatility surface plot for {ticker} ({option_type}) to {plot_file}")
def calculate_skew_slope_metrics(df, ticker, timestamp, r, q=0.0):
    vol_surf_file = f'data/{timestamp}/vol_surf/vol_surf.csv'
    if not os.path.exists(vol_surf_file):
        print(f"DEBUG: No volatility surface file for {ticker}")
        return pd.DataFrame(), pd.DataFrame()
    try:
        vol_surf_df = pd.read_csv(vol_surf_file)
    except Exception as e:
        print(f"DEBUG: Error reading volatility surface file for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame()
    vol_surf_df = vol_surf_df[vol_surf_df['Ticker'] == ticker]
    if vol_surf_df.empty:
        print(f"DEBUG: No volatility surface parameters for {ticker}")
        return pd.DataFrame(), pd.DataFrame()
    call_params = vol_surf_df[vol_surf_df['Option_Type'] == 'Call']
    put_params = vol_surf_df[vol_surf_df['Option_Type'] == 'Put']
    if call_params.empty or put_params.empty:
        print(f"DEBUG: Missing call or put parameters for {ticker}")
        return pd.DataFrame(), pd.DataFrame()
    call_params = call_params.iloc[0][['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']].values
    put_params = put_params.iloc[0][['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']].values
    S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2 if 'Bid Stock' in df and 'Ask Stock' in df else df['Last Stock Price'].iloc[0]
    if S <= 0:
        print(f"DEBUG: Invalid stock price for {ticker}")
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
        print(f"Saved skew metrics for {ticker} to {skew_file}")
    if not slope_metrics_df.empty:
        slope_file = os.path.join(slope_dir, f'slope_metrics_yfinance_{ticker}.csv')
        slope_metrics_df.to_csv(slope_file, index=False)
        print(f"Saved slope metrics for {ticker} to {slope_file}")
    return skew_metrics_df, slope_metrics_df
def process_ticker(ticker, timestamp, yields_dict, model, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau):
    data_file = f'data/{timestamp}/cleaned_yfinance/cleaned_yfinance_{ticker}.csv'
    if not os.path.exists(data_file):
        print(f"DEBUG: Data file not found for {ticker}: {data_file}")
        print(f"{ticker} (Call): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Call): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Call): One_Yr_ATM_Rel_Error = nan%")
        print(f"{ticker} (Put): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Put): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Put): One_Yr_ATM_Rel_Error = nan%")
        return None, None
    try:
        df = pd.read_csv(data_file, parse_dates=['Expiry'])
    except Exception as e:
        print(f"DEBUG: Error reading data file for {ticker}: {e}")
        print(f"{ticker} (Call): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Call): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Call): One_Yr_ATM_Rel_Error = nan%")
        print(f"{ticker} (Put): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Put): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Put): One_Yr_ATM_Rel_Error = nan%")
        return None, None
    if df.empty:
        print(f"DEBUG: Empty dataframe for {ticker}")
        print(f"{ticker} (Call): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Call): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Call): One_Yr_ATM_Rel_Error = nan%")
        print(f"{ticker} (Put): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Put): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Put): One_Yr_ATM_Rel_Error = nan%")
        return None, None
    df['Ticker'] = ticker
    q = get_dividend_yield(ticker)
    timestamp_dt = datetime.strptime(timestamp, '%Y%m%d_%H%M')
    df['Expiry_dt'] = pd.to_datetime(df['Expiry'])
    df['Years_to_Expiry'] = (df['Expiry_dt'] - timestamp_dt).dt.days / 365.25
    S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2 if 'Bid Stock' in df and 'Ask Stock' in df else df['Last Stock Price'].iloc[0]
    if S <= 0:
        print(f"DEBUG: Invalid stock price for {ticker}: {S}")
        print(f"{ticker} (Call): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Call): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Call): One_Yr_ATM_Rel_Error = nan%")
        print(f"{ticker} (Put): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Put): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Put): One_Yr_ATM_Rel_Error = nan%")
        return None, None
    df['Last Stock Price'] = S
    df = calculate_iv_binomial(df, yields_dict, q=q, default_r=0.05, max_workers=4)
    if df.empty or not (df['IV_mid'].notna() & (df['IV_mid'] > 0)).any():
        print(f"DEBUG: No valid IV_mid data for {ticker}")
        print(f"{ticker} (Call): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Call): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Call): One_Yr_ATM_Rel_Error = nan%")
        print(f"{ticker} (Put): p90=nan%, short_max=nan, long_min=nan")
        print(f"{ticker} (Put): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
        print(f"{ticker} (Put): One_Yr_ATM_Rel_Error = nan%")
        return None, None
    df['Forward'] = df['Last Stock Price'] * np.exp((df['r'] - q) * df['Years_to_Expiry'])
    df['Moneyness'] = df['Strike'] / df['Forward']
    df['Smoothed_IV'] = np.nan
    params_calls = None
    params_puts = None
    param_dfs = []
    param_names = ['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']
    for option_type in ['Call', 'Put']:
        df_type = df[df['Type'] == option_type].copy()
        valid_data = df_type[(df_type['Moneyness'].notna()) & (df_type['Years_to_Expiry'] > 0.001) & (df_type['IV_mid'].notna()) & (df_type['IV_mid'] > 0)]
        if df_type.empty or len(valid_data) < 4:
            print(f"DEBUG: Insufficient valid data ({len(valid_data)}) for {ticker} ({option_type})")
            print(f"{ticker} ({option_type}): p90=nan%, short_max=nan, long_min=nan")
            print(f"{ticker} ({option_type}): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
            print(f"{ticker} ({option_type}): One_Yr_ATM_Rel_Error = nan%")
            param_df = pd.DataFrame({
                'Ticker': [ticker], 'Model': [model], 'Residuals': [np.nan], 'Timestamp': [timestamp],
                'Option_Type': [option_type],
                'P90_Rel_Error_%': [np.nan], 'Restricted_P90_Rel_Error_%': [np.nan],
                'One_Yr_ATM_Rel_Error_%': [np.nan]
            })
            for param in param_names:
                param_df[param] = np.nan
            param_dfs.append(param_df)
            continue
        if len(df_type) < 10:
            best_exp_max_short = max(df_type['Years_to_Expiry'].min() + 0.3, 0.3)
            best_exp_min_long = min(df_type['Years_to_Expiry'].max() - 0.3, 1.8)
            print(f"DEBUG: Insufficient data points ({len(df_type)}) for optimization for {ticker} ({option_type})")
            print(f"{ticker} ({option_type}): p90=nan%, short_max={best_exp_max_short:.3f}, long_min={best_exp_min_long:.3f}")
            print(f"{ticker} ({option_type}): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
            print(f"{ticker} ({option_type}): One_Yr_ATM_Rel_Error = nan%")
            param_df = pd.DataFrame({
                'Ticker': [ticker], 'Model': [model], 'Residuals': [np.nan], 'Timestamp': [timestamp],
                'Option_Type': [option_type],
                'P90_Rel_Error_%': [np.nan], 'Restricted_P90_Rel_Error_%': [np.nan],
                'One_Yr_ATM_Rel_Error_%': [np.nan]
            })
            for param in param_names:
                param_df[param] = np.nan
            param_dfs.append(param_df)
        else:
            best_exp_max_short, best_exp_min_long = optimize_p90(df, df_type, ticker, option_type, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau, model)
            params_short, _ = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_short, exp_max=best_exp_max_short,
                                              mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                              model=model, option_type=option_type, p0=None, max_nfev=10000, max_iterations=3)
            p0_long = params_short if params_short is not None else None
            params_long, _ = fit_vol_surface(df, ticker=ticker, exp_min=best_exp_min_long, exp_max=exp_max_long,
                                             mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                             model=model, option_type=option_type, p0=p0_long, max_nfev=10000, max_iterations=3)
            p0_full = params_long if params_long is not None else None
            params, residuals = fit_vol_surface(df, ticker=ticker, exp_min=exp_min_full, exp_max=exp_max_full,
                                                mon_min=mon_min, mon_max=mon_max, extrap_tau=extrap_tau,
                                                model=model, option_type=option_type, p0=p0_full, max_nfev=10000, max_iterations=3)
            print(f"DEBUG: params_{option_type.lower()} for {ticker}: {params}")
            if option_type == 'Call':
                params_calls = params
            else:
                params_puts = params
            if params is not None:
                p90_full = compute_p90(best_exp_max_short, best_exp_min_long, df, df_type, ticker, option_type, exp_min_short, exp_max_long, exp_min_full, exp_max_full, mon_min, mon_max, extrap_tau, model)
                if not np.isfinite(p90_full):
                    p90_full = np.nan
                print(f"{ticker} ({option_type}): p90={p90_full:.2f}%, short_max={best_exp_max_short:.3f}, long_min={best_exp_min_long:.3f}")
                p90_restricted = compute_p90_restricted(df, params, ticker, option_type)
                print(f"{ticker} ({option_type}): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = {p90_restricted:.2f}%")
                valid_df = df[(df['Type'] == option_type) & df['IV_mid'].notna() & (df['IV_mid'] > 0)]
                if not valid_df.empty:
                    dist = (valid_df['Moneyness'] - 1)**2 + (valid_df['Years_to_Expiry'] - 1)**2
                    closest_idx = dist.idxmin()
                    iv_mid = valid_df.loc[closest_idx, 'IV_mid']
                    moneyness = valid_df.loc[closest_idx, 'Moneyness']
                    expiry = valid_df.loc[closest_idx, 'Years_to_Expiry']
                    try:
                        smoothed_iv = global_vol_model_hyp(np.array([[moneyness], [expiry]]), *params)
                        smoothed_iv = float(np.clip(smoothed_iv, 0.01, 5.0)) if isinstance(smoothed_iv, (np.ndarray, np.floating)) else np.nan
                        if not np.isnan(iv_mid) and not np.isnan(smoothed_iv) and iv_mid > 0:
                            atm_rel_error = abs(iv_mid - smoothed_iv) / iv_mid * 100
                        else:
                            atm_rel_error = np.nan
                            print(f"DEBUG: Invalid IV_mid ({iv_mid}) or Smoothed_IV ({smoothed_iv}) for {ticker} ({option_type}) at M={moneyness:.3f}, T={expiry:.3f}")
                    except Exception as e:
                        atm_rel_error = np.nan
                        print(f"DEBUG: Error computing Smoothed_IV for {ticker} ({option_type}) at M={moneyness:.3f}, T={expiry:.3f}: {e}")
                else:
                    atm_rel_error = np.nan
                    print(f"DEBUG: No valid data for ATM error calculation for {ticker} ({option_type})")
                print(f"{ticker} ({option_type}): One_Yr_ATM_Rel_Error = {atm_rel_error:.2f}%")
                param_df = pd.DataFrame([params], columns=param_names)
                param_df['Ticker'] = ticker
                param_df['Model'] = model
                param_df['Residuals'] = residuals if residuals is not None else np.nan
                param_df['Timestamp'] = timestamp
                param_df['Option_Type'] = option_type
                param_df['P90_Rel_Error_%'] = p90_full
                param_df['Restricted_P90_Rel_Error_%'] = p90_restricted
                param_df['One_Yr_ATM_Rel_Error_%'] = atm_rel_error
                param_dfs.append(param_df)
            else:
                print(f"DEBUG: No parameters fitted for {ticker} ({option_type})")
                print(f"{ticker} ({option_type}): p90=nan%, short_max={best_exp_max_short:.3f}, long_min={best_exp_min_long:.3f}")
                print(f"{ticker} ({option_type}): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
                print(f"{ticker} ({option_type}): One_Yr_ATM_Rel_Error = nan%")
                param_df = pd.DataFrame({
                    'Ticker': [ticker], 'Model': [model], 'Residuals': [np.nan], 'Timestamp': [timestamp],
                    'Option_Type': [option_type],
                    'P90_Rel_Error_%': [np.nan], 'Restricted_P90_Rel_Error_%': [np.nan],
                    'One_Yr_ATM_Rel_Error_%': [np.nan]
                })
                for param in param_names:
                    param_df[param] = np.nan
                param_dfs.append(param_df)
            if DEBUG and params is not None:
                plot_vol_surface(df_type, params, ticker, option_type, timestamp)
    try:
        df = calculate_smoothed_iv(df, params_calls, params_puts, model)
    except Exception as e:
        print(f"DEBUG: Error in calculate_smoothed_iv for {ticker}: {e}")
        df['Smoothed_IV'] = np.nan
    processed_dir = f'data/{timestamp}/processed_yfinance'
    vol_surf_dir = f'data/{timestamp}/vol_surf'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(vol_surf_dir, exist_ok=True)
    output_file = f'{processed_dir}/processed_yfinance_{ticker}.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved processed data for {ticker} to {output_file}")
    param_df = pd.concat(param_dfs, ignore_index=True) if param_dfs else pd.DataFrame()
    if not param_df.empty:
        vol_surf_file = f'{vol_surf_dir}/vol_surf.csv'
        param_df.to_csv(vol_surf_file, index=False)
        print(f"Saved vol surface parameters for {ticker} to {vol_surf_file}")
    # Add call to calculate skew and slope metrics
    if params_calls is not None and params_puts is not None:
        r = interpolate_r(1.0, yields_dict)  # Using T=1.0 as a representative for r
        skew_df, slope_df = calculate_skew_slope_metrics(df, ticker, timestamp, r, q)
        # Optionally, you can collect them or just rely on the internal saves
    else:
        print(f"DEBUG: Skipping skew/slope calculation for {ticker} due to missing params")
        skew_df = pd.DataFrame()
        slope_df = pd.DataFrame()
    # For now, return skew_df as metrics_df (or combine skew and slope if needed)
    metrics_df = pd.concat([skew_df, slope_df], ignore_index=True)
    return metrics_df, param_df
def process_volumes(timestamp):
    print(f"Starting process_volumes for timestamp {timestamp}")
    if not os.path.exists('tickers.txt'):
        print("tickers.txt not found, skipping processing")
        return
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    #tickers = ['COIN']
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
        for future in as_completed(futures): # Removed timeout=300
            ticker = futures[future]
            try:
                metrics_df, param_df = future.result()
                if not metrics_df.empty:
                    all_metrics.append(metrics_df)
                if not param_df.empty:
                    all_params.append(param_df)
            except Exception as e:
                print(f"DEBUG: Error processing {ticker}: {e}")
                print(f"{ticker} (Call): p90=nan%, short_max=nan, long_min=nan")
                print(f"{ticker} (Call): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
                print(f"{ticker} (Call): One_Yr_ATM_Rel_Error = nan%")
                print(f"{ticker} (Put): p90=nan%, short_max=nan, long_min=nan")
                print(f"{ticker} (Put): Restricted P90 (moneyness 0.5-2.0, expiry 0.5-2.0) = nan%")
                print(f"{ticker} (Put): One_Yr_ATM_Rel_Error = nan%")
    if all_metrics:
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        metrics_file = f'data/{timestamp}/processed_yfinance/fit_metrics_yfinance_all.csv'
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        all_metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved all metrics to {metrics_file}")
    if all_params:
        all_params_df = pd.concat(all_params, ignore_index=True)
        vol_surf_file = f'data/{timestamp}/vol_surf/vol_surf.csv'
        os.makedirs(os.path.dirname(vol_surf_file), exist_ok=True)
        all_params_df.to_csv(vol_surf_file, index=False)
        print(f"Saved all vol surface params to {vol_surf_file}")
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
    print(f"Updated dates.json with {timestamp}")
if __name__ == '__main__':
    import sys
    timestamp = sys.argv[1] if len(sys.argv) > 1 else '20250902_2137'
    process_volumes(timestamp)
