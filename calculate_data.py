```python
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import json
import yfinance as yf
from scipy.optimize import brentq, least_squares
from scipy.stats import norm
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RBFInterpolator
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import multiprocessing
import warnings
import statsmodels.api as sm
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

def black_scholes_call(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def black_scholes_delta(S, K, T, r, q, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def implied_vol(price, S, K, T, r, q, option_type, contract_name=""):
    if price <= 0 or T <= 0:
        with open('data_error.log', 'a') as f:
            f.write(f"Invalid price ({price}) or T ({T}) for {contract_name}\n")
        return np.nan
    intrinsic = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
    if price < intrinsic * np.exp(-r * T) * 0.99:
        with open('data_error.log', 'a') as f:
            f.write(f"Price ({price}) below intrinsic ({intrinsic * np.exp(-r * T)}) for {contract_name}\n")
        return np.nan
    def objective(sigma):
        if option_type.lower() == 'call':
            return black_scholes_call(S, K, T, r, q, sigma) - price
        else:
            return black_scholes_put(S, K, T, r, q, sigma) - price
    try:
        iv = brentq(objective, 0.0001, 50.0)
        return np.clip(iv, 0.05, 5.0)
    except ValueError as e:
        with open('data_error.log', 'a') as f:
            f.write(f"IV solver failed for {contract_name}: {str(e)}\n")
        return np.nan

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

def fit_single_ticker(df, model='hyp'):
    df = df.copy()
    df.loc[:, 'SMI'] = 100 * (df['Ask'] - df['Bid']).clip(lower=0) / (df['Bid'] + df['Ask']).clip(lower=1e-6) / 2
    df.loc[:, 'weight'] = np.log(1 + df['Open Interest']) / df['SMI'].clip(lower=1e-6)
    df.loc[:, 'weight'] = np.where(df['Years_to_Expiry'].between(2.0, 2.6), df['weight'] * 5.0, df['weight'])
    df = df[
        (df['IV_mid'] > 0) &
        (df['weight'] > 0) &
        (df['Bid'] >= 0) &
        (df['Ask'] >= df['Bid']) &
        (df['SMI'] > 0)
    ]
    ticker = df['Ticker'].iloc[0] if not df.empty else "unknown"
    with open('data_error.log', 'a') as f:
        f.write(f"Data points after filtering for {ticker}: {len(df)}, "
                f"Moneyness range: {df['Moneyness'].min():.2f}–{df['Moneyness'].max():.2f}, "
                f"Expiry range: {df['Years_to_Expiry'].min():.2f}–{df['Years_to_Expiry'].max():.2f}, "
                f"IV range: {df['IV_mid'].min():.2f}–{df['IV_mid'].max():.2f}\n" if not df.empty else
                f"No valid data points after filtering for {ticker}\n")
    if len(df) < 12:
        with open('data_error.log', 'a') as f:
            f.write(f"Insufficient data points ({len(df)}) for {ticker} in fit_single_ticker\n")
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
    bounds = model_config['bounds']
    try:
        def residuals(params):
            return np.sqrt(w_all) * (model_func(xdata, *params) - IV_all)
        result = least_squares(
            residuals,
            p0,
            bounds=bounds,
            method='trf',
            max_nfev=50000
        )
        popt = result.x
        IV_pred = model_func(xdata, *popt)
        residuals = np.sum(w * (IV - IV_pred)**2)
        return popt, residuals
    except Exception as e:
        with open('data_error.log', 'a') as f:
            f.write(f"Fit failed for {ticker}: {str(e)}\n")
        return None, None

def load_rvol_from_historic(ticker, timestamp, days=100):
    historic_file = f'data/{timestamp}/historic/historic_{ticker}.csv'
    if not os.path.exists(historic_file):
        return None
    df_hist = pd.read_csv(historic_file, parse_dates=['Date'])
    if df_hist.empty:
        return None
    col = f'Realised_Vol_Close_{days}'
    if col not in df_hist.columns:
        return None
    latest_vol = df_hist[col].iloc[-1] / 100
    return latest_vol if not pd.isna(latest_vol) else None

def calc_Ivol_Rvol(df, rvol100d):
    if df.empty:
        return df
    df = df.copy()
    df.loc[:, "Ivol/Rvol100d Ratio"] = df["IV_mid"] / rvol100d if rvol100d else np.nan
    return df

def compute_ivs(row, S, r, q):
    if pd.isna(row['Years_to_Expiry']):
        with open('data_error.log', 'a') as f:
            f.write(f"Invalid Years_to_Expiry for {row['Contract Name']}\n")
        return np.nan, np.nan, np.nan, np.nan, np.nan
    T = max(row['Years_to_Expiry'], 0.0001)
    option_type = row['Type'].lower()
    contract_name = row['Contract Name']
    iv_bid = implied_vol(row['Bid'], S, row['Strike'], T, r, q, option_type, contract_name)
    iv_ask = implied_vol(row['Ask'], S, row['Strike'], T, r, q, option_type, contract_name)
    iv_mid = implied_vol(0.5*(row['Bid']+row['Ask']), S, row['Strike'], T, r, q, option_type, contract_name)
    iv_spread = iv_ask - iv_bid if not np.isnan(iv_bid) else np.nan
    delta = black_scholes_delta(S, row['Strike'], T, r, q, iv_mid, option_type)
    return iv_bid, iv_ask, iv_mid, iv_spread, delta

def calculate_metrics(df, ticker, r):
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    skew_data = []
    for exp in df["Expiry"].unique():
        for strike in df["Strike"].unique():
            call_iv = df[(df["Type"] == "Call") & (df["Strike"] == strike) & (df["Expiry"] == exp)]["IV_mid"]
            put_iv = df[(df["Type"] == "Put") & (df["Strike"] == strike) & (df["Expiry"] == exp)]["IV_mid"]
            if not call_iv.empty and not put_iv.empty and call_iv.iloc[0] > 0:
                skew = put_iv.iloc[0] / call_iv.iloc[0]
                skew_data.append({"Expiry": exp, "Strike": strike, "Vol Skew": f"{skew*100:.2f}%"})
    skew_df = pd.DataFrame(skew_data)
    slope_data = []
    for strike in df["Strike"].unique():
        for opt_type in ["Call", "Put"]:
            subset = df[(df["Strike"] == strike) & (df["Type"] == opt_type)].sort_values("Expiry")
            if len(subset) > 1:
                iv_diff = subset["IV_mid"].diff()
                subset = subset.copy()
                subset.loc[:, "Expiry_dt"] = pd.to_datetime(subset["Expiry"])
                time_diff = (subset["Expiry_dt"] - subset["Expiry_dt"].shift(1)).map(lambda x: x.days / 365.0)
                slope = iv_diff / time_diff
                for i in range(1, len(subset)):
                    slope_data.append({
                        "Strike": strike,
                        "Type": opt_type,
                        "Expiry": subset["Expiry"].iloc[i],
                        "IV Slope": slope.iloc[i]
                    })
    slope_df = pd.DataFrame(slope_data)
    return df, skew_df, slope_df

def calculate_iv_mid(df, ticker, r, timestamp):
    if df.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"Empty input DataFrame for {ticker} in calculate_iv_mid\n")
        return df, None, None, None
    required_columns = ['Ticker', 'Type', 'Expiry', 'Strike', 'Bid', 'Ask', 'Bid Stock', 'Ask Stock', 'Contract Name']
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        with open('data_error.log', 'a') as f:
            f.write(f"Missing columns in {ticker} data: {missing_cols}\n")
        return df, None, None, None
    df = df.copy()
    invalid_prices = df[(df['Bid'] <= 0) | (df['Ask'] <= 0) | (df['Bid'].isna()) | (df['Ask'].isna())]
    if not invalid_prices.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"Invalid prices for {ticker}: {len(invalid_prices)} rows with Bid <= 0 or Ask <= 0\n")
    S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2
    if pd.isna(S) or S <= 0:
        with open('data_error.log', 'a') as f:
            f.write(f"Invalid stock price for {ticker}: S={S}\n")
        return df, None, None, None
    try:
        stock = yf.Ticker(ticker)
        q = float(stock.info.get('trailingAnnualDividendYield', 0.0))
    except Exception as e:
        with open('data_error.log', 'a') as f:
            f.write(f"Failed to fetch dividend yield for {ticker}: {str(e)}, using q=0.0\n")
        q = 0.0
    today = datetime.today()
    df.loc[:, "Expiry_dt"] = pd.to_datetime(df["Expiry"])
    df.loc[:, 'Years_to_Expiry'] = (df['Expiry_dt'] - today).dt.days / 365.25
    df = df[df['Years_to_Expiry'] > 0]
    if df.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"No valid expiries for {ticker} after filtering Years_to_Expiry > 0\n")
        return df, None, None, None
    df.loc[:, 'Forward'] = S * np.exp((r - q) * df['Years_to_Expiry'])
    df.loc[:, 'Moneyness'] = df['Strike'] / df['Forward']
    df.loc[:, 'LogMoneyness'] = np.log(df['Strike'] / df['Forward'])
    results = Parallel(n_jobs=4, backend='threading')(delayed(compute_ivs)(row, S, r, q) for _, row in df.iterrows())
    df.loc[:, ['IV_bid', 'IV_ask', 'IV_mid', 'IV_spread', 'Delta']] = pd.DataFrame(results, index=df.index)
    with open('data_error.log', 'a') as f:
        f.write(f"IV_mid stats for {ticker}: valid={len(df[df['IV_mid'] > 0])}, "
                f"nan={df['IV_mid'].isna().sum()}, min={df['IV_mid'].min():.2f}, max={df['IV_mid'].max():.2f}\n")
    return df, S, r, q

def smooth_iv_per_expiry(options_df):
    if options_df.empty:
        options_df = options_df.copy()
        options_df.loc[:, 'Smoothed_IV'] = np.nan
        return options_df
    options_df = options_df.copy()
    smoothed_iv = pd.Series(np.nan, index=options_df.index, dtype=float)
    for exp, group in options_df.groupby('Expiry'):
        if len(group) < 3 or group['IV_mid'].isna().all():
            smoothed_iv.loc[group.index] = group['IV_mid']
            continue
        if len(group) >= 5:
            mean_iv = np.mean(group['IV_mid'])
            std_iv = np.std(group['IV_mid'])
            if std_iv > 0:
                z_scores = np.abs((group['IV_mid'] - mean_iv) / std_iv)
                is_outlier = z_scores > 3
                cleaned_group = group[~is_outlier]
            else:
                cleaned_group = group
        else:
            cleaned_group = group
        if len(cleaned_group) < 3 or cleaned_group['IV_mid'].isna().all():
            smoothed_iv.loc[group.index] = group['IV_mid']
            continue
        if cleaned_group['LogMoneyness'].duplicated().any():
            agg_group = cleaned_group.groupby('LogMoneyness')['IV_mid'].mean().reset_index()
            agg_group = agg_group.sort_values('LogMoneyness')
            x = agg_group['LogMoneyness'].values
            y = agg_group['IV_mid'].values
        else:
            sorted_group = cleaned_group.sort_values('LogMoneyness')
            x = sorted_group['LogMoneyness'].values
            y = sorted_group['IV_mid'].values
        try:
            lowess_smoothed = sm.nonparametric.lowess(y, x, frac=0.3, it=3)
            x_smooth = lowess_smoothed[:, 0]
            y_smooth = lowess_smoothed[:, 1]
            interpolator = interp1d(x_smooth, y_smooth, bounds_error=False, fill_value="extrapolate")
            smoothed_values = interpolator(group['LogMoneyness'].values)
            smoothed_iv.loc[group.index] = pd.Series(smoothed_values, index=group.index)
        except Exception as e:
            with open('data_error.log', 'a') as f:
                f.write(f"LOWESS failed for expiry {exp} in {options_df['Ticker'].iloc[0]}: {str(e)}, using IV_mid directly\n")
            smoothed_iv.loc[group.index] = group['IV_mid']
    options_df['Smoothed_IV'] = smoothed_iv
    return options_df

def compute_local_vol_from_iv_row(row, r, q, interp):
    y = row['LogMoneyness']
    T = row['Years_to_Expiry']
    if T <= 0 or pd.isna(row['Smoothed_IV']):
        with open('data_error.log', 'a') as f:
            f.write(f"Invalid Smoothed_IV or Years_to_Expiry for Strike {row['Strike']}, Expiry {row['Expiry']}\n")
        return None
    w = row['Smoothed_IV'] ** 2 * T
    h_t = max(0.01 * T, 1e-4)
    h_y = max(0.01 * abs(y) if y != 0 else 0.01, 1e-4)
    try:
        w_T_plus = interp(np.array([[y, T + h_t]]))[0]
        w_T_minus = interp(np.array([[y, max(T - h_t, 1e-6)]]))[0]
        dw_dT = (w_T_plus - w_T_minus) / (2 * h_t)
        w_y_plus = interp(np.array([[y + h_y, T]]))[0]
        w_y_minus = interp(np.array([[y - h_y, T]]))[0]
        dw_dy = (w_y_plus - w_y_minus) / (2 * h_y)
        d2w_dy2 = (w_y_plus - 2 * w + w_y_minus) / (h_y ** 2)
        if np.isnan(dw_dT) or np.isnan(dw_dy) or np.isnan(d2w_dy2):
            return None
    except Exception as e:
        with open('data_error.log', 'a') as f:
            f.write(f"Interpolation failed for Strike {row['Strike']}, Expiry {row['Expiry']}: {str(e)}\n")
        return None
    denom = 1 - (y / w) * dw_dy + 0.25 * (-0.25 - 1/w + (y**2 / w**2)) * (dw_dy ** 2) + 0.5 * d2w_dy2
    if denom <= 1e-10 or dw_dT <= 0:
        local_vol = 0.0
    else:
        local_vol_sq = dw_dT / denom
        local_vol = np.sqrt(max(local_vol_sq, 0)) if local_vol_sq > 0 else 0.0
        if local_vol > 2.0:
            local_vol = np.nan
    return {
        "Strike": row['Strike'],
        "Expiry": row['Expiry'],
        "Local Vol": local_vol
    }

def process_options(options_df, option_type, r, q):
    if options_df.empty:
        options_df = options_df.copy()
        options_df.loc[:, 'Smoothed_IV'] = np.nan
        return pd.DataFrame(), None, options_df
    options_df = options_df[options_df['IV_mid'] > 0]
    options_df = options_df[options_df['Years_to_Expiry'] > 0]
    options_df = smooth_iv_per_expiry(options_df)
    if 'Smoothed_IV' not in options_df.columns:
        options_df.loc[:, 'Smoothed_IV'] = options_df['IV_mid']
    smoothed_df = options_df.copy()
    smoothed_df = smoothed_df.sort_values(['Years_to_Expiry', 'LogMoneyness'])
    smoothed_df.loc[:, 'TotalVariance'] = smoothed_df['Smoothed_IV']**2 * smoothed_df['Years_to_Expiry']
    points = np.column_stack((smoothed_df['LogMoneyness'], smoothed_df['Years_to_Expiry']))
    values = smoothed_df['TotalVariance'].values
    if len(smoothed_df) < 3:
        return pd.DataFrame(), None, smoothed_df
    try:
        interp = RBFInterpolator(points, values, kernel='thin_plate_spline', smoothing=0.1)
    except Exception as e:
        with open('data_error.log', 'a') as f:
            f.write(f"RBF fit failed for {option_type} in {options_df['Ticker'].iloc[0]}: {str(e)}, using linear fallback\n")
        interp = LinearNDInterpolator(points, values, fill_value=np.nan, rescale=True)
    local_data = Parallel(n_jobs=4, backend='threading')(
        delayed(compute_local_vol_from_iv_row)(row, r, q, interp)
        for _, row in smoothed_df.iterrows()
    )
    local_data = [d for d in local_data if d is not None]
    local_df = pd.DataFrame(local_data) if local_data else pd.DataFrame()
    return local_df, interp, smoothed_df

def calculate_local_vol_from_iv(df, S, r, q, ticker):
    required_columns = ['Type', 'Strike', 'Expiry', 'IV_mid', 'Years_to_Expiry', 'Forward', 'LogMoneyness']
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        with open('data_error.log', 'a') as f:
            f.write(f"Missing columns for local vol in {ticker}: {missing_cols}\n")
        return pd.DataFrame(), pd.DataFrame(), None, None, df
    calls = df[df['Type'] == 'Call'].copy()
    puts = df[df['Type'] == 'Put'].copy()
    call_local_df, call_interp, calls_smoothed = process_options(calls, 'Call', r, q)
    put_local_df, put_interp, puts_smoothed = process_options(puts, 'Put', r, q)
    smoothed_df = pd.concat([calls_smoothed, puts_smoothed]).sort_index()
    return call_local_df, put_local_df, call_interp, put_interp, smoothed_df

def find_strike_for_delta(S, T, r, q, sigma, target_delta, option_type):
    def delta_diff(K):
        delta = black_scholes_delta(S, K, T, r, q, sigma, option_type)
        return delta - target_delta if option_type.lower() == 'call' else delta - (-target_delta)
    try:
        K = brentq(delta_diff, S * 0.5, S * 2.0)
        return K
    except ValueError:
        return np.nan

def calculate_skew_metrics(df, call_interp, put_interp, S, r, q, ticker):
    if df.empty or call_interp is None or put_interp is None:
        with open('data_error.log', 'a') as f:
            f.write(f"Insufficient data for skew metrics in {ticker}\n")
        return pd.DataFrame(), pd.DataFrame()
    def get_iv(interp, y, T):
        if interp is None or T <= 0:
            return np.nan
        w = interp(np.array([[y, T]]))[0]
        if np.isnan(w) or w <= 0:
            return np.nan
        return np.sqrt(w / T)
    skew_data = []
    target_deltas = [0.25, 0.75]
    target_terms = [0.25, 1.0]
    for exp in sorted(df['Expiry'].unique()):
        T = df[df['Expiry'] == exp]['Years_to_Expiry'].iloc[0] if not df[df['Expiry'] == exp].empty else np.nan
        if np.isnan(T):
            continue
        atm_iv = get_iv(call_interp, 0.0, T)
        if np.isnan(atm_iv):
            continue
        call_strike_25 = find_strike_for_delta(S, T, r, q, atm_iv, 0.25, 'call')
        call_strike_75 = find_strike_for_delta(S, T, r, q, atm_iv, 0.75, 'call')
        put_strike_25 = find_strike_for_delta(S, T, r, q, atm_iv, 0.25, 'put')
        put_strike_75 = find_strike_for_delta(S, T, r, q, atm_iv, 0.75, 'put')
        iv_call_25 = get_iv(call_interp, np.log(call_strike_25 / (S * np.exp((r - q) * T))), T) if not np.isnan(call_strike_25) else np.nan
        iv_call_75 = get_iv(call_interp, np.log(call_strike_75 / (S * np.exp((r - q) * T))), T) if not np.isnan(call_strike_75) else np.nan
        iv_put_25 = get_iv(put_interp, np.log(put_strike_25 / (S * np.exp((r - q) * T))), T) if not np.isnan(put_strike_25) else np.nan
        iv_put_75 = get_iv(put_interp, np.log(put_strike_75 / (S * np.exp((r - q) * T))), T) if not np.isnan(put_strike_75) else np.nan
        skew_25 = iv_put_25 / iv_call_25 if not np.isnan(iv_put_25) and not np.isnan(iv_call_25) and iv_call_25 > 0 else np.nan
        skew_75 = iv_put_75 / iv_call_75 if not np.isnan(iv_put_75) and not np.isnan(iv_call_75) and iv_call_75 > 0 else np.nan
        skew_call_25_75 = iv_call_25 / iv_call_75 if not np.isnan(iv_call_25) and not np.isnan(iv_call_75) and iv_call_75 > 0 else np.nan
        skew_put_25_75 = iv_put_25 / iv_put_75 if not np.isnan(iv_put_25) and not np.isnan(iv_put_75) and iv_put_75 > 0 else np.nan
        skew_data.append({
            'Expiry': exp,
            'Skew_25_delta': skew_25,
            'Skew_75_delta': skew_75,
            'Skew_call_25_75': skew_call_25_75,
            'Skew_put_25_75': skew_put_25_75,
            'IV_put_25_delta': iv_put_25,
            'IV_put_75_delta': iv_put_75,
            'IV_call_25_delta': iv_call_25,
            'IV_call_75_delta': iv_call_75,
            'Strike_call_25_delta': call_strike_25,
            'Strike_call_75_delta': call_strike_75,
            'Strike_put_25_delta': put_strike_25,
            'Strike_put_75_delta': put_strike_75
        })
    slope_data = []
    for delta in target_deltas:
        for opt_type in ['call', 'put']:
            interp = call_interp if opt_type == 'call' else put_interp
            iv_3m = get_iv(interp, 0.0, 0.25)
            if np.isnan(iv_3m):
                continue
            strike_3m = find_strike_for_delta(S, 0.25, r, q, iv_3m, delta, opt_type)
            log_moneyness_3m = np.log(strike_3m / (S * np.exp((r - q) * 0.25))) if not np.isnan(strike_3m) else np.nan
            iv_3m_delta = get_iv(interp, log_moneyness_3m, 0.25) if not np.isnan(log_moneyness_3m) else np.nan
            iv_12m = get_iv(interp, 0.0, 1.0)
            if np.isnan(iv_12m):
                continue
            strike_12m = find_strike_for_delta(S, 1.0, r, q, iv_12m, delta, opt_type)
            log_moneyness_12m = np.log(strike_12m / (S * np.exp((r - q) * 1.0))) if not np.isnan(strike_12m) else np.nan
            iv_12m_delta = get_iv(interp, log_moneyness_12m, 1.0) if not np.isnan(log_moneyness_12m) else np.nan
            slope = (iv_12m_delta - iv_3m_delta) / (1.0 - 0.25) if not np.isnan(iv_3m_delta) and not np.isnan(iv_12m_delta) else np.nan
            slope_data.append({
                'Delta': delta,
                'Type': opt_type.capitalize(),
                'IV_Slope_3m_12m': slope,
                'IV_3m': iv_3m_delta,
                'IV_12m': iv_12m_delta,
                'Strike_3m': strike_3m,
                'Strike_12m': strike_12m
            })
    skew_metrics_df = pd.DataFrame(skew_data)
    slope_metrics_df = pd.DataFrame(slope_data)
    atm_iv_3m = get_iv(call_interp, 0.0, 0.25)
    atm_iv_12m = get_iv(call_interp, 0.0, 1.0)
    atm_ratio = atm_iv_12m / atm_iv_3m if not np.isnan(atm_iv_3m) and not np.isnan(atm_iv_12m) and atm_iv_3m > 0 else np.nan
    skew_metrics_df.loc[:, 'ATM_12m_3m_Ratio'] = atm_ratio
    skew_metrics_df.loc[:, 'ATM_IV_3m'] = atm_iv_3m
    skew_metrics_df.loc[:, 'ATM_IV_12m'] = atm_iv_12m
    if skew_metrics_df.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"No skew metrics generated for {ticker} in calculate_skew_metrics\n")
    return skew_metrics_df, slope_metrics_df

def process_ticker(ticker, df, full_df, r, timestamp):
    if df.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"Empty input DataFrame for {ticker}\n")
        return df, pd.DataFrame(), pd.DataFrame(), None
    required_cols = ['Ticker', 'Expiry', 'Strike', 'Bid', 'Ask', 'Type', 'Contract Name']
    if not all(col in df.columns for col in required_cols):
        with open('data_error.log', 'a') as f:
            f.write(f"Missing columns in {ticker} data: {set(required_cols) - set(df.columns)}\n")
        return df, pd.DataFrame(), pd.DataFrame(), None
    try:
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_full = full_df[full_df['Ticker'] == ticker].copy()
        rvol100d = load_rvol_from_historic(ticker, timestamp)
        ticker_df, S, r, q = calculate_iv_mid(ticker_df, ticker, r, timestamp)
        if ticker_df.empty:
            with open('data_error.log', 'a') as f:
                f.write(f"No valid data after IV calculation for {ticker}\n")
            return ticker_df, pd.DataFrame(), pd.DataFrame(), None
        ticker_df = calc_Ivol_Rvol(ticker_df, rvol100d)
        ticker_df, skew_df, slope_df = calculate_metrics(ticker_df, ticker, r)
        try:
            call_local_df, put_local_df, call_interp, put_interp, smoothed_df = calculate_local_vol_from_iv(ticker_df, S, r, q, ticker)
            ticker_df = smoothed_df
        except Exception as e:
            with open('data_error.log', 'a') as f:
                f.write(f"Local vol calculation failed for {ticker}: {str(e)}\n")
            call_local_df, put_local_df, call_interp, put_interp = pd.DataFrame(), pd.DataFrame(), None, None
        skew_metrics_df, slope_metrics_df = calculate_skew_metrics(ticker_df, call_interp, put_interp, S, r, q, ticker)
        skew_metrics_df.loc[:, 'Ticker'] = ticker
        slope_metrics_df.loc[:, 'Ticker'] = ticker
        if not call_local_df.empty:
            ticker_df = ticker_df.merge(
                call_local_df.rename(columns={'Local Vol': 'Call Local Vol'}),
                on=['Strike', 'Expiry'],
                how='left'
            )
        else:
            ticker_df.loc[:, 'Call Local Vol'] = np.nan
        if not put_local_df.empty:
            ticker_df = ticker_df.merge(
                put_local_df.rename(columns={'Local Vol': 'Put Local Vol'}),
                on=['Strike', 'Expiry'],
                how='left'
            )
        else:
            ticker_df.loc[:, 'Put Local Vol'] = np.nan
        ticker_df.loc[:, 'Realised Vol 100d'] = rvol100d if rvol100d is not None else np.nan
        vol_fit_params, vol_fit_residuals = fit_single_ticker(ticker_df[ticker_df['Type'] == 'Call'], model='hyp')
        vol_fit_data = {
            'Ticker': ticker,
            'a0': vol_fit_params[0] if vol_fit_params is not None else np.nan,
            'a1': vol_fit_params[1] if vol_fit_params is not None else np.nan,
            'b0': vol_fit_params[2] if vol_fit_params is not None else np.nan,
            'b1': vol_fit_params[3] if vol_fit_params is not None else np.nan,
            'm0': vol_fit_params[4] if vol_fit_params is not None else np.nan,
            'm1': vol_fit_params[5] if vol_fit_params is not None else np.nan,
            'rho0': vol_fit_params[6] if vol_fit_params is not None else np.nan,
            'rho1': vol_fit_params[7] if vol_fit_params is not None else np.nan,
            'sigma0': vol_fit_params[8] if vol_fit_params is not None else np.nan,
            'sigma1': vol_fit_params[9] if vol_fit_params is not None else np.nan,
            'c': vol_fit_params[10] if vol_fit_params is not None else np.nan,
            'Residuals': vol_fit_residuals if vol_fit_residuals is not None else np.nan
        }
        with open('data_error.log', 'a') as f:
            if not ticker_df.empty:
                f.write(f"Generated processed data for {ticker}: {len(ticker_df)} rows\n")
            if not skew_df.empty:
                f.write(f"Generated skew metrics for {ticker}: {len(skew_df)} rows\n")
            if not slope_df.empty:
                f.write(f"Generated slope metrics for {ticker}: {len(slope_df)} rows\n")
            if not skew_metrics_df.empty:
                f.write(f"Generated skew metrics (delta-based) for {ticker}: {len(skew_metrics_df)} rows\n")
            if vol_fit_data['Residuals'] is not np.nan:
                f.write(f"Generated vol fit data for {ticker}\n")
            else:
                f.write(f"No vol fit data for {ticker} (insufficient points or fitting failed)\n")
        return ticker_df, skew_df, slope_df, vol_fit_data
    except Exception as e:
        with open('data_error.log', 'a') as f:
            f.write(f"Error processing {ticker}: {str(e)}\n")
        return df, pd.DataFrame(), pd.DataFrame(), None

def fetch_tnx_data(timestamp):
    """Fetch ^TNX data from yfinance and save to historic file."""
    tnx_file = f'data/{timestamp}/historic/historic_^TNX.csv'
    try:
        tnx = yf.download('^TNX', period='1d', progress=False)
        if tnx.empty:
            with open('data_error.log', 'a') as f:
                f.write(f"yfinance returned empty data for ^TNX\n")
            return None
        tnx_data = tnx.reset_index()
        tnx_data = tnx_data[['Date', 'Close']]
        tnx_data['Close'] = pd.to_numeric(tnx_data['Close'], errors='coerce')
        os.makedirs(os.path.dirname(tnx_file), exist_ok=True)
        tnx_data.to_csv(tnx_file, index=False)
        with open('data_error.log', 'a') as f:
            f.write(f"Saved ^TNX data to {tnx_file}\n")
        close_price = tnx_data['Close'].iloc[-1]
        if isinstance(close_price, (pd.Series, np.ndarray)):
            close_price = close_price.item()
        return float(close_price) / 100
    except Exception as e:
        with open('data_error.log', 'a') as f:
            f.write(f"Failed to fetch ^TNX from yfinance: {str(e)}\n")
        return None

def process_data(timestamp, prefix="_yfinance"):
    cleaned_dir = f'data/{timestamp}/cleaned{prefix}'
    raw_dir = f'data/{timestamp}/raw{prefix}'
    if not os.path.exists(cleaned_dir):
        with open('data_error.log', 'a') as f:
            f.write(f"No cleaned directory found: {cleaned_dir}\n")
        return None, None, None
    cleaned_files = glob.glob(f'{cleaned_dir}/cleaned{prefix}_*.csv')
    if not cleaned_files:
        with open('data_error.log', 'a') as f:
            f.write(f"No cleaned files found in {cleaned_dir}\n")
        return None, None, None
    # Fetch or load ^TNX data
    tnx_file = f'data/{timestamp}/historic/historic_^TNX.csv'
    if os.path.exists(tnx_file):
        tnx_data = pd.read_csv(tnx_file)
        tnx_data['Close'] = pd.to_numeric(tnx_data['Close'], errors='coerce')
        r = tnx_data['Close'].iloc[-1] / 100 if not tnx_data.empty and not pd.isna(tnx_data['Close'].iloc[-1]) else 0.05
        with open('data_error.log', 'a') as f:
            f.write(f"Loaded ^TNX from {tnx_file}: r={r:.4f}\n")
    else:
        r = fetch_tnx_data(timestamp)
        if r is None:
            r = 0.05
            with open('data_error.log', 'a') as f:
                f.write(f"No ^TNX data fetched, using default r={r:.4f}\n")
        else:
            with open('data_error.log', 'a') as f:
                f.write(f"Fetched ^TNX from yfinance: r={r:.4f}\n")
    processed_dir = f'data/{timestamp}/processed{prefix}'
    skew_dir = f'data/{timestamp}/skew_metrics{prefix}'
    slope_dir = f'data/{timestamp}/slope_metrics{prefix}'
    vol_fit_dir = f'data/{timestamp}/vol_fit'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(skew_dir, exist_ok=True)
    os.makedirs(slope_dir, exist_ok=True)
    os.makedirs(vol_fit_dir, exist_ok=True)
    processed_dfs = []
    skew_metrics_dfs = []
    slope_metrics_dfs = []
    vol_fit_data = []
    for clean_file in cleaned_files:
        ticker = os.path.basename(clean_file).split(f'cleaned{prefix}_')[1].split('.csv')[0]
        raw_file = f'{raw_dir}/raw{prefix}_{ticker}.csv'
        if not os.path.exists(raw_file):
            with open('data_error.log', 'a') as f:
                f.write(f"No raw file found for {ticker}: {raw_file}\n")
            continue
        try:
            df = pd.read_csv(clean_file, parse_dates=['Expiry'])
            full_df = pd.read_csv(raw_file, parse_dates=['Expiry'])
            if df.empty:
                with open('data_error.log', 'a') as f:
                    f.write(f"Empty cleaned file for {ticker}: {clean_file}\n")
                continue
            ticker_df, skew_df, slope_df, vol_fit_row = process_ticker(ticker, df, full_df, r, timestamp)
            if not ticker_df.empty:
                processed_filename = f'{processed_dir}/processed{prefix}_{ticker}.csv'
                ticker_df.to_csv(processed_filename, index=False)
                processed_dfs.append(ticker_df)
            if not skew_df.empty:
                skew_filename = f'{skew_dir}/skew_metrics{prefix}_{ticker}.csv'
                skew_df.to_csv(skew_filename, index=False)
                skew_metrics_dfs.append(skew_df)
            if not slope_df.empty:
                slope_filename = f'{slope_dir}/slope_metrics{prefix}_{ticker}.csv'
                slope_df.to_csv(slope_filename, index=False)
                slope_metrics_dfs.append(slope_df)
            if vol_fit_row is not None:
                vol_fit_data.append(vol_fit_row)
        except Exception as e:
            with open('data_error.log', 'a') as f:
                f.write(f"Error processing {ticker} file {clean_file}: {str(e)}\n")
            continue
    vol_fit_df = pd.DataFrame(vol_fit_data)
    vol_fit_filename = f'{vol_fit_dir}/vol_fit.csv'
    vol_fit_df.to_csv(vol_fit_filename, index=False)
    with open('data_error.log', 'a') as f:
        f.write(f"Generated vol_fit.csv with {len(vol_fit_data)} tickers\n")
    return processed_dfs, skew_metrics_dfs, slope_metrics_dfs

def main():
    dates_file = 'data/dates.json'
    if not os.path.exists(dates_file):
        with open('data_error.log', 'a') as f:
            f.write("No dates.json found\n")
        return
    with open(dates_file, 'r') as f:
        dates = json.load(f)
    if not dates:
        with open('data_error.log', 'a') as f:
            f.write("Empty dates.json\n")
        return
    timestamp = max(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
    process_data(timestamp, prefix="_yfinance")

main()
