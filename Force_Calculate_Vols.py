import sys
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import json
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import interp1d
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
    if pd.isna(price) or price <= 0 or T <= 0:
        with open('data_error.log', 'a') as f:
            f.write(f"Invalid price ({price}) or T ({T}) for {contract_name}\n")
        return np.nan
    intrinsic = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
    if price < intrinsic * np.exp(-r * T) * 0.8:
        with open('data_error.log', 'a') as f:
            f.write(f"Price ({price}) below relaxed intrinsic ({intrinsic * np.exp(-r * T) * 0.8}) for {contract_name}\n")
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

def load_rvol_from_historic(ticker, timestamp, days=100):
    historic_file = f'data/{timestamp}/historic/historic_{ticker}.csv'
    if not os.path.exists(historic_file):
        with open('data_error.log', 'a') as f:
            f.write(f"No historic file found for {ticker}: {historic_file}\n")
        return None
    df_hist = pd.read_csv(historic_file, parse_dates=['Date'])
    if df_hist.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"Empty historic file for {ticker}: {historic_file}\n")
        return None
    col = f'Realised_Vol_Close_{days}'
    if col not in df_hist.columns:
        with open('data_error.log', 'a') as f:
            f.write(f"No {col} column in historic file for {ticker}\n")
        return None
    latest_vol = df_hist[col].iloc[-1] / 100
    if pd.isna(latest_vol):
        with open('data_error.log', 'a') as f:
            f.write(f"NaN realized volatility for {ticker} in {col}\n")
        return None
    with open('data_error.log', 'a') as f:
        f.write(f"Loaded realized volatility for {ticker}: {latest_vol:.4f}\n")
    return latest_vol

def calc_Ivol_Rvol(df, rvol100d):
    if df.empty:
        with open('data_error.log', 'a') as f:
            f.write("Empty DataFrame in calc_Ivol_Rvol\n")
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
        with open('data_error.log', 'a') as f:
            f.write(f"Empty DataFrame for {ticker} in calculate_metrics\n")
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
                time_diff = (subset["Expiry_dt"] - subset["Expiry_dt"].shift(1)).dt.days / 365.0
                slope = iv_diff / time_diff
                for i in range(1, len(subset)):
                    slope_data.append({
                        "Strike": strike,
                        "Type": opt_type,
                        "Expiry": subset["Expiry"].iloc[i],
                        "IV Slope": slope.iloc[i]
                    })
    slope_df = pd.DataFrame(slope_data)
    with open('data_error.log', 'a') as f:
        f.write(f"Generated {len(skew_data)} skew metrics and {len(slope_data)} slope metrics for {ticker}\n")
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
    invalid_prices = df[(df['Bid'].isna()) | (df['Ask'].isna()) | (df['Bid'] < 0) | (df['Ask'] < 0)]
    if not invalid_prices.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"Invalid prices for {ticker}: {len(invalid_prices)} rows with Bid < 0 or Ask < 0 or NaN\n")
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
    try:
        today = pd.to_datetime(timestamp, format='%Y%m%d_%H%M')
    except ValueError as e:
        with open('data_error.log', 'a') as f:
            f.write(f"Invalid timestamp format {timestamp} for {ticker}: {str(e)}, using default date\n")
        today = datetime.today()
    df.loc[:, "Expiry_dt"] = pd.to_datetime(df["Expiry"])
    df.loc[:, 'Years_to_Expiry'] = (df['Expiry_dt'] - today).dt.days / 365.25
    invalid_expiry = df[df['Years_to_Expiry'] <= 0]
    if not invalid_expiry.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"Found {len(invalid_expiry)} rows with Years_to_Expiry <= 0 for {ticker}, setting IV columns to NaN\n")
        df.loc[df['Years_to_Expiry'] <= 0, ['IV_bid', 'IV_ask', 'IV_mid', 'IV_spread', 'Delta']] = np.nan
    df.loc[:, 'Forward'] = S * np.exp((r - q) * df['Years_to_Expiry'].clip(lower=0.0001))
    df.loc[:, 'Moneyness'] = df['Strike'] / df['Forward']
    df.loc[:, 'LogMoneyness'] = np.log(df['Strike'] / df['Forward'])
    if 'Last Stock Price' not in df.columns:
        df.loc[:, 'Last Stock Price'] = S
    if 'Implied Volatility' in df.columns:
        df = df.rename(columns={'Implied Volatility': 'Input_Implied_Volatility'})
    results = [compute_ivs(row, S, r, q) for _, row in df.iterrows()]
    df.loc[:, ['IV_bid', 'IV_ask', 'IV_mid', 'IV_spread', 'Delta']] = pd.DataFrame(results, index=df.index)
    df.loc[:, 'Implied Volatility'] = df['IV_mid']
    with open('data_error.log', 'a') as f:
        f.write(f"IV_mid stats for {ticker}: valid={len(df[df['IV_mid'].notna() & (df['IV_mid'] > 0)])}, "
                f"nan={df['IV_mid'].isna().sum()}, min={df['IV_mid'].min():.2f}, max={df['IV_mid'].max():.2f}\n")
        f.write(f"Columns after calculate_iv_mid for {ticker}: {list(df.columns)}\n")
    return df, S, r, q

def smooth_iv_per_expiry(options_df):
    if options_df.empty:
        options_df = options_df.copy()
        options_df.loc[:, 'Smoothed_IV'] = np.nan
        with open('data_error.log', 'a') as f:
            f.write(f"Empty DataFrame in smooth_iv_per_expiry for {options_df['Ticker'].iloc[0] if not options_df.empty else 'unknown'}\n")
        return options_df
    options_df = options_df.copy()
    smoothed_iv = pd.Series(np.nan, index=options_df.index, dtype=float)
    for exp, group in options_df.groupby('Expiry'):
        if len(group) < 3 or group['IV_mid'].isna().all():
            smoothed_iv.loc[group.index] = group['IV_mid']
            with open('data_error.log', 'a') as f:
                f.write(f"Insufficient data ({len(group)}) or all NaN IV_mid for expiry {exp} in {options_df['Ticker'].iloc[0]}\n")
            continue
        if len(group) >= 5:
            mean_iv = np.mean(group['IV_mid'][group['IV_mid'].notna()])
            std_iv = np.std(group['IV_mid'][group['IV_mid'].notna()])
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
            with open('data_error.log', 'a') as f:
                f.write(f"Insufficient cleaned data ({len(cleaned_group)}) or all NaN IV_mid for expiry {exp} in {options_df['Ticker'].iloc[0]}\n")
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
            valid_idx = ~np.isnan(x) & ~np.isnan(y)
            x = x[valid_idx]
            y = y[valid_idx]
            if len(x) < 3:
                smoothed_iv.loc[group.index] = group['IV_mid']
                with open('data_error.log', 'a') as f:
                    f.write(f"Insufficient valid points ({len(x)}) for LOWESS in expiry {exp} for {options_df['Ticker'].iloc[0]}\n")
                continue
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
    with open('data_error.log', 'a') as f:
        f.write(f"Smoothed IV for {options_df['Ticker'].iloc[0]}: {len(options_df)} rows\n")
    return options_df

def find_strike_for_delta(S, T, r, q, sigma, target_delta, option_type):
    def delta_diff(K):
        delta = black_scholes_delta(S, K, T, r, q, sigma, option_type)
        return delta - target_delta if option_type.lower() == 'call' else delta - (-target_delta)
    try:
        K = brentq(delta_diff, S * 0.5, S * 2.0)
        return K
    except ValueError:
        with open('data_error.log', 'a') as f:
            f.write(f"Failed to find strike for delta {target_delta} in {option_type}\n")
        return np.nan

def calculate_skew_metrics(df, S, r, q, ticker):
    if df.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"Empty DataFrame for skew metrics in {ticker}\n")
        return pd.DataFrame(), pd.DataFrame()
    calls = df[df['Type'] == 'Call'].copy()
    puts = df[df['Type'] == 'Put'].copy()
    calls = calls[calls['IV_mid'].notna() & (calls['IV_mid'] > 0)]
    puts = puts[puts['IV_mid'].notna() & (puts['IV_mid'] > 0)]
    if calls.empty or puts.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"Insufficient valid IV data for {ticker}: calls={len(calls)}, puts={len(puts)}\n")
        return pd.DataFrame(), pd.DataFrame()
    calls = calls.sort_values(['Years_to_Expiry', 'LogMoneyness'])
    puts = puts.sort_values(['Years_to_Expiry', 'LogMoneyness'])
    call_points = np.column_stack((calls['LogMoneyness'], calls['Years_to_Expiry']))
    put_points = np.column_stack((puts['LogMoneyness'], puts['Years_to_Expiry']))
    call_values = calls['Smoothed_IV'].values
    put_values = puts['Smoothed_IV'].values
    try:
        call_interp = LinearNDInterpolator(call_points, call_values, fill_value=np.nan, rescale=True)
        put_interp = LinearNDInterpolator(put_points, put_values, fill_value=np.nan, rescale=True)
    except Exception as e:
        with open('data_error.log', 'a') as f:
            f.write(f"Interpolation failed for {ticker}: {str(e)}\n")
        return pd.DataFrame(), pd.DataFrame()
    def get_iv(interp, y, T):
        if interp is None or T <= 0:
            with open('data_error.log', 'a') as f:
                f.write(f"Invalid interp or T ({T}) for skew metrics\n")
            return np.nan
        try:
            iv = interp(np.array([[y, T]]))[0]
            if np.isnan(iv) or iv <= 0:
                with open('data_error.log', 'a') as f:
                    f.write(f"Invalid IV ({iv}) for y={y}, T={T}\n")
                return np.nan
            return iv
        except Exception as e:
            with open('data_error.log', 'a') as f:
                f.write(f"Interpolation failed for y={y}, T={T} in {ticker}: {str(e)}\n")
            return np.nan
    skew_data = []
    target_deltas = [0.25, 0.75]
    for exp in sorted(df['Expiry'].unique()):
        T = df[df['Expiry'] == exp]['Years_to_Expiry'].iloc[0] if not df[df['Expiry'] == exp].empty else np.nan
        if np.isnan(T):
            with open('data_error.log', 'a') as f:
                f.write(f"NaN Years_to_Expiry for expiry {exp} in {ticker}\n")
            continue
        atm_iv = get_iv(call_interp, 0.0, T)
        if np.isnan(atm_iv):
            with open('data_error.log', 'a') as f:
                f.write(f"NaN ATM IV for T={T} in {ticker}\n")
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
                with open('data_error.log', 'a') as f:
                    f.write(f"NaN 3m IV for {opt_type} delta {delta} in {ticker}\n")
                continue
            strike_3m = find_strike_for_delta(S, 0.25, r, q, iv_3m, delta, opt_type)
            log_moneyness_3m = np.log(strike_3m / (S * np.exp((r - q) * 0.25))) if not np.isnan(strike_3m) else np.nan
            iv_3m_delta = get_iv(interp, log_moneyness_3m, 0.25) if not np.isnan(log_moneyness_3m) else np.nan
            iv_12m = get_iv(interp, 0.0, 1.0)
            if np.isnan(iv_12m):
                with open('data_error.log', 'a') as f:
                    f.write(f"NaN 12m IV for {opt_type} delta {delta} in {ticker}\n")
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
    with open('data_error.log', 'a') as f:
        f.write(f"Generated {len(skew_data)} skew metrics and {len(slope_data)} delta-based slope metrics for {ticker}\n")
    return skew_metrics_df, slope_metrics_df

def process_ticker(ticker, df, full_df, r, timestamp):
    with open('data_error.log', 'a') as f:
        f.write(f"Processing calculations for {ticker}...\n")
    ticker_df = df[df['Ticker'] == ticker].copy()
    ticker_full = full_df[full_df['Ticker'] == ticker].copy()
    if ticker_df.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"No data for ticker {ticker} in df\n")
        return None, None, None
    with open('data_error.log', 'a') as f:
        f.write(f"Processing {ticker}: {len(ticker_df)} rows in cleaned data, {len(ticker_full)} rows in raw data\n")
    rvol100d = load_rvol_from_historic(ticker, timestamp)
    with open('data_error.log', 'a') as f:
        f.write(f"Realised Volatility for {ticker}: 100-day: {rvol100d * 100:.2f}% if not None else 'N/A'\n")
    ticker_df, S, r, q = calculate_iv_mid(ticker_df, ticker, r, timestamp)
    if ticker_df.empty:
        with open('data_error.log', 'a') as f:
            f.write(f"No valid data after IV calculation for {ticker}\n")
        return None, None, None
    with open('data_error.log', 'a') as f:
        f.write(f"Post-IV calculation columns for {ticker}: {list(ticker_df.columns)}\n")
    ticker_df = calc_Ivol_Rvol(ticker_df, rvol100d)
    ticker_df = smooth_iv_per_expiry(ticker_df)
    ticker_df, skew_df, slope_df = calculate_metrics(ticker_df, ticker, r)
    skew_metrics_df, slope_metrics_df = calculate_skew_metrics(ticker_df, S, r, q, ticker)
    skew_metrics_df.loc[:, 'Ticker'] = ticker
    slope_metrics_df.loc[:, 'Ticker'] = ticker
    ticker_df.loc[:, 'Realised Vol 100d'] = rvol100d if rvol100d is not None else np.nan
    with open('data_error.log', 'a') as f:
        f.write(f"Final columns for {ticker} before saving: {list(ticker_df.columns)}\n")
    return ticker_df, skew_df, slope_metrics_df

def fetch_tnx_data(timestamp):
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
    processed_dir = f'data/{timestamp}/processed{prefix}'
    skew_dir = f'data/{timestamp}/skew_metrics{prefix}'
    slope_dir = f'data/{timestamp}/slope_metrics{prefix}'
    try:
        os.makedirs(cleaned_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(skew_dir, exist_ok=True)
        os.makedirs(slope_dir, exist_ok=True)
        with open('data_error.log', 'a') as f:
            f.write(f"Created directories: {cleaned_dir}, {raw_dir}, {processed_dir}, {skew_dir}, {slope_dir}\n")
    except Exception as e:
        with open('data_error.log', 'a') as f:
            f.write(f"Failed to create directories: {str(e)}\n")
        return None, None, None
    tickers_file = 'tickers.txt'
    if not os.path.exists(tickers_file):
        with open('data_error.log', 'a') as f:
            f.write(f"No tickers.txt found\n")
        return None, None, None
    with open(tickers_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    with open('data_error.log', 'a') as f:
        f.write(f"Read {len(tickers)} tickers from {tickers_file}: {tickers}\n")
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
    processed_dfs = []
    skew_metrics_dfs = []
    slope_metrics_dfs = []
    for ticker in tickers:
        clean_file = f'{cleaned_dir}/cleaned{prefix}_{ticker}.csv'
        raw_file = f'{raw_dir}/raw{prefix}_{ticker}.csv'
        if not os.path.exists(clean_file):
            with open('data_error.log', 'a') as f:
                f.write(f"No cleaned file found for {ticker}: {clean_file}\n")
            continue
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
            ticker_df, skew_df, slope_metrics_df = process_ticker(ticker, df, full_df, r, timestamp)
            if ticker_df is not None:
                processed_filename = f'{processed_dir}/processed{prefix}_{ticker}.csv'
                ticker_df.to_csv(processed_filename, index=False)
                with open('data_error.log', 'a') as f:
                    f.write(f"Saved processed data for {ticker}: {len(ticker_df)} rows to {processed_filename}\n")
                processed_dfs.append(ticker_df)
            else:
                with open('data_error.log', 'a') as f:
                    f.write(f"Empty ticker_df for {ticker}, not saving processed file\n")
            if not skew_df.empty:
                skew_filename = f'{skew_dir}/strike_skew_metrics{prefix}_{ticker}.csv'
                skew_df.to_csv(skew_filename, index=False)
                with open('data_error.log', 'a') as f:
                    f.write(f"Saved strike skew metrics for {ticker}: {len(skew_df)} rows to {skew_filename}\n")
            else:
                with open('data_error.log', 'a') as f:
                    f.write(f"Empty skew_df for {ticker}, not saving strike skew metrics\n")
            if not slope_metrics_df.empty:
                slope_metrics_filename = f'{slope_dir}/slope_metrics{prefix}_{ticker}.csv'
                slope_metrics_df.to_csv(slope_metrics_filename, index=False)
                with open('data_error.log', 'a') as f:
                    f.write(f"Saved delta-based slope metrics for {ticker}: {len(slope_metrics_df)} rows to {slope_metrics_filename}\n")
                slope_metrics_dfs.append(slope_metrics_df)
            else:
                with open('data_error.log', 'a') as f:
                    f.write(f"Empty slope_metrics_df for {ticker}, not saving delta-based slope metrics\n")
        except Exception as e:
            with open('data_error.log', 'a') as f:
                f.write(f"Error processing {ticker} file {clean_file}: {str(e)}\n")
            continue
    return processed_dfs, skew_metrics_dfs, slope_metrics_dfs

def main():
    if len(sys.argv) < 2:
        print("Usage: python Force_Calculate_Vols.py <timestamp>")
        sys.exit(1)
    timestamp = sys.argv[1]
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
        with open('data_error.log', 'a') as f:
            f.write(f"Updated dates.json with timestamp: {timestamp}\n")
    process_data(timestamp, prefix="_yfinance")

main()
