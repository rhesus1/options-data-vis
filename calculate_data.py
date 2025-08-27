import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import json
from scipy.optimize import brentq
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
        return np.nan
    intrinsic = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
    if price < intrinsic * np.exp(-r * T):
        return np.nan
    def objective(sigma):
        if option_type.lower() == 'call':
            return black_scholes_call(S, K, T, r, q, sigma) - price
        else:
            return black_scholes_put(S, K, T, r, q, sigma) - price
    try:
        iv = brentq(objective, 0.0001, 50.0)
        return np.clip(iv, 0.05, 5.0)
    except ValueError:
        return np.nan

def calculate_rvol_days(ticker, days):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < days + 1:
            return None
        hist_last = hist.iloc[-(days + 1):]
        log_returns = np.log(hist_last["Close"] / hist_last["Close"].shift(1)).dropna()
        if len(log_returns) < 2:
            return None
        realised_vol = np.std(log_returns, ddof=1) * np.sqrt(252)
        return realised_vol
    except Exception:
        return None

def calc_Ivol_Rvol(df, rvol100d):
    if df.empty:
        return df
    df["Ivol/Rvol100d Ratio"] = df["IV_mid"] / rvol100d
    return df

def compute_ivs(row, S, r, q):
    if pd.isna(row['Years_to_Expiry']):
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
                subset["Expiry_dt"] = pd.to_datetime(subset["Expiry"])
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

def calculate_iv_mid(df, ticker, r):
    if df.empty:
        return df, None, None, None
    stock = yf.Ticker(ticker)
    S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2
    q = float(stock.info.get('trailingAnnualDividendYield', 0.0))
    today = datetime.today()
    df["Expiry_dt"] = df["Expiry"]
    df['Years_to_Expiry'] = (df['Expiry_dt'] - today).dt.days / 365.25
    df['Forward'] = S * np.exp((r - q) * df['Years_to_Expiry'])
    df['Moneyness'] = df['Strike'] / df['Forward']
    df['LogMoneyness'] = np.log(df['Strike'] / df['Forward'])
    df['IV_bid'] = np.nan
    df['IV_ask'] = np.nan
    df['IV_mid'] = np.nan
    df['IV_spread'] = np.nan
    df['Delta'] = np.nan
    results = Parallel(n_jobs=-1, backend='threading')(delayed(compute_ivs)(row, S, r, q) for _, row in df.iterrows())
    df[['IV_bid', 'IV_ask', 'IV_mid', 'IV_spread', 'Delta']] = pd.DataFrame(results, index=df.index)
    return df, S, r, q

def smooth_iv_per_expiry(options_df, iv_col, S, r, q, exp):
    subset = options_df[options_df['Expiry'] == exp]
    subset['Years_to_Expiry'] = (pd.to_datetime(subset['Expiry']) - datetime.now()).dt.days / 365.25
    subset = subset[subset['Years_to_Expiry'] > 0]
    if len(subset) < 5:
        return subset
    subset = subset.sort_values('Moneyness')
    x = subset['Moneyness'].values
    y = subset[iv_col].values
    f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    new_x = np.linspace(min(x), max(x), num=100)
    new_y = f(new_x)
    smoothed_df = pd.DataFrame({'Moneyness': new_x, iv_col: new_y})
    smoothed_df['Expiry'] = exp
    return smoothed_df

def calculate_local_vol_from_iv(options_df, S, r, q):
    options_df['Expiry'] = pd.to_datetime(options_df['Expiry'])
    expiries = options_df['Expiry'].unique()
    calls = options_df[options_df['Type'] == 'Call']
    puts = options_df[options_df['Type'] == 'Put']
    call_iv_smoothed = Parallel(n_jobs=-1)(delayed(smooth_iv_per_expiry)(calls, 'IV_mid', S, r, q, exp) for exp in expiries)
    put_iv_smoothed = Parallel(n_jobs=-1)(delayed(smooth_iv_per_expiry)(puts, 'IV_mid', S, r, q, exp) for exp in expiries)
    call_iv_smoothed_df = pd.concat(call_iv_smoothed)
    put_iv_smoothed_df = pd.concat(put_iv_smoothed)
    call_iv_smoothed_df['Years_to_Expiry'] = (call_iv_smoothed_df['Expiry'] - datetime.now()).dt.days / 365.25
    put_iv_smoothed_df['Years_to_Expiry'] = (put_iv_smoothed_df['Expiry'] - datetime.now()).dt.days / 365.25
    call_iv_smoothed_df['Strike'] = call_iv_smoothed_df['Moneyness'] * S * np.exp(r * call_iv_smoothed_df['Years_to_Expiry'])
    put_iv_smoothed_df['Strike'] = put_iv_smoothed_df['Moneyness'] * S * np.exp(r * put_iv_smoothed_df['Years_to_Expiry'])
    call_iv_smoothed_df['Local Vol'] = call_iv_smoothed_df['IV_mid']  # Placeholder for actual local vol calculation
    put_iv_smoothed_df['Local Vol'] = put_iv_smoothed_df['IV_mid']  # Placeholder for actual local vol calculation
    return call_iv_smoothed_df, put_iv_smoothed_df

def calculate_skew_metrics(options_df, call_interp, put_interp, S, r, q):
    skew_metrics = []
    slope_metrics = []
    expiries = options_df['Expiry'].unique()
    for exp in expiries:
        T = (exp - datetime.now()).days / 365.25
        if T <= 0:
            continue
        # Example skew calculation
        skew = 0.0  # Replace with actual calculation
        skew_metrics.append({'Expiry': exp, 'Skew': skew})
        # Example slope calculation
        slope = 0.0  # Replace with actual calculation
        slope_metrics.append({'Expiry': exp, 'Slope': slope})
    skew_df = pd.DataFrame(skew_metrics)
    slope_df = pd.DataFrame(slope_metrics)
    return skew_df, slope_df

def process_ticker(ticker, df, full_df, r):
    ticker_df = df[df['Ticker'] == ticker].copy()
    ticker_full = full_df[full_df['Ticker'] == ticker].copy()
    rvol100d = calculate_rvol_days(ticker, 100)
    ticker_df, S, r, q = calculate_iv_mid(ticker_df, ticker, r)
    ticker_df = calc_Ivol_Rvol(ticker_df, rvol100d)
    ticker_df, skew_df, slope_df = calculate_metrics(ticker_df, ticker, r)
    call_local_df, put_local_df = calculate_local_vol_from_iv(ticker_df, S, r, q)
    skew_metrics_df, slope_metrics_df = calculate_skew_metrics(ticker_df, call_interp, put_interp, S, r, q)
    skew_metrics_df['Ticker'] = ticker
    slope_metrics_df['Ticker'] = ticker
    if not call_local_df.empty:
        ticker_df = ticker_df.merge(
            call_local_df.rename(columns={'Local Vol': 'Call Local Vol'}),
            on=['Strike', 'Expiry'],
            how='left'
        )
    else:
        ticker_df['Call Local Vol'] = np.nan
    if not put_local_df.empty:
        ticker_df = ticker_df.merge(
            put_local_df.rename(columns={'Local Vol': 'Put Local Vol'}),
            on=['Strike', 'Expiry'],
            how='left'
        )
    else:
        ticker_df['Put Local Vol'] = np.nan
    ticker_df['Realised Vol 100d'] = rvol100d if rvol100d is not None else np.nan
    return ticker_df, skew_metrics_df, slope_metrics_df

def process_data(timestamp, prefix=""):
    cleaned_dir = f'data/{timestamp}/cleaned{prefix}'
    raw_dir = f'data/{timestamp}/raw{prefix}'
    if not os.path.exists(cleaned_dir):
        print(f"Cleaned directory {cleaned_dir} not found")
        return None, None, None
    cleaned_files = glob.glob(f'{cleaned_dir}/cleaned_{prefix}*.csv')
    if not cleaned_files:
        print(f"No cleaned files found in {cleaned_dir}")
        return None, None, None
    tnx_data = yf.download('^TNX', period='1d', auto_adjust=True)
    r = tnx_data['Close'].iloc[-1].item() / 100 if not tnx_data.empty else 0.05
    processed_dir = f'data/{timestamp}/processed{prefix}'
    skew_dir = f'data/{timestamp}/skew_metrics{prefix}'
    slope_dir = f'data/{timestamp}/slope_metrics{prefix}'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(skew_dir, exist_ok=True)
    os.makedirs(slope_dir, exist_ok=True)
    processed_dfs = []
    skew_metrics_dfs = []
    slope_metrics_dfs = []
    for clean_file in cleaned_files:
        ticker = os.path.basename(clean_file).split('cleaned_{prefix}')[1].split('.csv')[0]
        raw_file = f'{raw_dir}/raw_{prefix}{ticker}.csv'
        if not os.path.exists(raw_file):
            print(f"Corresponding raw file {raw_file} not found")
            continue
        df = pd.read_csv(clean_file, parse_dates=['Expiry'])
        full_df = pd.read_csv(raw_file, parse_dates=['Expiry'])
        if df.empty:
            print(f"No data for {ticker} in {clean_file}")
            continue
        ticker_df, skew_df, slope_df = process_ticker(ticker, df, full_df, r)
        if not ticker_df.empty:
            processed_filename = f'{processed_dir}/processed_{prefix}{ticker}.csv'
            ticker_df.to_csv(processed_filename, index=False)
            print(f"Processed {prefix}data for {ticker} saved to {processed_filename}")
            processed_json_filename = f'{processed_dir}/processed_{prefix}{ticker}.json'
            ticker_df.to_json(processed_json_filename, orient='records', date_format='iso')
            print(f"Processed {prefix}data for {ticker} saved to {processed_json_filename}")
            processed_dfs.append(ticker_df)
        if not skew_df.empty:
            skew_filename = f'{skew_dir}/skew_metrics_{prefix}{ticker}.csv'
            skew_df.to_csv(skew_filename, index=False)
            print(f"Skew metrics {prefix}for {ticker} saved to {skew_filename}")
            skew_metrics_dfs.append(skew_df)
        if not slope_df.empty:
            slope_filename = f'{slope_dir}/slope_metrics_{prefix}{ticker}.csv'
            slope_df.to_csv(slope_filename, index=False)
            print(f"Slope metrics {prefix}for {ticker} saved to {slope_filename}")
            slope_metrics_dfs.append(slope_df)
    combined_processed = None
    combined_skew_metrics = None
    combined_slope_metrics = None
    if processed_dfs:
        combined_processed = pd.concat(processed_dfs, ignore_index=True)
    if skew_metrics_dfs:
        combined_skew_metrics = pd.concat(skew_metrics_dfs, ignore_index=True)
    if slope_metrics_dfs:
        combined_slope_metrics = pd.concat(slope_metrics_dfs, ignore_index=True)
    return combined_processed, combined_skew_metrics, combined_slope_metrics

def main():
    # Find the latest timestamp folder
    timestamp_dirs = [d for d in glob.glob('data/*') if os.path.isdir(d) and d.split('/')[-1].replace('_', '').isdigit() and len(d.split('/')[-1]) == 13]
    if not timestamp_dirs:
        print("No timestamp folders found")
        return
    latest_timestamp_dir = max(timestamp_dirs, key=os.path.getctime)
    timestamp = os.path.basename(latest_timestamp_dir)
    
    # Initialize dates.json
    dates_file = 'data/dates.json'
    if os.path.exists(dates_file):
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    else:
        dates = []
    
    # Add timestamp if not present
    if timestamp not in dates:
        dates.append(timestamp)
        dates.sort(reverse=True)
        with open(dates_file, 'w') as f:
            json.dump(dates, f)
        print(f"Updated dates list in {dates_file} with timestamp: {timestamp}")
    
    # Process Nasdaq data
    print(f"Processing Nasdaq data for {timestamp}")
    process_data(timestamp, prefix="")
    
    # Process yfinance data
    print(f"Processing yfinance data for {timestamp}")
    process_data(timestamp, prefix="_yfinance")

main()
