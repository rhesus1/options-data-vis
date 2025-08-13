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
from scipy.optimize import minimize
from joblib import Parallel, delayed
import multiprocessing
import sys
import warnings
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

def implied_vol(price, S, K, T, r, q, option_type, contract_name=""):
    if price <= 0 or T <= 0:
        return 0.0
    intrinsic = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
    if price < intrinsic * np.exp(-r * T):
        return 0.0001
    def objective(sigma):
        if option_type.lower() == 'call':
            return black_scholes_call(S, K, T, r, q, sigma) - price
        else:
            return black_scholes_put(S, K, T, r, q, sigma) - price
    try:
        iv = brentq(objective, 0.0001, 50.0)
        return iv
    except ValueError as e:
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
    except Exception as e:
        return None

def calc_Ivol_Rvol(df, rvol90d):
    if df.empty:
        return df
    df["Ivol/Rvol90d Ratio"] = df["IV_mid"] / rvol90d
    return df

def compute_ivs(row, S, r, q):
    if pd.isna(row['Years_to_Expiry']):
        return np.nan, np.nan, np.nan, np.nan
    T = max(row['Years_to_Expiry'], 0.0001)
    option_type = row['Type'].lower()
    contract_name = row['Contract Name']
    iv_bid = implied_vol(row['Bid'], S, row['Strike'], T, r, q, option_type, contract_name)
    iv_ask = implied_vol(row['Ask'], S, row['Strike'], T, r, q, option_type, contract_name)
    iv_mid = implied_vol(0.5*(row['Bid']+row['Ask']), S, row['Strike'], T, r, q, option_type, contract_name)
    iv_spread = iv_ask - iv_bid if not np.isnan(iv_bid) else np.nan
    return iv_bid, iv_ask, iv_mid, iv_spread

def calculate_metrics(df, ticker, r):
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame(), None, None, None
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
        return df, pd.DataFrame(), pd.DataFrame(), None, None, None
    stock = yf.Ticker(ticker)
    S = stock.history(period='1d')['Close'].iloc[-1]
    q = float(stock.info.get('trailingAnnualDividendYield', 0.0))
    today = datetime.today()
    df["Expiry_dt"] = df["Expiry"]
    df['Years_to_Expiry'] = (df['Expiry_dt'] - today).dt.days / 365.25
    df['IV_bid'] = np.nan
    df['IV_ask'] = np.nan
    df['IV_mid'] = np.nan
    df['IV_spread'] = np.nan
    results = Parallel(n_jobs=-1, backend='threading')(delayed(compute_ivs)(row, S, r, q) for _, row in df.iterrows())
    df[['IV_bid', 'IV_ask', 'IV_mid', 'IV_spread']] = pd.DataFrame(results, index=df.index)
    return df, S, r, q

def compute_local_vol_row(row, r, q, option_type, h_k_base, interp):
    k = row['Strike']
    t = row['T']
    if t <= 0:
        return None
    
    price = interp((k, t))
    if np.isnan(price):
        return None
    
    h_t = 0.01 * t
    h_k = h_k_base
    
    price_T_plus = interp((k, t + h_t))
    price_T_minus = interp((k, max(t - h_t, 0.0001)))  # Avoid negative time
    if np.isnan(price_T_plus) or np.isnan(price_T_minus):
        return None
    dP_dT = (price_T_plus - price_T_minus) / (2 * h_t)
    
    price_K_plus = interp((k + h_k, t))
    price_K_minus = interp((k - h_k, t))
    if np.isnan(price_K_plus) or np.isnan(price_K_minus):
        return None
    dP_dK = (price_K_plus - price_K_minus) / (2 * h_k)
    
    d2P_dK2 = (price_K_plus - 2 * price + price_K_minus) / (h_k ** 2)
    
    if np.isnan(dP_dT) or np.isnan(dP_dK) or np.isnan(d2P_dK2):
        return None
    
    numer = dP_dT + (r - q) * k * dP_dK + q * price
    denom = 0.5 * k ** 2 * d2P_dK2
    
    if denom <= 1e-10 or numer <= 0:
        local_vol = 0.0
    else:
        local_vol_sq = numer / denom
        if local_vol_sq <= 0:
            local_vol = 0.0
        else:
            local_vol = np.sqrt(local_vol_sq)
            if local_vol < 0 or local_vol > 2.0:
                local_vol = np.nan
    
    return {
        "Strike": k,
        "Expiry": row['Expiry'],
        "Local Vol": local_vol
    }

def calculate_local_vol(full_df, S, r, q):
    required_columns = ['Type', 'Strike', 'Expiry', 'Bid', 'Ask']
    if not all(col in full_df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")
    
    calls = full_df[full_df['Type'] == 'Call'].copy()
    puts = full_df[full_df['Type'] == 'Put'].copy()
    
    call_local_df = pd.DataFrame()
    put_local_df = pd.DataFrame()
    
    # Process calls
    if not calls.empty:
        calls['mid_price'] = (calls['Bid'] + calls['Ask']) / 2
        calls['T'] = (calls['Expiry'] - datetime.today()).dt.days / 365.25
        calls = calls[calls['mid_price'] > 0]
        calls = calls[calls['T'] > 0]
        calls = calls.sort_values(['T', 'Strike'])
        call_points = np.column_stack((calls['Strike'], calls['T']))
        call_values = calls['mid_price'].values
        
        strikes = np.sort(np.unique(calls['Strike']))
        if len(strikes) > 1:
            h_k_base = np.median(np.diff(strikes))
        else:
            h_k_base = 0.005 * np.mean(strikes) if len(strikes) > 0 else 1.0
        h_k_base = max(h_k_base, 0.1)
        
        if len(calls) >= 3:
            try:
                call_interp = CloughTocher2DInterpolator(call_points, call_values, fill_value=np.nan, rescale=True)
            except Exception as e:
                print(f"Warning: Interpolator fit failed for calls: {e}. Using linear fallback.")
                from scipy.interpolate import LinearNDInterpolator
                call_interp = LinearNDInterpolator(call_points, call_values, fill_value=np.nan, rescale=True)
            
            call_local_data = Parallel(n_jobs=-1, backend='threading')(
                delayed(compute_local_vol_row)(row, r, q, 'Call', h_k_base, call_interp)
                for _, row in calls.iterrows()
            )
            call_local_data = [d for d in call_local_data if d is not None]
            if call_local_data:
                call_local_df = pd.DataFrame(call_local_data)
    
    # Process puts (similar to calls)
    if not puts.empty:
        puts['mid_price'] = (puts['Bid'] + puts['Ask']) / 2
        puts['T'] = (puts['Expiry'] - datetime.today()).dt.days / 365.25
        puts = puts[puts['mid_price'] > 0]
        puts = puts[puts['T'] > 0]
        puts = puts.sort_values(['T', 'Strike'])
        put_points = np.column_stack((puts['Strike'], puts['T']))
        put_values = puts['mid_price'].values
        
        strikes = np.sort(np.unique(puts['Strike']))
        if len(strikes) > 1:
            h_k_base = np.median(np.diff(strikes))
        else:
            h_k_base = 0.005 * np.mean(strikes) if len(strikes) > 0 else 1.0
        h_k_base = max(h_k_base, 0.1)
        
        if len(puts) >= 3:
            try:
                put_interp = CloughTocher2DInterpolator(put_points, put_values, fill_value=np.nan, rescale=True)
            except Exception as e:
                print(f"Warning: Interpolator fit failed for puts: {e}. Using linear fallback.")
                from scipy.interpolate import LinearNDInterpolator
                put_interp = LinearNDInterpolator(put_points, put_values, fill_value=np.nan, rescale=True)
            
            put_local_data = Parallel(n_jobs=-1, backend='threading')(
                delayed(compute_local_vol_row)(row, r, q, 'Put', h_k_base, put_interp)
                for _, row in puts.iterrows()
            )
            put_local_data = [d for d in put_local_data if d is not None]
            if put_local_data:
                put_local_df = pd.DataFrame(put_local_data)
    
    return call_local_df, put_local_df

def process_ticker(ticker, df, full_df, r):
    print(f"Processing calculations for {ticker}...")
    ticker_df = df[df['Ticker'] == ticker].copy()
    ticker_full = full_df[full_df['Ticker'] == ticker].copy()
    if ticker_df.empty:
        print(f"Warning: No data for ticker {ticker} in df")
        return None
    rvol90d = calculate_rvol_days(ticker, 90)
    print(f"\nRealised Volatility for {ticker}:")
    print(f"90-day: {rvol90d * 100:.2f}%" if rvol90d is not None else "90-day: N/A")
    ticker_df, S, r, q = calculate_iv_mid(ticker_df, ticker, r)
    ticker_df = calc_Ivol_Rvol(ticker_df, rvol90d)
    ticker_df, skew_df, slope_df = calculate_metrics(ticker_df, ticker, r)
    call_local_df, put_local_df = calculate_local_vol(ticker_full, S, r, q)
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
   
    ticker_df['Realised Vol 90d'] = rvol90d if rvol90d is not None else np.nan
    return ticker_df

def main():
    clean_files = glob.glob('data/cleaned_*.csv')
    if not clean_files:
        print("No cleaned data files found")
        return
    latest_clean = max(clean_files, key=os.path.getctime)
    df = pd.read_csv(latest_clean, parse_dates=['Expiry'])
    timestamp = os.path.basename(latest_clean).split('cleaned_')[1].split('.csv')[0]
    raw_file = f'data/raw_{timestamp}.csv'
    if not os.path.exists(raw_file):
        print(f"Corresponding raw file {raw_file} not found")
        return
    full_df = pd.read_csv(raw_file, parse_dates=['Expiry'])
    tickers = df['Ticker'].unique()
    if len(tickers) == 0:
        print("No tickers found")
        return
    tnx_data = yf.download('^TNX', period='1d', auto_adjust=True)
    r = float(tnx_data['Close'].iloc[-1] / 100) if not tnx_data.empty else 0.05
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        processed_dfs = pool.starmap(process_ticker, [(ticker, df, full_df, r) for ticker in tickers])
    processed_dfs = [pdf for pdf in processed_dfs if pdf is not None]
    if processed_dfs:
        combined_processed = pd.concat(processed_dfs, ignore_index=True)
        processed_filename = f'data/processed_{timestamp}.json'
        combined_processed.to_json(processed_filename, orient='records', date_format='iso')
        print(f"Processed data saved to {processed_filename}")
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
    else:
        print("No processed data to save")

main()
