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

def calc_Ivol_Rvol(df, rvol100d):
    if df.empty:
        return df
    df["Ivol/Rvol100d Ratio"] = df["IV_mid"] / rvol100d
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
    S = stock.history(period='1d')['Close'].iloc[-1]
    q = float(stock.info.get('trailingAnnualDividendYield', 0.0))
    today = datetime.today()
    df["Expiry_dt"] = df["Expiry"]
    df['Years_to_Expiry'] = (df['Expiry_dt'] - today).dt.days / 365.25
    df['Forward'] = S * np.exp((r - q) * df['Years_to_Expiry'])
    df['LogMoneyness'] = np.log(df['Strike'] / df['Forward'])
    df['IV_bid'] = np.nan
    df['IV_ask'] = np.nan
    df['IV_mid'] = np.nan
    df['IV_spread'] = np.nan
    results = Parallel(n_jobs=-1, backend='threading')(delayed(compute_ivs)(row, S, r, q) for _, row in df.iterrows())
    df[['IV_bid', 'IV_ask', 'IV_mid', 'IV_spread']] = pd.DataFrame(results, index=df.index)
    return df, S, r, q

def compute_local_vol_from_iv_row(row, r, q, interp):
    y = row['LogMoneyness']
    T = row['Years_to_Expiry']
    if T <= 0:
        return None
    
    w = interp(np.array([[y, T]]))[0]
    if np.isnan(w):
        return None
    
    h_t = max(0.01 * T, 1e-4)  # Adaptive, with min to avoid zero
    h_y = max(0.01 * abs(y) if y != 0 else 0.01, 1e-4)  # Adaptive to moneyness scale
    
    # ∂w/∂T
    w_T_plus = interp(np.array([[y, T + h_t]]))[0]
    w_T_minus = interp(np.array([[y, max(T - h_t, 1e-6)]]))[0]
    if np.isnan(w_T_plus) or np.isnan(w_T_minus):
        return None
    dw_dT = (w_T_plus - w_T_minus) / (2 * h_t)
    
    # ∂w/∂y
    w_y_plus = interp(np.array([[y + h_y, T]]))[0]
    w_y_minus = interp(np.array([[y - h_y, T]]))[0]
    if np.isnan(w_y_plus) or np.isnan(w_y_minus):
        return None
    dw_dy = (w_y_plus - w_y_minus) / (2 * h_y)
    
    # ∂²w/∂y²
    d2w_dy2 = (w_y_plus - 2 * w + w_y_minus) / (h_y ** 2)
    
    if np.isnan(dw_dT) or np.isnan(dw_dy) or np.isnan(d2w_dy2):
        return None
    
    # Dupire formula for local vol from IV
    denom = 1 - (y / w) * dw_dy + 0.25 * (-0.25 - 1/w + (y**2 / w**2)) * (dw_dy ** 2) + 0.5 * d2w_dy2
    if denom <= 1e-10 or dw_dT <= 0:
        local_vol = 0.0
    else:
        local_vol_sq = dw_dT / denom
        local_vol = np.sqrt(max(local_vol_sq, 0)) if local_vol_sq > 0 else 0.0
        if local_vol > 2.0:  # Cap outliers
            local_vol = np.nan
    
    return {
        "Strike": row['Strike'],
        "Expiry": row['Expiry'],
        "Local Vol": local_vol
    }

def calculate_local_vol_from_iv(df, S, r, q):
    required_columns = ['Type', 'Strike', 'Expiry', 'IV_mid', 'Years_to_Expiry', 'Forward', 'LogMoneyness']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")
    
    calls = df[df['Type'] == 'Call'].copy()
    puts = df[df['Type'] == 'Put'].copy()
    
    call_local_df = pd.DataFrame()
    put_local_df = pd.DataFrame()
    
    def process_options(options_df, option_type):
        if options_df.empty:
            return pd.DataFrame()
        
        options_df = options_df[options_df['IV_mid'] > 0]
        options_df = options_df[options_df['Years_to_Expiry'] > 0]
        options_df = options_df.sort_values(['Years_to_Expiry', 'LogMoneyness'])
        
        # Compute total variance w = IV^2 * T
        options_df['TotalVariance'] = options_df['IV_mid']**2 * options_df['Years_to_Expiry']
        
        points = np.column_stack((options_df['LogMoneyness'], options_df['Years_to_Expiry']))
        values = options_df['TotalVariance'].values
        
        if len(options_df) < 3:
            return pd.DataFrame()
        
        # Use RBF for smoother interpolation with some regularization
        try:
            interp = RBFInterpolator(points, values, kernel='thin_plate_spline', smoothing=0.1)  # Increased smoothing
        except Exception as e:
            print(f"Warning: RBF fit failed for {option_type}: {e}. Using linear fallback.")
            interp = LinearNDInterpolator(points, values, fill_value=np.nan, rescale=True)
        
        local_data = Parallel(n_jobs=-1, backend='threading')(
            delayed(compute_local_vol_from_iv_row)(row, r, q, interp)
            for _, row in options_df.iterrows()
        )
        local_data = [d for d in local_data if d is not None]
        return pd.DataFrame(local_data) if local_data else pd.DataFrame()
    
    call_local_df = process_options(calls, 'Call')
    put_local_df = process_options(puts, 'Put')
    
    return call_local_df, put_local_df

def process_ticker(ticker, df, r):
    print(f"Processing calculations for {ticker}...")
    ticker_df = df[df['Ticker'] == ticker].copy()
    if ticker_df.empty:
        print(f"Warning: No data for ticker {ticker} in df")
        return None
    rvol100d = calculate_rvol_days(ticker, 100)
    print(f"\nRealised Volatility for {ticker}:")
    print(f"100-day: {rvol100d * 100:.2f}%" if rvol100d is not None else "100-day: N/A")
    ticker_df, S, r, q = calculate_iv_mid(ticker_df, ticker, r)
    ticker_df = calc_Ivol_Rvol(ticker_df, rvol100d)
    ticker_df, skew_df, slope_df = calculate_metrics(ticker_df, ticker, r)
    call_local_df, put_local_df = calculate_local_vol_from_iv(ticker_df, S, r, q)
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
    return ticker_df

def main():
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
        latest_clean = f'data/cleaned_{timestamp}.csv'
    else:
        clean_files = glob.glob('data/cleaned_*.csv')
        if not clean_files:
            print("No cleaned data files found")
            return
        latest_clean = max(clean_files, key=os.path.getctime)
        timestamp = os.path.basename(latest_clean).split('cleaned_')[1].split('.csv')[0]
    df = pd.read_csv(latest_clean, parse_dates=['Expiry'])
    tickers = df['Ticker'].unique()
    if len(tickers) == 0:
        print("No tickers found")
        return
    
    tnx_data = yf.download('^TNX', period='1d', auto_adjust=True)
    r = tnx_data['Close'].iloc[-1].item() / 100 if not tnx_data.empty else 0.05
    
    processed_dfs = []
    for ticker in tickers:
        processed = process_ticker(ticker, df, r)
        if processed is not None:
            processed_dfs.append(processed)
    
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
