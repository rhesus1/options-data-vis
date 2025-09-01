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

def smooth_iv_per_expiry(options_df):
    if options_df.empty:
        return options_df
    smoothed_iv = pd.Series(np.nan, index=options_df.index, dtype=float)
    for exp, group in options_df.groupby('Expiry'):
        if len(group) < 3:
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
        if len(cleaned_group) < 3:
            smoothed_iv.loc[group.index] = group['IV_mid']
            continue
        if cleaned_group['LogMoneyness'].duplicated().any():
            agg_group = cleaned_group.groupby('LogMoneyness')['IV_mid'].mean().reset_index()
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
            print(f"Warning: LOWESS failed for expiry {exp}: {e}. Using IV_mid directly.")
            smoothed_iv.loc[group.index] = group['IV_mid']
    options_df['Smoothed_IV'] = smoothed_iv
    return options_df

def compute_local_vol_from_iv_row(row, r, q, interp):
    y = row['LogMoneyness']
    T = row['Years_to_Expiry']
    if T <= 0:
        return None
    if pd.isna(row['Smoothed_IV']):
        print(f"Warning: Smoothed_IV is NaN for Strike {row['Strike']}, Expiry {row['Expiry']}. Using IV_mid.")
        iv = np.clip(row['IV_mid'], 0.05, 5.0)
        w = iv ** 2 * T
        dw_dy = 0
        d2w_dy2 = 0
        dw_dT = iv ** 2
    else:
        w = interp(np.array([[y, T]]))[0]
        if np.isnan(w):
            return None
        h_t = max(0.01 * T, 1e-4)
        h_y = max(0.01 * abs(y) if y != 0 else 0.01, 1e-4)
        w_T_plus = interp(np.array([[y, T + h_t]]))[0]
        w_T_minus = interp(np.array([[y, max(T - h_t, 1e-6)]]))[0]
        if np.isnan(w_T_plus) or np.isnan(w_T_minus):
            return None
        dw_dT = (w_T_plus - w_T_minus) / (2 * h_t)
        w_y_plus = interp(np.array([[y + h_y, T]]))[0]
        w_y_minus = interp(np.array([[y - h_y, T]]))[0]
        if np.isnan(w_y_plus) or np.isnan(w_y_minus):
            return None
        dw_dy = (w_y_plus - w_y_minus) / (2 * h_y)
        d2w_dy2 = (w_y_plus - 2 * w + w_y_minus) / (h_y ** 2)
        if np.isnan(dw_dT) or np.isnan(dw_dy) or np.isnan(d2w_dy2):
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
        return pd.DataFrame(), None, options_df
    options_df = options_df[options_df['IV_mid'] > 0]
    options_df = options_df[options_df['Years_to_Expiry'] > 0]
    options_df = smooth_iv_per_expiry(options_df)
    smoothed_df = options_df.copy()
    smoothed_df = smoothed_df.sort_values(['Years_to_Expiry', 'LogMoneyness'])
    smoothed_df['TotalVariance'] = smoothed_df['Smoothed_IV']**2 * smoothed_df['Years_to_Expiry']
    points = np.column_stack((smoothed_df['LogMoneyness'], smoothed_df['Years_to_Expiry']))
    values = smoothed_df['TotalVariance'].values
    if len(smoothed_df) < 3:
        return pd.DataFrame(), None, smoothed_df
    try:
        interp = RBFInterpolator(points, values, kernel='thin_plate_spline', smoothing=0.1)
    except Exception as e:
        print(f"Warning: RBF fit failed for {option_type}: {e}. Using linear fallback.")
        interp = LinearNDInterpolator(points, values, fill_value=np.nan, rescale=True)
    local_data = Parallel(n_jobs=-1, backend='threading')(
        delayed(compute_local_vol_from_iv_row)(row, r, q, interp)
        for _, row in smoothed_df.iterrows()
    )
    local_data = [d for d in local_data if d is not None]
    local_df = pd.DataFrame(local_data) if local_data else pd.DataFrame()
    return local_df, interp, smoothed_df

def calculate_local_vol_from_iv(df, S, r, q):
    required_columns = ['Type', 'Strike', 'Expiry', 'IV_mid', 'Years_to_Expiry', 'Forward', 'LogMoneyness']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")
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

def calculate_skew_metrics(df, call_interp, put_interp, S, r, q):
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
    skew_metrics_df['ATM_12m_3m_Ratio'] = atm_ratio
    skew_metrics_df['ATM_IV_3m'] = atm_iv_3m
    skew_metrics_df['ATM_IV_12m'] = atm_iv_12m
    
    return skew_metrics_df, slope_metrics_df

def process_ticker(ticker, df, full_df, r):
    print(f"Processing calculations for {ticker}...")
    ticker_df = df[df['Ticker'] == ticker].copy()
    ticker_full = full_df[full_df['Ticker'] == ticker].copy()
    if ticker_df.empty:
        print(f"Warning: No data for ticker {ticker} in df")
        return None, None, None
    rvol100d = calculate_rvol_days(ticker, 100)
    print(f"\nRealised Volatility for {ticker}:")
    print(f"100-day: {rvol100d * 100:.2f}%" if rvol100d is not None else "100-day: N/A")
    ticker_df, S, r, q = calculate_iv_mid(ticker_df, ticker, r)
    ticker_df = calc_Ivol_Rvol(ticker_df, rvol100d)
    ticker_df, skew_df, slope_df = calculate_metrics(ticker_df, ticker, r)
    call_local_df, put_local_df, call_interp, put_interp, smoothed_df = calculate_local_vol_from_iv(ticker_df, S, r, q)
    ticker_df = smoothed_df
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

def process_data(clean_file, raw_file, timestamp, prefix=""):
    df = pd.read_csv(clean_file, parse_dates=['Expiry'])
    if not os.path.exists(raw_file):
        print(f"Corresponding raw file {raw_file} not found")
        return None, None, None
    full_df = pd.read_csv(raw_file, parse_dates=['Expiry'])
    tickers = df['Ticker'].unique()
    if len(tickers) == 0:
        print(f"No tickers found in {clean_file}")
        return None, None, None
    tnx_data = yf.download('^TNX', period='1d', auto_adjust=True)
    r = tnx_data['Close'].iloc[-1].item() / 100 if not tnx_data.empty else 0.05
    processed_dfs = []
    skew_metrics_dfs = []
    slope_metrics_dfs = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = pool.starmap(process_ticker, [(ticker, df, full_df, r) for ticker in tickers])
    for pdf, sdf, slope_df in results:
        if pdf is not None:
            processed_dfs.append(pdf)
        if sdf is not None:
            skew_metrics_dfs.append(sdf)
        if slope_df is not None:
            slope_metrics_dfs.append(slope_df)
    combined_processed = None
    combined_skew_metrics = None
    combined_slope_metrics = None
    if processed_dfs:
        combined_processed = pd.concat(processed_dfs, ignore_index=True)
        processed_json_filename = f'data/processed_{prefix}{timestamp}.json'
        combined_processed.to_json(processed_json_filename, orient='records', date_format='iso')
        print(f"Processed {prefix}data saved to {processed_json_filename}")
        processed_csv_filename = f'data/processed_{prefix}{timestamp}.csv'
        combined_processed.to_csv(processed_csv_filename, index=False)
        print(f"Processed {prefix}data saved to {processed_csv_filename}")
    if skew_metrics_dfs:
        combined_skew_metrics = pd.concat(skew_metrics_dfs, ignore_index=True)
        skew_metrics_filename = f'data/skew_metrics_{prefix}{timestamp}.csv'
        combined_skew_metrics.to_csv(skew_metrics_filename, index=False)
        print(f"Skew metrics {prefix}saved to {skew_metrics_filename}")
    if slope_metrics_dfs:
        combined_slope_metrics = pd.concat(slope_metrics_dfs, ignore_index=True)
        slope_metrics_filename = f'data/slope_metrics_{prefix}{timestamp}.csv'
        combined_slope_metrics.to_csv(slope_metrics_filename, index=False)
        print(f"Slope metrics {prefix}saved to {slope_metrics_filename}")
    else:
        print(f"No skew or slope metrics to save for {prefix}")
    return combined_processed, combined_skew_metrics, combined_slope_metrics

def main():
    # Use specific glob patterns to match Nasdaq and yfinance files
    clean_files = glob.glob('data/cleaned_[0-9]*.csv')
    clean_yfinance_files = glob.glob('data/cleaned_yfinance_*.csv')
    if not clean_files and not clean_yfinance_files:
        print("No cleaned data files found")
        return
    
    # Initialize dates.json
    dates_file = 'data/dates.json'
    if os.path.exists(dates_file):
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    else:
        dates = []
    
    # Collect unique timestamps
    timestamps = set()
    
    # Process the most recent Nasdaq file
    if clean_files:
        latest_clean = max(clean_files, key=os.path.getctime)
        timestamp = os.path.basename(latest_clean).split('cleaned_')[1].split('.csv')[0]
        timestamps.add(timestamp)
        raw_file = f'data/raw_{timestamp}.csv'
        print(f"Processing Nasdaq data: {latest_clean}")
        result = process_data(latest_clean, raw_file, timestamp, prefix="")
        if result[0] is None and result[1] is None and result[2] is None:
            print(f"Failed to process Nasdaq data for {timestamp}")
    
    # Process the most recent yfinance file
    if clean_yfinance_files:
        latest_yfinance_clean = max(clean_yfinance_files, key=os.path.getctime)
        timestamp = os.path.basename(latest_yfinance_clean).split('cleaned_yfinance_')[1].split('.csv')[0]
        timestamps.add(timestamp)
        raw_yfinance_file = f'data/raw_yfinance_{timestamp}.csv'
        print(f"Processing yfinance data: {latest_yfinance_clean}")
        result = process_data(latest_yfinance_clean, raw_yfinance_file, timestamp, prefix="yfinance_")
        if result[0] is None and result[1] is None and result[2] is None:
            print(f"Failed to process yfinance data for {timestamp}")
    
    # Update dates.json with all unique timestamps
    for timestamp in timestamps:
        if timestamp not in dates:
            dates.append(timestamp)
    if timestamps:
        dates.sort(reverse=True)
        with open(dates_file, 'w') as f:
            json.dump(dates, f)
        print(f"Updated dates list in {dates_file} with timestamps: {timestamps}")
        
main()
