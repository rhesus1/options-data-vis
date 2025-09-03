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
warnings.filterwarnings("ignore", category=FutureWarning)

def black_scholes_call(S, K, T, r, q, sigma):
    if not all([S > 0, K > 0, T > 0, sigma > 0]):
        return max(S - K, 0)
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    except (ValueError, ZeroDivisionError):
        return max(S - K, 0)

def black_scholes_put(S, K, T, r, q, sigma):
    if not all([S > 0, K > 0, T > 0, sigma > 0]):
        return max(K - S, 0)
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    except (ValueError, ZeroDivisionError):
        return max(K - S, 0)

def black_scholes_delta(S, K, T, r, q, sigma, option_type):
    if not all([S > 0, K > 0, T > 0, sigma > 0]):
        return 0.0
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    except (ValueError, ZeroDivisionError):
        return 0.0

def implied_vol(price, S, K, T, r, q, option_type, contract_name=""):
    if not all([price > 0, S > 0, K > 0, T > 0]):
        return np.nan
    intrinsic = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
    if price < intrinsic * np.exp(-r * T):
        return np.nan
    def objective(sigma):
        if sigma <= 0:
            return np.inf
        if option_type.lower() == 'call':
            return black_scholes_call(S, K, T, r, q, sigma) - price
        else:
            return black_scholes_put(S, K, T, r, q, sigma) - price
    try:
        iv = brentq(objective, 0.0001, 50.0)
        return np.clip(iv, 0.05, 5.0)
    except ValueError:
        return np.nan

def calc_Ivol_Rvol(df, rvol100d):
    if df.empty:
        return df
    df["Ivol/Rvol100d Ratio"] = df["IV_mid"] / rvol100d
    return df

def compute_ivs(row, S, r, q):
    if pd.isna(row['Years_to_Expiry']) or row['Years_to_Expiry'] <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    T = max(row['Years_to_Expiry'], 0.0001)
    option_type = row['Type'].lower()
    contract_name = row['Contract Name']
    iv_bid = implied_vol(row['Bid'], S, row['Strike'], T, r, q, option_type, contract_name)
    iv_ask = implied_vol(row['Ask'], S, row['Strike'], T, r, q, option_type, contract_name)
    iv_mid = implied_vol(0.5*(row['Bid']+row['Ask']), S, row['Strike'], T, r, q, option_type, contract_name)
    iv_spread = iv_ask - iv_bid if not np.isnan(iv_bid) and not np.isnan(iv_ask) else np.nan
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

def calculate_iv_mid(df, ticker, r, timestamp):
    if df.empty:
        return df, None, None, None
    try:
        timestamp_dt = datetime.strptime(timestamp, '%Y%m%d_%H%M')
    except ValueError:
        print(f"Warning: Invalid timestamp format {timestamp}. Using default risk-free rate and no date reference.")
        return df, None, None, None
    S = (df['Bid Stock'].iloc[0] + df['Ask Stock'].iloc[0]) / 2
    if S <= 0:
        print(f"Warning: Invalid stock price {S} for {ticker}")
        return df, None, None, None
    q = 0.0
    df["Expiry_dt"] = pd.to_datetime(df["Expiry"])
    df['Years_to_Expiry'] = (df['Expiry_dt'] - timestamp_dt).dt.days / 365.25
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
        print("Warning: Input DataFrame is empty in smooth_iv_per_expiry")
        options_df['Smoothed_IV'] = np.nan
        return options_df
    required_columns = ['Expiry', 'LogMoneyness', 'IV_mid']
    if not all(col in options_df.columns for col in required_columns):
        print(f"Warning: Missing required columns {set(required_columns) - set(options_df.columns)} in smooth_iv_per_expiry")
        options_df['Smoothed_IV'] = np.nan
        return options_df
    smoothed_iv = pd.Series(np.nan, index=options_df.index, dtype=float)
    for exp, group in options_df.groupby('Expiry'):
        if len(group) < 3:
            print(f"Warning: Insufficient data points ({len(group)}) for expiry {exp}. Using IV_mid.")
            smoothed_iv.loc[group.index] = group['IV_mid']
            continue
        if group['IV_mid'].isna().all() or group['LogMoneyness'].isna().all():
            print(f"Warning: All IV_mid or LogMoneyness are NaN for expiry {exp}. Using IV_mid.")
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
            print(f"Warning: Insufficient valid data points ({len(cleaned_group)}) after cleaning for expiry {exp}. Using IV_mid.")
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
        print(f"Warning: No {option_type} options data provided")
        options_df['Smoothed_IV'] = np.nan
        options_df['TotalVariance'] = np.nan
        return pd.DataFrame(), None, options_df
    options_df = options_df[options_df['Years_to_Expiry'] > 0]
    options_df = options_df[options_df['IV_mid'].notna() & (options_df['IV_mid'] >= 0)]
    if options_df.empty:
        print(f"Warning: No valid {option_type} options after filtering")
        options_df['Smoothed_IV'] = np.nan
        options_df['TotalVariance'] = np.nan
        return pd.DataFrame(), None, options_df
    options_df = smooth_iv_per_expiry(options_df)
    if 'Smoothed_IV' not in options_df.columns:
        print(f"Warning: Smoothed_IV not created for {option_type} options")
        options_df['Smoothed_IV'] = options_df['IV_mid']
    options_df['TotalVariance'] = options_df['Smoothed_IV']**2 * options_df['Years_to_Expiry']
    options_df['TotalVariance'] = options_df['TotalVariance'].fillna(np.nan)
    smoothed_df = options_df.copy()
    smoothed_df = smoothed_df.sort_values(['Years_to_Expiry', 'LogMoneyness'])
    points = np.column_stack((smoothed_df['LogMoneyness'], smoothed_df['Years_to_Expiry']))
    values = smoothed_df['TotalVariance'].values
    if len(smoothed_df) < 3:
        print(f"Warning: Insufficient data points ({len(smoothed_df)}) for {option_type} interpolation")
        return pd.DataFrame(), None, smoothed_df
    try:
        interp = RBFInterpolator(points, values, kernel='thin_plate_spline', smoothing=0.001)
    except Exception as e:
        print(f"Warning: RBFInterpolator failed for {option_type}: {e}. Falling back to LinearNDInterpolator.")
        try:
            interp = LinearNDInterpolator(points, values, fill_value=np.nan)
        except Exception as e:
            print(f"Warning: LinearNDInterpolator failed for {option_type}: {e}. Falling back to CloughTocher2DInterpolator.")
            try:
                interp = CloughTocher2DInterpolator(points, values, fill_value=np.nan)
            except Exception as e:
                print(f"Error: All interpolators failed for {option_type}: {e}")
                return pd.DataFrame(), None, smoothed_df
    results = Parallel(n_jobs=-1)(delayed(compute_local_vol_from_iv_row)(row, r, q, interp) for _, row in smoothed_df.iterrows())
    local_vol_df = pd.DataFrame([res for res in results if res is not None])
    return local_vol_df, interp, smoothed_df

def calculate_local_vol_from_iv(df, S, r, q):
    calls = df[df['Type'] == 'Call'].copy()
    puts = df[df['Type'] == 'Put'].copy()
    call_local_df, call_interp, calls_smoothed = process_options(calls, 'Call', r, q)
    put_local_df, put_interp, puts_smoothed = process_options(puts, 'Put', r, q)
    smoothed_df = pd.concat([calls_smoothed, puts_smoothed]).sort_index()
    return call_local_df, put_local_df, call_interp, put_interp, smoothed_df

def find_strike_for_delta(S, T, r, q, sigma, target_delta, option_type):
    if not all([S > 0, T > 0, sigma > 0]):
        return np.nan
    def delta_diff(K):
        if K <= 0:
            return np.inf
        delta = black_scholes_delta(S, K, T, r, q, sigma, option_type)
        return delta - target_delta if option_type.lower() == 'call' else delta - (-target_delta)
    try:
        K = brentq(delta_diff, S * 0.5, S * 2.0)
        return K
    except ValueError:
        return np.nan

# New helper function to find put strike with matching mid-price
def find_put_strike_for_price(call_price, S, T, r, q, put_df, exp, tolerance=0.01):
    def price_diff(K):
        # Use average put IV from put_df as a guess, or default to 0.25
        put_row = put_df[(put_df['Expiry'] == exp) & (put_df['Strike'] == put_df['Strike'].iloc[0])]
        sigma_guess = put_row['IV_mid'].iloc[0] if not put_row.empty and not pd.isna(put_row['IV_mid'].iloc[0]) else 0.25
        put_price = black_scholes_put(S, K, T, r, q, sigma_guess)
        return put_price - call_price

    try:
        put_strike = brentq(price_diff, S * 0.1, S * 2.0, xtol=0.01)
        sigma_guess = 0.25
        put_price = black_scholes_put(S, put_strike, T, r, q, sigma_guess)
        if abs(put_price - call_price) <= tolerance:
            return put_strike
        return np.nan
    except ValueError:
        return np.nan

# Modified calculate_skew_metrics function
def calculate_skew_metrics(df, call_interp, put_interp, S, r, q):
    def get_iv(interp, y, T):
        if interp is None or T <= 0:
            return np.nan
        try:
            w = interp(np.array([[y, T]]))[0]
            if np.isnan(w) or w <= 0:
                return np.nan
            return np.sqrt(w / T)
        except Exception:
            return np.nan
    
    moneyness_levels = [0.6, 0.8, 1.2, 1.4]  # 60%, 80%, 120%, 140%
    target_strikes = [S * m for m in moneyness_levels]  # Call strikes
    skew_data = []
    target_deltas = [0.25, 0.75]
    
    for exp in sorted(df['Expiry'].unique()):
        T = df[df['Expiry'] == exp]['Years_to_Expiry'].iloc[0] if not df[df['Expiry'] == exp].empty else np.nan
        if np.isnan(T) or T <= 0:
            continue
        atm_iv = get_iv(call_interp, 0.0, T)
        if np.isnan(atm_iv):
            continue
        
        # Delta-based skews
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
        
        # Price-matched skews for specified moneyness levels
        put_df = df[(df['Type'] == 'Put') & (df['Expiry'] == exp)]
        skew_moneyness = {}
        for m, call_strike in zip(moneyness_levels, target_strikes):
            call_row = df[(df["Type"] == "Call") & (df["Strike"] == call_strike) & (df["Expiry"] == exp)]
            if call_row.empty:
                # Find closest strike if exact match not found
                call_row = df[(df["Type"] == "Call") & (df["Expiry"] == exp)]
                if not call_row.empty:
                    call_row = call_row.iloc[(call_row['Strike'] - call_strike).abs().argsort()[:1]]
                else:
                    skew_moneyness[f'Skew_{int(m*100)}_Moneyness'] = np.nan
                    continue
            call_mid = (call_row["Bid"].iloc[0] + call_row["Ask"].iloc[0]) / 2
            call_iv = call_row["IV_mid"].iloc[0]
            if pd.isna(call_mid) or pd.isna(call_iv) or call_mid <= 0:
                skew_moneyness[f'Skew_{int(m*100)}_Moneyness'] = np.nan
                continue
            
            # Find put with matching mid-price
            put_strike = find_put_strike_for_price(call_mid, S, T, r, q, put_df, exp)
            if np.isnan(put_strike):
                skew_moneyness[f'Skew_{int(m*100)}_Moneyness'] = np.nan
                continue
            # Get put IV for the matched price
            put_iv = implied_vol(call_mid, S, put_strike, T, r, q, 'put')
            if pd.isna(put_iv):
                skew_moneyness[f'Skew_{int(m*100)}_Moneyness'] = np.nan
                continue
            iv_skew_price_matched = put_iv / call_iv if call_iv > 0 else np.nan
            skew_moneyness[f'Skew_{int(m*100)}_Moneyness'] = iv_skew_price_matched
        
        # ATM IV ratio for 3m and 12m
        atm_iv_3m = get_iv(call_interp, 0.0, 0.25)
        atm_iv_12m = get_iv(call_interp, 0.0, 1.0)
        atm_ratio = atm_iv_12m / atm_iv_3m if not np.isnan(atm_iv_3m) and not np.isnan(atm_iv_12m) and atm_iv_3m > 0 else np.nan
        
        skew_data.append({
            'Expiry': exp,
            'Skew_25_delta': skew_25,
            'Skew_75_delta': skew_75,
            'Skew_call_25_75': skew_call_25_75,
            'Skew_put_25_75': skew_put_25_75,
            'Skew_60_Moneyness': skew_moneyness.get('Skew_60_Moneyness', np.nan),
            'Skew_80_Moneyness': skew_moneyness.get('Skew_80_Moneyness', np.nan),
            'Skew_120_Moneyness': skew_moneyness.get('Skew_120_Moneyness', np.nan),
            'Skew_140_Moneyness': skew_moneyness.get('Skew_140_Moneyness', np.nan),
            'ATM_12m_3m_Ratio': atm_ratio
        })
    
    skew_metrics_df = pd.DataFrame(skew_data)
    return skew_metrics_df, pd.DataFrame()  # Return empty slope_metrics_df as it's not needed for skew file

# Modified process_ticker function
def process_ticker(ticker, df, full_df, r, timestamp):
    print(f"Processing calculations for {ticker}...")
    ticker_df = df.copy()
    ticker_full = full_df.copy()
    if ticker_df.empty:
        print(f"Warning: No data for ticker {ticker} in df")
        return None, None, None
    
    # Load historic data
    historic_file = f'data/{timestamp}/historic/historic_{ticker}.csv'
    rvol100d = np.nan
    if os.path.exists(historic_file):
        try:
            historic_df = pd.read_csv(historic_file, parse_dates=['Date'])
            if not historic_df.empty and 'Realised_Vol_Close_100' in historic_df.columns:
                rvol100d = historic_df['Realised_Vol_Close_100'].iloc[-1] / 100
            else:
                print(f"Warning: No valid Realised_Vol_Close_100 data for {ticker} in {historic_file}")
        except Exception as e:
            print(f"Error reading historic file for {ticker}: {e}")
    else:
        print(f"No historic file found for {ticker} in data/{timestamp}/historic")
    
    ticker_df, S, r, q = calculate_iv_mid(ticker_df, ticker, r, timestamp)
    if ticker_df.empty or S is None:
        print(f"Warning: Failed to calculate IV_mid for {ticker}")
        return None, None, None
    
    ticker_df = calc_Ivol_Rvol(ticker_df, rvol100d)
    ticker_df, skew_df, slope_df = calculate_metrics(ticker_df, ticker, r)
    call_local_df, put_local_df, call_interp, put_interp, smoothed_df = calculate_local_vol_from_iv(ticker_df, S, r, q)
    
    # Use smoothed_df as ticker_df
    ticker_df = smoothed_df.copy()
    if 'TotalVariance' not in ticker_df.columns:
        ticker_df['TotalVariance'] = ticker_df['Smoothed_IV']**2 * ticker_df['Years_to_Expiry']
        ticker_df['TotalVariance'] = ticker_df['TotalVariance'].fillna(np.nan)
    
    skew_metrics_df, slope_metrics_df = calculate_skew_metrics(ticker_df, call_interp, put_interp, S, r, q)
    skew_metrics_df['Ticker'] = ticker
    
    # Ensure skew_metrics_df has only the required columns
    required_columns = [
        'Expiry', 'Skew_25_delta', 'Skew_75_delta', 'Skew_call_25_75', 'Skew_put_25_75',
        'Skew_60_Moneyness', 'Skew_80_Moneyness', 'Skew_120_Moneyness', 'Skew_140_Moneyness',
        'ATM_12m_3m_Ratio', 'Ticker'
    ]
    for col in required_columns:
        if col not in skew_metrics_df.columns:
            skew_metrics_df[col] = np.nan
    skew_metrics_df = skew_metrics_df[required_columns]
    
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
    ticker_df['Realised Vol 100d'] = rvol100d if not np.isnan(rvol100d) else np.nan
    
    desired_columns = [
        "Ticker", "Contract Name", "Type", "Expiry", "Strike", "Moneyness", "Bid", "Ask", "Volume",
        "Open Interest", "Bid Stock", "Ask Stock", "Last Stock Price", "Implied Volatility",
        "Expiry_dt", "Years_to_Expiry", "Forward", "LogMoneyness", "IV_bid", "IV_ask", "IV_mid",
        "IV_spread", "Delta", "Ivol/Rvol100d Ratio", "Smoothed_IV", "TotalVariance",
        "Call Local Vol", "Put Local Vol", "Realised Vol 100d"
    ]
    available_columns = [col for col in desired_columns if col in ticker_df.columns]
    ticker_df = ticker_df[available_columns]
    
    return ticker_df, skew_metrics_df, slope_df

def main():
    data_dirs = [d for d in glob.glob('data/*/') if os.path.isdir(d)]
    if not data_dirs:
        print("No data directories found")
        return
    latest_dir = max(data_dirs, key=os.path.getctime)
    timestamp = os.path.basename(latest_dir.rstrip('/'))
    clean_dir = f'data/{timestamp}/cleaned_yfinance'
    if not os.path.exists(clean_dir):
        print(f"No cleaned_yfinance directory for {timestamp}")
        return
    raw_dir = f'data/{timestamp}/raw_yfinance'
    if not os.path.exists(raw_dir):
        print(f"No raw_yfinance directory for {timestamp}")
        return
    if not os.path.exists('tickers.txt'):
        print("tickers.txt not found")
        return
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    processed_dir = f'data/{timestamp}/processed_yfinance'
    skew_dir = f'data/{timestamp}/skew_metrics_yfinance'
    slope_dir = f'data/{timestamp}/slope_metrics_yfinance'
    historic_dir = f'data/{timestamp}/historic'
    ranking_dir = f'data/{timestamp}/ranking'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(skew_dir, exist_ok=True)
    os.makedirs(slope_dir, exist_ok=True)
    os.makedirs(historic_dir, exist_ok=True)
    os.makedirs(ranking_dir, exist_ok=True)
    r = 0.05
    for ticker in tickers:
        clean_file = os.path.join(clean_dir, f'cleaned_yfinance_{ticker}.csv')
        raw_file = os.path.join(raw_dir, f'raw_yfinance_{ticker}.csv')
        if not os.path.exists(clean_file):
            print(f"No cleaned file for {ticker} in {clean_dir}")
            continue
        if not os.path.exists(raw_file):
            print(f"No raw file for {ticker} in {raw_dir}")
            continue
        df_ticker = pd.read_csv(clean_file, parse_dates=['Expiry'])
        full_df_ticker = pd.read_csv(raw_file, parse_dates=['Expiry'])
        pdf, sdf, slope_df = process_ticker(ticker, df_ticker, full_df_ticker, r, timestamp)
        if pdf is not None and not pdf.empty:
            pdf.to_csv(os.path.join(processed_dir, f'processed_yfinance_{ticker}.csv'), index=False)
            pdf.to_json(os.path.join(processed_dir, f'processed_yfinance_{ticker}.json'), orient='records', date_format='iso')
            print(f"Processed data for {ticker} saved to {processed_dir}")
        if sdf is not None and not sdf.empty:
            sdf.to_csv(os.path.join(skew_dir, f'skew_metrics_yfinance_{ticker}.csv'), index=False)
            print(f"Skew metrics for {ticker} saved to {skew_dir}")
        if slope_df is not None and not slope_df.empty:
            slope_df.to_csv(os.path.join(slope_dir, f'slope_metrics_yfinance_{ticker}.csv'), index=False)
            print(f"Slope metrics for {ticker} saved to {slope_dir}")
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

main()
