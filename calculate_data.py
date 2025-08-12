# calculate_data.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import json
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import griddata
from scipy.optimize import minimize

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
        print(f"Warning: {option_type.capitalize()} {contract_name} price {price} below intrinsic {intrinsic * np.exp(-r * T):.2f}; returning 0.0001")
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
        low = objective(0.0001)
        high = objective(20.0)
        print(f"Warning: Failed to solve IV for {option_type.capitalize()} {contract_name}: low={low:.2f}, high={high:.2f}, price={price}, S={S}, K={K}, T={T:.4f}")
        return np.nan

def calculate_rvol(ticker, period):
    try:
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            print(f"Invalid period '{period}'. Valid periods: {valid_periods}")
            return None
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            print(f"No data retrieved for ticker '{ticker}' with period '{period}'")
            return None
        log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        if len(log_returns) < 2:
            print(f"Insufficient data points for '{ticker}' with period '{period}' to calculate volatility")
            return None
        realized_vol = np.std(log_returns, ddof=1) * np.sqrt(252)
        return realized_vol
    except Exception as e:
        print(f"Error calculating realized volatility for '{ticker}': {str(e)}")
        return None

def calculate_rvol_days(ticker, days):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < days + 1:
            print(f"Insufficient data for '{ticker}' over {days} days (need at least {days + 1} prices)")
            return None
        hist_last = hist.iloc[-(days + 1):]
        log_returns = np.log(hist_last["Close"] / hist_last["Close"].shift(1)).dropna()
        if len(log_returns) < 2:
            print(f"Insufficient returns for '{ticker}' over {days} days")
            return None
        realized_vol = np.std(log_returns, ddof=1) * np.sqrt(252)
        return realized_vol
    except Exception as e:
        print(f"Error calculating {days}-day realized volatility for '{ticker}': {str(e)}")
        return None

def calc_Ivol_Rvol(df, rvol5d, rvol1m, rvol3m, rvol6m, rvol1y, rvol2y, rvol90d):
    if df.empty:
        return df
    df["Ivol/Rvol5d Ratio"] = df["Implied Volatility"] / rvol5d
    df["Ivol/Rvol1m Ratio"] = df["Implied Volatility"] / rvol1m
    df["Ivol/Rvol3m Ratio"] = df["Implied Volatility"] / rvol3m
    df["Ivol/Rvol6m Ratio"] = df["Implied Volatility"] / rvol6m
    df["Ivol/Rvol1y Ratio"] = df["Implied Volatility"] / rvol1y
    df["Ivol/Rvol2y Ratio"] = df["Implied Volatility"] / rvol2y
    df["Ivol/Rvol90d Ratio"] = df["Implied Volatility"] / rvol90d
    return df

def calculate_metrics(df, ticker):
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame(), None, None, None
  
    skew_data = []
    for exp in df["Expiry"].unique():
        for strike in df["Strike"].unique():
            call_iv = df[(df["Type"] == "Call") & (df["Strike"] == strike) & (df["Expiry"] == exp)]["Implied Volatility"]
            put_iv = df[(df["Type"] == "Put") & (df["Strike"] == strike) & (df["Expiry"] == exp)]["Implied Volatility"]
            if not call_iv.empty and not put_iv.empty and call_iv.iloc[0] > 0:
                skew = put_iv.iloc[0] / call_iv.iloc[0]
                skew_data.append({"Expiry": exp, "Strike": strike, "Vol Skew": f"{skew*100:.2f}%"})
  
    skew_df = pd.DataFrame(skew_data)
  
    slope_data = []
    for strike in df["Strike"].unique():
        for opt_type in ["Call", "Put"]:
            subset = df[(df["Strike"] == strike) & (df["Type"] == opt_type)].sort_values("Expiry")
            if len(subset) > 1:
                iv_diff = subset["Implied Volatility"].diff()
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
  
    stock = yf.Ticker(ticker)
    S = stock.history(period='1d')['Close'].iloc[-1]
    tnx_data = yf.download('^TNX', period='1d')
    r = float(tnx_data['Close'].iloc[-1] / 100) if not tnx_data.empty else 0.05
    q = float(stock.info.get('trailingAnnualDividendYield', 0.0))
    today = datetime.today()
    df["Expiry_dt"] = df["Expiry"]
    df['Years_to_Expiry'] = (df['Expiry_dt'] - today).dt.days / 365.25
    invalid_rows = df[df['Years_to_Expiry'].isna()]
    if not invalid_rows.empty:
        print("Warning: NaN in Years_to_Expiry for the following contracts:")
        for idx, row in invalid_rows.iterrows():
            print(f"- {row['Contract Name']} (Expiry: {row['Expiry']})")
    df['IV_bid'] = np.nan
    df['IV_ask'] = np.nan
    df['IV_bid-ask'] = np.nan
    df['IV_spread'] = np.nan
    for idx, row in df.iterrows():
        if pd.isna(row['Years_to_Expiry']):
            continue
        T = max(row['Years_to_Expiry'], 0.0001)
        option_type = row['Type'].lower()
        contract_name = row['Contract Name']
        df.at[idx, 'IV_bid'] = implied_vol(row['Bid'], S, row['Strike'], T, r, q, option_type, contract_name) * 100
        df.at[idx, 'IV_ask'] = implied_vol(row['Ask'], S, row['Strike'], T, r, q, option_type, contract_name) * 100
        df.at[idx, 'IV_bid-ask'] = implied_vol(0.5*(row['Bid']+row['Ask']), S, row['Strike'], T, r, q, option_type, contract_name) * 100
        df.at[idx, 'IV_spread'] = df.at[idx, 'IV_ask'] - df.at[idx, 'IV_bid'] if not np.isnan(df.at[idx, 'IV_bid']) else np.nan
    return df, skew_df, slope_df, S, r, q

def heston_char_func(phi, S0, v0, kappa, theta, sigma_vol, rho, r, tau):
    i = 1j
    d = np.sqrt((rho * sigma_vol * i * phi - kappa)**2 + sigma_vol**2 * (i * phi + phi**2))
    g = (kappa - rho * sigma_vol * i * phi - d) / (kappa - rho * sigma_vol * i * phi + d)
    C = r * i * phi * tau + (kappa * theta / sigma_vol**2) * ((kappa - rho * sigma_vol * i * phi - d) * tau - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
    D = ((kappa - rho * sigma_vol * i * phi - d) / sigma_vol**2) * ((1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)))
    return np.exp(C + D * v0 + i * phi * np.log(S0))

def heston_price_call(S0, K, v0, kappa, theta, sigma_vol, rho, r, tau):
    i = 1j
    def integrand(phi):
        return np.real(np.exp(-i * phi * np.log(K)) / (i * phi) * heston_char_func(phi - i, S0, v0, kappa, theta, sigma_vol, rho, r, tau) / heston_char_func(-i, S0, v0, kappa, theta, sigma_vol, rho, r, tau))
    phi_vals = np.linspace(0.01, 100, 1000)
    integral = np.trapz(integrand(phi_vals), dx=phi_vals[1] - phi_vals[0])
    P1 = 0.5 + (1 / np.pi) * integral
    def integrand2(phi):
        return np.real(heston_char_func(phi, S0, v0, kappa, theta, sigma_vol, rho, r, tau) / (i * phi * np.exp(i * phi * np.log(K))))
    integral2 = np.trapz(integrand2(phi_vals), dx=phi_vals[1] - phi_vals[0])
    P2 = 0.5 + (1 / np.pi) * integral2
    return S0 * P1 - K * np.exp(-r * tau) * P2

def calibrate_heston(df, S, r, q):
    def objective(params):
        v0, kappa, theta, sigma_vol, rho = params
        error = 0.0
        for idx, row in df.iterrows():
            T = row['Years_to_Expiry']
            K = row['Strike']
            market_price = 0.5 * (row['Bid'] + row['Ask'])
            model_price = heston_price_call(S, K, v0, kappa, theta, sigma_vol, rho, r, T)
            error += (model_price - market_price)**2
        return error
    initial = [0.04, 2.0, 0.04, 0.3, -0.7]
    bounds = [(0.01, 0.2), (0.1, 5.0), (0.01, 0.2), (0.1, 1.0), (-0.99, 0.99)]
    result = minimize(objective, initial, bounds=bounds, method='L-BFGS-B')
    if result.success:
        return result.x
    else:
        print("Heston calibration failed:", result.message)
        return None

def calculate_heston_iv(df, S, r, q, heston_params):
    if heston_params is None:
        df['Heston IV'] = np.nan
        return df
    v0, kappa, theta, sigma_vol, rho = heston_params
    df['Heston IV'] = np.nan
    for idx, row in df.iterrows():
        T = row['Years_to_Expiry']
        K = row['Strike']
        heston_p = heston_price_call(S, K, v0, kappa, theta, sigma_vol, rho, r, T)
        df.at[idx, 'Heston IV'] = implied_vol(heston_p, S, K, T, r, q, row['Type'].lower()) * 100
    return df

def calculate_local_vol(full_df, S, r, q):
    calls = full_df[full_df['Type'] == 'Call'].copy()
    if calls.empty:
        return pd.DataFrame()
    calls['mid_price'] = (calls['Bid'] + calls['Ask']) / 2
    calls['T'] = (calls['Expiry'] - datetime.today()).dt.days / 365.25
    calls = calls[calls['mid_price'] > 0]
    calls = calls[calls['T'] > 0]
    calls = calls.sort_values(['T', 'Strike'])
    if len(calls) < 3:
        print("Warning: Insufficient call data points for local vol surface interpolation.")
        return pd.DataFrame()
    points = np.column_stack((calls['Strike'], calls['T']))
    values = calls['mid_price'].values
    
    def call_price_interp(k, t):
        if t <= 0:
            return np.nan
        interp_val = griddata(points, values, (k, t), method='linear', fill_value=np.nan, rescale=False)
        return interp_val
    
    def finite_diff_1st(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)
    
    def finite_diff_2nd(f, x, h):
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
    
    local_vol_data = []
    for idx, row in calls.iterrows():
        k = row['Strike']
        t = row['T']
        c = call_price_interp(k, t)
        if np.isnan(c):
            continue
        h_t = max(1e-3, t * 1e-3)
        dC_dT = finite_diff_1st(lambda tt: call_price_interp(k, tt), t, h_t)
        h_k = max(0.1, k * 0.001)
        dC_dK = finite_diff_1st(lambda kk: call_price_interp(kk, t), k, h_k)
        d2C_dK2 = finite_diff_2nd(lambda kk: call_price_interp(kk, t), k, h_k)
        if np.isnan(dC_dT) or np.isnan(dC_dK) or np.isnan(d2C_dK2) or d2C_dK2 <= 0:
            print(f"Warning: Invalid derivatives for K={k}, T={t}; skipping.")
            local_vol = np.nan
        else:
            numer = dC_dT + (r - q) * k * dC_dK + q * c
            denom = 0.5 * k**2 * d2C_dK2
            if denom == 0 or numer <= 0:
                print(f"Warning: Non-positive local vol sq for K={k}, T={t} (numer={numer:.2f}, denom={denom:.2f}); setting NaN.")
                local_vol = np.nan
            else:
                local_vol_sq = numer / denom
                local_vol = np.sqrt(local_vol_sq) * 100
        local_vol_data.append({
            "Strike": k,
            "Expiry": row['Expiry'],
            "Local Vol": local_vol
        })
    
    return pd.DataFrame(local_vol_data)

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
    
    processed_dfs = []
    for ticker in df['Ticker'].unique():
        print(f"Processing calculations for {ticker}...")
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_full = full_df[full_df['Ticker'] == ticker].copy()
        if ticker_df.empty:
            continue
        rvol5d = calculate_rvol(ticker, "5d")
        rvol1m = calculate_rvol(ticker, "1mo")
        rvol3m = calculate_rvol(ticker, "3mo")
        rvol6m = calculate_rvol(ticker, "6mo")
        rvol1y = calculate_rvol(ticker, "1y")
        rvol2y = calculate_rvol(ticker, "2y")
        rvol90d = calculate_rvol_days(ticker, 90)
        
        print(f"\nRealized Volatility for {ticker}:")
        print(f"5-day: {rvol5d * 100:.2f}%" if rvol5d is not None else "5-day: N/A")
        print(f"1-month: {rvol1m * 100:.2f}%" if rvol1m is not None else "1-month: N/A")
        print(f"3-month: {rvol3m * 100:.2f}%" if rvol3m is not None else "3-month: N/A")
        print(f"6-month: {rvol6m * 100:.2f}%" if rvol6m is not None else "6-month: N/A")
        print(f"1-year: {rvol1y * 100:.2f}%" if rvol1y is not None else "1-year: N/A")
        print(f"2-year: {rvol2y * 100:.2f}%" if rvol2y is not None else "2-year: N/A")
        print(f"90-day: {rvol90d * 100:.2f}%" if rvol90d is not None else "90-day: N/A")
        
        ticker_df = calc_Ivol_Rvol(ticker_df, rvol5d, rvol1m, rvol3m, rvol6m, rvol1y, rvol2y, rvol90d)
        ticker_df, skew_df, slope_df, S, r, q = calculate_metrics(ticker_df, ticker)
        #heston_params = calibrate_heston(ticker_df, S, r, q)
        #ticker_df = calculate_heston_iv(ticker_df, S, r, q, heston_params)
        local_df = calculate_local_vol(ticker_full, S, r, q)
        if not local_df.empty:
            ticker_df = ticker_df.merge(local_df, on=['Strike', 'Expiry'], how='left')
        else:
            ticker_df['Local Vol'] = np.nan
        ticker_df['Realized Vol 90d'] = rvol90d * 100 if rvol90d is not None else np.nan
        ticker_df['Implied Volatility'] = ticker_df['Implied Volatility'] * 100
        ticker_df['Moneyness'] = ticker_df['Moneyness'] * 100
        
        # Add skew gradient and convexity calculations
        ticker_df['Call Skew Gradient'] = np.nan
        ticker_df['Put Skew Gradient'] = np.nan
        ticker_df['Upside Convexity'] = np.nan
        ticker_df['Downside Convexity'] = np.nan
        for exp in ticker_df['Expiry'].unique():
            for opt_type in ["Call", "Put"]:
                subset = ticker_df[(ticker_df['Expiry'] == exp) & (ticker_df['Type'] == opt_type)].sort_values('Moneyness')
                if len(subset) < 2:
                    continue
                M = subset['Moneyness'].values
                IV = subset['Implied Volatility'].values
                grad = np.gradient(IV, M)
                if len(subset) < 3:
                    conv = np.full(len(grad), np.nan)
                else:
                    conv = np.gradient(grad, M)
                if opt_type == "Call":
                    ticker_df.loc[subset.index, 'Call Skew Gradient'] = grad
                    ticker_df.loc[subset.index, 'Upside Convexity'] = conv
                else:
                    ticker_df.loc[subset.index, 'Put Skew Gradient'] = grad
                    ticker_df.loc[subset.index, 'Downside Convexity'] = conv
        
        processed_dfs.append(ticker_df)
    
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
