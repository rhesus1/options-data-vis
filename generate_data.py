import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from scipy.optimize import brentq
from scipy.stats import norm
import json

def fetch_option_data(ticker, strikes, min_volume_percentile=25, min_oi_percentile=25):
    option_data = []
    stock = yf.Ticker(ticker)
   
    try:
        # Get all expiration dates
        expirations = stock.options
       
        for expiry in expirations:
            # Fetch option chain for each expiration
            opt = stock.option_chain(expiry)
            calls = opt.calls
            puts = opt.puts
           
            # Process calls
            for _, row in calls.iterrows():
                #if row['strike'] not in strikes:
                #    continue
                
                volume = row['volume'] if pd.notna(row['volume']) else 0
                open_interest = row['openInterest'] if pd.notna(row['openInterest']) else 0
                bid = row['bid']
                ask = row['ask']
                
                # Store data temporarily without filtering
                option_data.append({
                    "Type": "Call",
                    "Strike": row['strike'],
                    "Expiry": pd.to_datetime(expiry),
                    "Last Price": row['lastPrice'],
                    "Bid": bid,
                    "Ask": ask,
                    "Change": row['change'],
                    "% Change": row['percentChange'] if pd.notna(row['percentChange']) else 0,
                    "Volume": volume,
                    "Open Interest": open_interest,
                    "Implied Volatility": row['impliedVolatility'] if pd.notna(row['impliedVolatility']) else 0,
                    "Contract Name": row['contractSymbol'],
                    "Last Trade Date": pd.to_datetime(row['lastTradeDate']) if pd.notna(row['lastTradeDate']) else ""
                })
           
            # Process puts
            for _, row in puts.iterrows():
                #if row['strike'] not in strikes:
                #    continue
                
                volume = row['volume'] if pd.notna(row['volume']) else 0
                open_interest = row['openInterest'] if pd.notna(row['openInterest']) else 0
                bid = row['bid']
                ask = row['ask']
                
                # Store data temporarily without filtering
                option_data.append({
                    "Type": "Put",
                    "Strike": row['strike'],
                    "Expiry": pd.to_datetime(expiry),
                    "Last Price": row['lastPrice'],
                    "Bid": bid,
                    "Ask": ask,
                    "Change": row['change'],
                    "% Change": row['percentChange'] if pd.notna(row['percentChange']) else 0,
                    "Volume": volume,
                    "Open Interest": open_interest,
                    "Implied Volatility": row['impliedVolatility'] if pd.notna(row['impliedVolatility']) else 0,
                    "Contract Name": row['contractSymbol'],
                    "Last Trade Date": pd.to_datetime(row['lastTradeDate']) if pd.notna(row['lastTradeDate']) else ""
                })
           
            time.sleep(1) # Rate limiting to avoid overwhelming the API
           
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
   
    # Convert to DataFrame
    df = pd.DataFrame(option_data)
   
    if not df.empty:
        # Timezone handling
        if "Expiry" in df.columns:
            df["Expiry"] = df["Expiry"].apply(lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tz') else x)
        if "Last Trade Date" in df.columns:
            df["Last Trade Date"] = df["Last Trade Date"].apply(lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tz') else x)
        
        # Calculate percentile thresholds for volume and open interest
        volume_threshold = df['Volume'].quantile(min_volume_percentile / 100)
        oi_threshold = df['Open Interest'].quantile(min_oi_percentile / 100)
        
        # Filter out low volume, low open interest, or invalid bid-ask spreads
        df = df[
            (df['Volume'] >= volume_threshold) &
            (df['Open Interest'] >= oi_threshold) &
            #(df['Ask'] - df['Bid'] >= 0.1) &
            (df['Bid'] >= 0) &
            (df['Ask'] >= 0) &
            (df['Bid'] <= df['Ask'])
        ]
   
    return df
    
def black_scholes_call(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)  # Intrinsic value at expiration or zero vol
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
def black_scholes_put(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)  # Intrinsic value at expiration or zero vol
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def implied_vol(price, S, K, T, r, q, option_type, contract_name=""):
    if price <= 0 or T <= 0:
        return 0.0  # No IV for invalid price or time
    intrinsic = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
    if price < intrinsic * np.exp(-r * T):  # Price below discounted intrinsic value
        print(f"Warning: {option_type.capitalize()} {contract_name} price {price} below intrinsic {intrinsic * np.exp(-r * T):.2f}; returning 0.0001")
        return 0.0001
    def objective(sigma):
        if option_type.lower() == 'call':
            return black_scholes_call(S, K, T, r, q, sigma) - price
        else:
            return black_scholes_put(S, K, T, r, q, sigma) - price
    try:
        iv = brentq(objective, 0.0001, 50.0)  # Increased upper bound to 1000%
        return iv
    except ValueError as e:
        low = objective(0.0001)
        high = objective(20.0)
        print(f"Warning: Failed to solve IV for {option_type.capitalize()} {contract_name}: low={low:.2f}, high={high:.2f}, price={price}, S={S}, K={K}, T={T:.4f}")
        return np.nan

def calculate_rvol(ticker, period):
    try:
        # Validate period
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            print(f"Invalid period '{period}'. Valid periods: {valid_periods}")
            return None

        # Fetch historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        # Check if data is empty
        if hist.empty:
            print(f"No data retrieved for ticker '{ticker}' with period '{period}'")
            return None
        
        # Calculate daily returns
        daily_returns = hist["Close"].pct_change().dropna()
        
        # Check if enough data points exist
        if len(daily_returns) < 2:
            print(f"Insufficient data points for '{ticker}' with period '{period}' to calculate volatility")
            return None
        
        # Calculate annualized realized volatility
        # Use ddof=1 for sample standard deviation (default in np.std)
        realized_vol = np.std(daily_returns, ddof=1) * np.sqrt(252)
        
        return realized_vol
    
    except Exception as e:
        print(f"Error calculating realized volatility for '{ticker}': {str(e)}")
        return None

def calc_Ivol_Rvol(df, rvol5d, rvol1m, rvol3m, rvol6m, rvol1y, rvol2y):
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()
    # Calculate Ivol/Rvol Ratio
    df["Ivol/Rvol5d Ratio"] = df["Implied Volatility"] / rvol5d
    df["Ivol/Rvol1m Ratio"] = df["Implied Volatility"] / rvol1m
    df["Ivol/Rvol3m Ratio"] = df["Implied Volatility"] / rvol3m
    df["Ivol/Rvol6m Ratio"] = df["Implied Volatility"] / rvol6m
    df["Ivol/Rvol1y Ratio"] = df["Implied Volatility"] / rvol1y
    df["Ivol/Rvol2y Ratio"] = df["Implied Volatility"] / rvol2y
    return df

def calculate_metrics(df):
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()
    
    # Calculate Vol Skew (Put IV / Call IV)
    skew_data = []
    for exp in df["Expiry"].unique():
        for strike in df["Strike"].unique():
            call_iv = df[(df["Type"] == "Call") & (df["Strike"] == strike) & (df["Expiry"] == exp)]["Implied Volatility"]
            put_iv = df[(df["Type"] == "Put") & (df["Strike"] == strike) & (df["Expiry"] == exp)]["Implied Volatility"]
            if not call_iv.empty and not put_iv.empty and call_iv.iloc[0] > 0:
                skew = put_iv.iloc[0] / call_iv.iloc[0]
                skew_data.append({"Expiry": exp, "Strike": strike, "Vol Skew": f"{skew*100:.2f}%"})
    
    skew_df = pd.DataFrame(skew_data)
    
    # Calculate IV Slope (change in IV per year)
    slope_data = []
    for strike in df["Strike"].unique():
        for opt_type in ["Call", "Put"]:
            subset = df[(df["Strike"] == strike) & (df["Type"] == opt_type)].sort_values("Expiry")
            if len(subset) > 1:
                iv_diff = subset["Implied Volatility"].diff()
                subset["Expiry_dt"] = pd.to_datetime(subset["Expiry"], format="%m/%d/%Y")
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
# Calculate Implied Volatility from Bid-Ask Spread
    stock = yf.Ticker(ticker)
    S = stock.history(period='1d')['Close'].iloc[-1]  # Current stock price
    tnx_data = yf.download('^TNX', period='1d')  # 10-year Treasury yield
    r = float(tnx_data['Close'].iloc[-1] / 100) if not tnx_data.empty else 0.05  # Ensure scalar
    q = float(stock.info.get('trailingAnnualDividendYield', 0.0))  # Ensure scalar
    today = datetime.today()
    df["Expiry_dt"] = df["Expiry"]  # Use existing Timestamp
    df['Years_to_Expiry'] = (df['Expiry_dt'] - today).dt.days / 365.25
    # Check for NaN in Years_to_Expiry
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
            continue  # Skip rows with invalid T
        T = max(row['Years_to_Expiry'], 0.0001)  # Use Years_to_Expiry, avoid zero
        option_type = row['Type'].lower()
        contract_name = row['Contract Name']
        df.at[idx, 'IV_bid'] = implied_vol(row['Bid'], S, row['Strike'], T, r, q, option_type, contract_name) * 100
        df.at[idx, 'IV_ask'] = implied_vol(row['Ask'], S, row['Strike'], T, r, q, option_type, contract_name) * 100
        df.at[idx, 'IV_bid-ask'] = implied_vol(0.5*(row['Bid']+row['Ask']), S, row['Strike'], T, r, q, option_type, contract_name) * 100
        df.at[idx, 'IV_spread'] = df.at[idx, 'IV_ask'] - df.at[idx, 'IV_bid'] if not np.isnan(df.at[idx, 'IV_bid']) else np.nan

    return df, skew_df, slope_df

def calc_moneyness(df, ticker):
    stock = yf.Ticker(ticker)
    S = stock.history(period='1d')['Close'].iloc[-1]  # Current stock price
    df["Moneyness"] = np.round(S / (df['Strike']) / 0.1) * 0.1
    return df

def main(ticker, strikes):
    # Fetch data
    df = fetch_option_data(ticker, strikes)

    # Calculate Realized Volatility
    rvol5d = calculate_rvol(ticker, "5d")
    rvol1m = calculate_rvol(ticker, "1mo")
    rvol3m = calculate_rvol(ticker, "3mo")
    rvol6m = calculate_rvol(ticker, "6mo")
    rvol1y = calculate_rvol(ticker, "1y")
    rvol2y = calculate_rvol(ticker, "2y")

    df = calc_moneyness(df, ticker)

    # Calc Ivol / Rvol
    df = calc_Ivol_Rvol(df, rvol5d, rvol1m, rvol3m, rvol6m, rvol1y, rvol2y)
    
    # Calculate metrics
    df, skew_df, slope_df = calculate_metrics(df)
    
    df['Implied Volatility'] = df['Implied Volatility'] * 100  # Convert to % for plotting
    df['IV_bid-ask'] = df['IV_bid-ask']  # Already in % from calculation
    df['Moneyness'] = df['Moneyness'] * 100  # Convert to % for plotting
    df.to_json('data.json', orient='records', date_format='iso')  # Save as list of dicts, dates as ISO strings

    # Save to CSV
    print(f"\nRealized Volatility (5-day): {rvol5d:.4f}")
    print(f"\nRealized Volatility (1-month): {rvol1m:.4f}")
    print(f"\nRealized Volatility (3-month): {rvol3m:.4f}")
    print(f"\nRealized Volatility (6-month): {rvol6m:.4f}")
    print(f"\nRealized Volatility (1-year): {rvol1y:.4f}")
    print(f"\nRealized Volatility (2-year): {rvol2y:.4f}")

strikes = np.arange(0, 2000.1, 0.5).tolist()#[40, 45, 50, 55, 60, 65, 70, 75]
ticker = "COIN"
main(ticker, strikes)
