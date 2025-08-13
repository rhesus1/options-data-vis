import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os

def fetch_option_data(ticker, strikes, min_volume_percentile=25, min_oi_percentile=25):
    option_data = []
    stock = yf.Ticker(ticker)
    try:
        expirations = stock.options
   
        for expiry in expirations:
            opt = stock.option_chain(expiry)
            calls = opt.calls
            puts = opt.puts
       
            for _, row in calls.iterrows():
                volume = row['volume'] if pd.notna(row['volume']) else 0
                open_interest = row['openInterest'] if pd.notna(row['openInterest']) else 0
                bid = row['bid']
                ask = row['ask']
            
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
                    "Ticker": ticker,
                    "Last Trade Date": pd.to_datetime(row['lastTradeDate']) if pd.notna(row['lastTradeDate']) else ""
                })
       
            for _, row in puts.iterrows():
                volume = row['volume'] if pd.notna(row['volume']) else 0
                open_interest = row['openInterest'] if pd.notna(row['openInterest']) else 0
                bid = row['bid']
                ask = row['ask']
            
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
                    "Ticker": ticker,
                    "Last Trade Date": pd.to_datetime(row['lastTradeDate']) if pd.notna(row['lastTradeDate']) else ""
                })
       
            time.sleep(1)
       
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    full_df = pd.DataFrame(option_data)
    if not full_df.empty:
        if "Expiry" in full_df.columns:
            full_df["Expiry"] = full_df["Expiry"].apply(lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tz') else x)
        if "Last Trade Date" in full_df.columns:
            full_df["Last Trade Date"] = full_df["Last Trade Date"].apply(lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tz') else x)
    return full_df

def calculate_vix(ticker, option_df, stock_price, risk_free_rate=0.02):
    """
    Calculate a VIX-like index for the ticker using option data.
    Follows Cboe VIX methodology: interpolates near-term and next-term variances to 30 days.
    """
    if option_df.empty:
        return np.nan
    
    # Calculate days to expiration
    today = pd.Timestamp.now().floor('D')
    option_df['Days_to_Expiry'] = (option_df['Expiry'] - today).dt.total_seconds() / (60 * 60 * 24)
    
    # Select near-term (23-37 days) and next-term expirations
    expiries = option_df['Days_to_Expiry'].unique()
    expiries = expiries[(expiries >= 23) & (expiries <= 37)]
    if len(expiries) < 1:
        near_term = expiries[expiries >= 23].min() if len(expiries[expiries >= 23]) > 0 else np.nan
        next_term = expiries[expiries > near_term].min() if len(expiries[expiries > near_term]) > 0 else np.nan
    else:
        near_term = expiries.min()
        next_term = expiries[expiries > near_term].min() if len(expiries[expiries > near_term]) > 0 else np.nan
    
    if pd.isna(near_term) or pd.isna(next_term):
        print(f"Insufficient expiration data for {ticker} VIX calculation")
        return np.nan
    
    # Convert days to minutes for VIX formula
    T1 = near_term / 365  # Time to near-term expiry in years
    T2 = next_term / 365  # Time to next-term expiry in years
    N1 = near_term * 24 * 60  # Near-term in minutes
    N2 = next_term * 24 * 60  # Next-term in minutes
    N_30 = 30 * 24 * 60  # 30 days in minutes
    
    # Filter options for each term
    near_df = option_df[option_df['Days_to_Expiry'] == near_term]
    next_df = option_df[option_df['Days_to_Expiry'] == next_term]
    
    def compute_variance(df, T, R, S):
        if df.empty:
            return np.nan
        
        # Calculate forward price F = Strike + e^(R*T) * (Call - Put)
        strikes = df['Strike'].unique()
        min_diff = np.inf
        F = S  # Fallback to stock price if no suitable pair
        K0 = strikes[0]  # Fallback
        for K in strikes:
            call = df[(df['Type'] == 'Call') & (df['Strike'] == K)]
            put = df[(df['Type'] == 'Put') & (df['Strike'] == K)]
            if not call.empty and not put.empty:
                call_price = call['Mid Option'].iloc[0]
                put_price = put['Mid Option'].iloc[0]
                diff = abs(call_price - put_price)
                if diff < min_diff:
                    min_diff = diff
                    F = K + np.exp(R * T) * (call_price - put_price)
                    K0 = K
        
        # Select out-of-the-money options
        puts = df[(df['Type'] == 'Put') & (df['Strike'] < K0) & (df['Bid'] > 0)]
        calls = df[(df['Type'] == 'Call') & (df['Strike'] > K0) & (df['Bid'] > 0)]
        atm = df[((df['Type'] == 'Call') | (df['Type'] == 'Put')) & (df['Strike'] == K0) & (df['Bid'] > 0)]
        
        # Combine and sort by strike
        options = pd.concat([puts, calls, atm]).sort_values('Strike')
        if options.empty:
            return np.nan
        
        # Calculate Delta K
        strikes = options['Strike'].values
        delta_K = np.zeros_like(strikes)
        for i in range(len(strikes)):
            if i == 0:
                delta_K[i] = strikes[1] - strikes[0]
            elif i == len(strikes) - 1:
                delta_K[i] = strikes[-1] - strikes[-2]
            else:
                delta_K[i] = (strikes[i + 1] - strikes[i - 1]) / 2
        
        # Calculate variance contribution
        sum_term = 0
        for i, row in options.iterrows():
            K_i = row['Strike']
            Q_i = row['Mid Option']
            sum_term += (delta_K[i] / (K_i ** 2)) * np.exp(R * T) * Q_i
        
        variance = (2 / T) * sum_term - (1 / T) * ((F / K0 - 1) ** 2)
        return variance if variance > 0 else np.nan
    
    # Compute variances
    sigma1_sq = compute_variance(near_df, T1, risk_free_rate, stock_price)
    sigma2_sq = compute_variance(next_df, T2, risk_free_rate, stock_price)
    
    if pd.isna(sigma1_sq) or pd.isna(sigma2_sq):
        print(f"Insufficient option data for {ticker} VIX calculation")
        return np.nan
    
    # Interpolate to 30 days
    variance_30 = T1 * sigma1_sq * ((N2 - N_30) / (N2 - N1)) + T2 * sigma2_sq * ((N_30 - N1) / (N2 - N1))
    
    # Convert to VIX (annualized volatility in percentage)
    vix = 100 * np.sqrt(variance_30 * 365 / 30)
    return vix if not np.isnan(vix) else np.nan

def process_ticker_fetch(ticker):
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    try:
        S = stock.history(period='1d')['Close'].iloc[-1]
        info = stock.info
        bid = info.get('bid', None)
        ask = info.get('ask', None)
        mid = (bid + ask)/2 if bid is not None and ask is not None else 0
    except:
        print(f"Failed to fetch stock price for {ticker}")
        return pd.DataFrame()
    full_df = fetch_option_data(ticker, [])
    if full_df.empty:
        return pd.DataFrame()
    full_df['Last Stock Price'] = S
    full_df['Bid Stock'] = bid
    full_df['Ask Stock'] = ask
    full_df['Mid Option'] = (full_df['Bid'] + full_df['Ask'])/2
    full_df['Mid Stock'] = (full_df['Bid Stock'] + full_df['Ask Stock'])/2 if bid is not None and ask is not None else 0
    if mid > 0:
        full_df['Moneyness'] = np.round(mid / full_df['Strike'] / 0.01) * 0.01
    else:
        full_df['Moneyness'] = np.round(S / full_df['Strike'] / 0.01) * 0.01
    
    # Calculate VIX-like index for the ticker
    vix_value = calculate_vix(ticker, full_df, S)
    full_df['VIX'] = vix_value
    
    columns = ['Ticker', 'Contract Name', 'Type', 'Expiry', 'Strike', 'Moneyness', 'Bid', 'Ask', 'Volume', 'Open Interest', 'Bid Stock', 'Ask Stock', 'Last Stock Price', 'Implied Volatility', 'VIX']
    return full_df[columns]

def fetch_historic_data(ticker):
    print(f"Fetching historic data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period='max')
    if hist.empty:
        return pd.DataFrame()
    hist = hist[['Close']]
    hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist['Realised_Vol_30'] = hist['Log_Return'].rolling(window=30).std() * np.sqrt(252) * 100
    hist['Realised_Vol_60'] = hist['Log_Return'].rolling(window=60).std() * np.sqrt(252) * 100
    hist['Realised_Vol_90'] = hist['Log_Return'].rolling(window=90).std() * np.sqrt(252) * 100
    hist['Realised_Vol_180'] = hist['Log_Return'].rolling(window=180).std() * np.sqrt(252) * 100
    hist['Realised_Vol_360'] = hist['Log_Return'].rolling(window=360).std() * np.sqrt(252) * 100
    hist = hist.dropna()
    hist['Date'] = hist.index.strftime('%Y-%m-%d')
    hist['Ticker'] = ticker
    return hist[['Ticker', 'Date', 'Close', 'Realised_Vol_30', 'Realised_Vol_60', 'Realised_Vol_90', 'Realised_Vol_180', 'Realised_Vol_360']]

def main():
    with open('tickers.txt', 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]
 
    all_data = []
    all_hist = []
    for ticker in tickers:
        df = process_ticker_fetch(ticker)
        if not df.empty:
            all_data.append(df)
       
        df_hist = fetch_historic_data(ticker)
        if not df_hist.empty:
            all_hist.append(df_hist)
       
        time.sleep(1) # Avoid rate limiting
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs('data', exist_ok=True)
   
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        filename = f'data/raw_{timestamp}.csv'
        combined_df.to_csv(filename, index=False)
        print(f"Raw data saved to {filename}")
    else:
        print("No data to save")
   
    if all_hist:
        combined_hist_df = pd.concat(all_hist, ignore_index=True)
        hist_filename = f'data/historic_{timestamp}.csv'
        combined_hist_df.to_csv(hist_filename, index=False)
        print(f"Historic data saved to {hist_filename}")
    else:
        print("No historic data to save")
     
main()
