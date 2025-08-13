# fetch_data.py
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
import multiprocessing
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
 
    return full_df  # Only return full_df since filtering is in clean script

def process_ticker_fetch(ticker):
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    try:
        S = stock.history(period='1d')['Close'].iloc[-1]
        info = stock.info
        bid = info.get('bid', None)
        ask = info.get('ask', None)
        mid = (bid + ask)/2;
    except:
        print(f"Failed to fetch stock price for {ticker}")
        return pd.DataFrame()
    full_df = fetch_option_data(ticker, [])
    if full_df.empty:
        return pd.DataFrame()
    full_df['Last Stock Price'] = S
    full_df['Bid Stock'] = bid
    full_df['Ask Stock'] = ask

    full_df['Mid Option'] = (full_df['Bid'] + full_df['Ask'])/2;
    full_df['Mid Stock'] = (full_df['Bid Stock'] + full_df['Ask Stock'])/2;

    if mid > 0:
        full_df['Moneyness'] = np.round(mid / full_df['Strike'] / 0.05) * 0.05
    else:
        full_df['Moneyness'] = np.round(S / full_df['Strike'] / 0.05) * 0.05
    columns = ['Ticker', 'Contract Name', 'Type', 'Expiry', 'Strike', 'Moneyness', 'Bid', 'Ask', 'Volume', 'Open Interest', 'Last Stock Price', 'Implied Volatility']
    return full_df[columns]

def main():
    with open('tickers.txt', 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]
   
    num_processes = min(4, len(tickers))
    with multiprocessing.Pool(processes=num_processes) as pool:
        all_data = pool.map(process_ticker_fetch, tickers)
   
    all_data = [df for df in all_data if not df.empty]
   
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs('data', exist_ok=True)
        filename = f'data/raw_{timestamp}.csv'
        combined_df.to_csv(filename, index=False)
        print(f"Raw data saved to {filename}")
    else:
        print("No data to save")
       
main()
