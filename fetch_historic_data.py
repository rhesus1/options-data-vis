# fetch_historic_data.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time

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
    
    all_hist = []
    for ticker in tickers:
        df = fetch_historic_data(ticker)
        if not df.empty:
            all_hist.append(df)
        time.sleep(1)  # Avoid rate limiting
    
    if all_hist:
        combined_df = pd.concat(all_hist, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs('data', exist_ok=True)
        filename = f'data/historic_{timestamp}.csv'
        combined_df.to_csv(filename, index=False)
        print(f"Historic data saved to {filename}")
    else:
        print("No historic data to save")

main()
