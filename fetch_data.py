import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import json
from scipy.stats import kurtosis

def fetch_option_data_yfinance(ticker):
    print(f"Fetching option data for {ticker} from yfinance...")
    stock = yf.Ticker(ticker)
    option_data = []
    expiries = stock.options
    last_stock_price = stock.info.get('regularMarketPrice', stock.info.get('lastPrice', None))
    bid_stock = stock.info.get('bid', None)
    ask_stock = stock.info.get('ask', None)
    for expiry in expiries:
        chain = stock.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts
        calls['Type'] = 'Call'
        puts['Type'] = 'Put'
        calls['Expiry'] = expiry
        puts['Expiry'] = expiry
        calls['Ticker'] = ticker
        puts['Ticker'] = ticker
        calls['Last Stock Price'] = last_stock_price
        puts['Last Stock Price'] = last_stock_price
        calls['Bid Stock'] = bid_stock
        puts['Bid Stock'] = bid_stock
        calls['Ask Stock'] = ask_stock
        puts['Ask Stock'] = ask_stock
        calls['Moneyness'] = last_stock_price / calls['strike'] if last_stock_price else np.nan
        puts['Moneyness'] = puts['strike'] / last_stock_price if last_stock_price else np.nan
        option_data.append(calls)
        option_data.append(puts)
    if not option_data:
        return pd.DataFrame()
    yfinance_df = pd.concat(option_data, ignore_index=True)
    columns = ['Ticker', 'Contract Name', 'Type', 'Expiry', 'Strike', 'Bid', 'Ask', 'Volume', 'Open Interest', 'Implied Volatility', 'Last Stock Price', 'Bid Stock', 'Ask Stock', 'Moneyness']
    yfinance_df = yfinance_df.rename(columns={
        'contractSymbol': 'Contract Name',
        'strike': 'Strike',
        'bid': 'Bid',
        'ask': 'Ask',
        'volume': 'Volume',
        'openInterest': 'Open Interest',
        'impliedVolatility': 'Implied Volatility'
    })
    yfinance_df['Strike'] = (yfinance_df['Strike'] / 0.01) * 0.01
    yfinance_df = yfinance_df[columns]
    return yfinance_df

def fetch_historic_data(ticker, historic_dir):
    print(f"Fetching historic data for {ticker}...")
   
    # Persistent history folder
    persistent_history_dir = 'data/history'
    os.makedirs(persistent_history_dir, exist_ok=True)
    persistent_hist_file = f'{persistent_history_dir}/historic_{ticker}.csv'
   
    # Load existing persistent history if available
    hist = pd.DataFrame()
    start_date = None
    if os.path.exists(persistent_hist_file):
        try:
            hist = pd.read_csv(persistent_hist_file)
            hist['Ticker'] = ticker  # Always set Ticker to ensure it's present
            # Ensure 'Date' column exists and is datetime
            if 'Date' in hist.columns:
                hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
                # Drop rows with invalid dates
                hist = hist.dropna(subset=['Date'])
                # Check for duplicate 'Date' columns
                date_cols = [col for col in hist.columns if col.lower() == 'date']
                if len(date_cols) > 1:
                    print(f"Warning: Multiple 'Date' columns found for {ticker}, keeping the first.")
                    hist = hist.loc[:, ~hist.columns.duplicated()]
                if not hist.empty:
                    last_date = hist['Date'].max().date()
                    today = datetime.now().date()
                    next_date = (datetime.combine(last_date, datetime.min.time()) + timedelta(days=1)).date()
                    if next_date <= today:
                        start_date = (datetime.combine(next_date, datetime.min.time())).strftime('%Y-%m-%d')
                    else:
                        print(f"Warning: Computed start_date {next_date} is after today; fetching full history.")
                        start_date = None
                else:
                    start_date = None
            else:
                print(f"Warning: No 'Date' column in {persistent_hist_file}, treating as empty.")
                hist = pd.DataFrame()
                start_date = None
        except Exception as e:
            print(f"Error loading {persistent_hist_file}: {e}")
            hist = pd.DataFrame()
            start_date = None
    else:
        start_date = None
   
    # Define columns here so always available
    columns = [
        'Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100',
        'Realised_Vol_Close_180', 'Realised_Vol_Close_252',
        'Vol_of_Vol_100d', 'Vol_of_Vol_100d_Percentile',
        'Kurtosis_Close_100d', 'Kurtosis_Returns_100d'
    ]
   
    # Prepare hist with all columns if it exists
    if not hist.empty and 'Date' in hist.columns:
        hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
        hist = hist.dropna(subset=['Date'])
        hist['Date'] = hist['Date'].dt.tz_localize(None)
        hist['Ticker'] = ticker  # Ensure Ticker is set
        # Add missing columns with NaN
        for col in columns:
            if col not in hist.columns:
                hist[col] = np.nan
        hist = hist[columns]
        # Sort by Date
        hist = hist.sort_values('Date').reset_index(drop=True)
    else:
        hist = pd.DataFrame(columns=columns)
        hist['Ticker'] = ticker
   
    stock = yf.Ticker(ticker)
   
    # Fetch new data (incremental if start_date exists)
    new_hist = pd.DataFrame()
    try:
        if start_date:
            new_hist = stock.history(start=start_date)
        else:
            new_hist = stock.history(period='max')
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        new_hist = pd.DataFrame()
   
    if not new_hist.empty:
        new_hist = new_hist[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Strip timezone from index and reset to make Date a column
        new_hist.index = pd.to_datetime(new_hist.index).tz_localize(None)
        new_hist = new_hist.reset_index()
        new_hist = new_hist.rename(columns={'index': 'Date'})
       
        # Combine with existing history if hist has data
        if not hist.empty:
            # Concatenate and remove duplicates
            hist = pd.concat([hist, new_hist], ignore_index=True)
            hist = hist.drop_duplicates(subset=['Date'], keep='last')
            hist = hist.sort_values('Date').reset_index(drop=True)
        else:
            new_hist['Ticker'] = ticker
            hist = new_hist
       
        # Recompute derived columns
        hist['Log_Return_Close'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hist['Realised_Vol_Close_30'] = hist['Log_Return_Close'].rolling(window=30).std() * np.sqrt(252) * 100
        hist['Realised_Vol_Close_60'] = hist['Log_Return_Close'].rolling(window=60).std() * np.sqrt(252) * 100
        hist['Realised_Vol_Close_100'] = hist['Log_Return_Close'].rolling(window=100).std() * np.sqrt(252) * 100
        hist['Realised_Vol_Close_180'] = hist['Log_Return_Close'].rolling(window=180).std() * np.sqrt(252) * 100
        hist['Realised_Vol_Close_252'] = hist['Log_Return_Close'].rolling(window=252).std() * np.sqrt(252) * 100
       
        # Calculate Vol of Vol and Percentile
        vol_series = hist['Realised_Vol_Close_100'].copy()
        hist['Vol_of_Vol_100d'] = vol_series.rolling(window=100, min_periods=100).std().round(2)
        hist['Vol_of_Vol_100d_Percentile'] = vol_series.rolling(window=100, min_periods=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x.dropna()) >= 100 else np.nan, raw=False
        ).round(2)
       
        # Calculate Kurtosis
        hist['Kurtosis_Close_100d'] = hist['Close'].rolling(window=100, min_periods=100).apply(
            lambda x: kurtosis(x.dropna(), nan_policy='omit') if len(x.dropna()) >= 100 else np.nan, raw=False
        ).round(2)
        hist['Kurtosis_Returns_100d'] = hist['Log_Return_Close'].rolling(window=100, min_periods=100).apply(
            lambda x: kurtosis(x.dropna(), nan_policy='omit') if len(x.dropna()) >= 100 else np.nan, raw=False
        ).round(2)
       
        hist = hist[columns]
       
        # Format Date as string for saving
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
       
        # Save to persistent and timestamped directories
        hist.to_csv(persistent_hist_file, index=False)
        hist_filename = f'{historic_dir}/historic_{ticker}.csv'
        hist.to_csv(hist_filename, index=False)
        print(f"Historic data for {ticker} saved to {hist_filename} (rows: {len(hist)})")
    else:
        if hist.empty:
            print(f"No historic data available for {ticker}")
            return pd.DataFrame()
        # No new data, but use/save existing hist
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        hist_filename = f'{historic_dir}/historic_{ticker}.csv'
        hist.to_csv(hist_filename, index=False)
        print(f"Historic data for {ticker} saved to {hist_filename} (rows: {len(hist)})")
   
    return hist
def main():
    with open('tickers.txt', 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]
    print(f"Number of tickers: {len(tickers)}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_dir = f'data/{timestamp}'
    raw_yfinance_dir = f'{base_dir}/raw_yfinance'
    historic_dir = f'{base_dir}/historic'
    os.makedirs(raw_yfinance_dir, exist_ok=True)
    os.makedirs(historic_dir, exist_ok=True)
    for ticker in tickers:
        yfinance_df = fetch_option_data_yfinance(ticker)
        if not yfinance_df.empty:
            yfinance_filename = f'{raw_yfinance_dir}/raw_yfinance_{ticker}.csv'
            yfinance_df.to_csv(yfinance_filename, index=False)
            print(f"yfinance raw data for {ticker} saved to {yfinance_filename}")
        df_hist = fetch_historic_data(ticker, historic_dir) # Pass historic_dir
        time.sleep(1)
    # Update dates.json with the new timestamp
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

main()
