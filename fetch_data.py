# fetch_data.py (updated)
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import json
import sqlite3

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

def fetch_historic_data_raw(ticker, conn):
    print(f"Fetching raw historic data for {ticker}...")
    
    # Get last date from DB
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(Date) FROM historic WHERE Ticker = ?", (ticker,))
    last_date_result = cursor.fetchone()[0]
    start_date = None
    if last_date_result:
        last_date = datetime.strptime(last_date_result, '%Y-%m-%d').date()
        next_date = (datetime.combine(last_date, datetime.min.time()) + timedelta(days=1)).date()
        today = datetime.now().date()
        if next_date <= today:
            start_date = next_date.strftime('%Y-%m-%d')
    
    stock = yf.Ticker(ticker)
    new_hist = pd.DataFrame()
    try:
        if start_date:
            new_hist = stock.history(start=start_date)
        else:
            new_hist = stock.history(period='max')
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return
    
    if not new_hist.empty:
        new_hist = new_hist[['Open', 'High', 'Low', 'Close', 'Volume']]
        new_hist.index = pd.to_datetime(new_hist.index).tz_localize(None)
        new_hist = new_hist.reset_index()
        new_hist = new_hist.rename(columns={'index': 'Date'})
        new_hist['Date'] = new_hist['Date'].dt.strftime('%Y-%m-%d')
        new_hist['Ticker'] = ticker
        # Insert new rows only (ignore existing dates)
        new_hist.to_sql('historic', conn, if_exists='append', index=False)
        # But since we have PRIMARY KEY, duplicates are ignored
        print(f"Inserted/appended {len(new_hist)} new raw historic rows for {ticker}")
    else:
        print(f"No new historic data for {ticker}")

def main():
    with open('tickers.txt', 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]
    print(f"Number of tickers: {len(tickers)}")
    
    # Ensure data dir
    os.makedirs('data', exist_ok=True)
    
    # Connect to DBs (create if not exist)
    historic_conn = sqlite3.connect('data/historic.db')
    options_conn = sqlite3.connect('data/options_raw.db')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fetch_date = datetime.now().strftime('%Y-%m-%d')
    
    for ticker in tickers:
        # Fetch raw options and insert with fetch_date
        yfinance_df = fetch_option_data_yfinance(ticker)
        if not yfinance_df.empty:
            yfinance_df['Fetch_Date'] = fetch_date
            yfinance_df = yfinance_df.rename(columns={'Contract Name': 'Contract_Name', 'Open Interest': 'Open_Interest'})
            yfinance_df.to_sql('raw_options', options_conn, if_exists='append', index=False)
            print(f"Raw options for {ticker} appended to DB (rows: {len(yfinance_df)})")
        
        # Fetch raw historic (only missing)
        fetch_historic_data_raw(ticker, historic_conn)
        
        time.sleep(1)
    
    # Close connections
    historic_conn.close()
    options_conn.close()
    
    # Update dates.json
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
    
    print("Fetch complete. Raw data appended to databases.")

if __name__ == "__main__":
    main()
