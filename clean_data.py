# clean_data.py (updated)
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import json
import os

def clean_data(db_conn, fetch_date, ticker=None):
    query = f"""
    SELECT * FROM raw_options 
    WHERE Fetch_Date = '{fetch_date}'
    """
    if ticker:
        query += f" AND Ticker = '{ticker}'"
    
    df = pd.read_sql_query(query, db_conn)
    if df.empty:
        return pd.DataFrame()
    
    df['Expiry'] = pd.to_datetime(df['Expiry'])
    
    # Remove duplicates based on Contract_Name
    duplicates = df.duplicated(subset=['Contract_Name']).sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['Contract_Name'])
        print(f"Removed {duplicates} duplicates")
    
    cleaned_groups = []
    for t, group in df.groupby('Ticker'):
        # 1. Remove options with empty Bid or empty Ask
        group = group.dropna(subset=['Bid', 'Ask', 'Strike'])
        
        # 2. Filter moneyness: keep >= 0.3 and <= 3.5
        group = group[(group['Moneyness'] >= 0.3) & (group['Moneyness'] <= 3.5)]
        
        # 3. Remove options where both open interest and volume are 0
        group = group[~((group['Open_Interest'] == 0) & (group['Volume'] == 0))]
        
        # 4. Remove the bottom 25% based on open interest (sorted ascending)
        if not group.empty:
            n = len(group)
            threshold_index = int(n * 0.25)
            group = group.sort_values(by='Open_Interest').iloc[threshold_index:]
        
        # 5. Calculate ln(1 + open interest) / SM%, where SM% = 100 * (ask - bid) / mid, mid = (bid + ask)/2
        group = group[(group['Bid'] + group['Ask']) > 0]
        mid = (group['Bid'] + group['Ask']) / 2
        spread = group['Ask'] - group['Bid']
        sm_percent = 100 * (spread / mid)
        group['weight'] = np.log(1 + group['Open_Interest']) / sm_percent
        group = group[np.isfinite(group['weight'])]
        if not group.empty:
            n = len(group)
            threshold_index = int(n * 0.25)
            group = group.sort_values(by='weight').iloc[threshold_index:]
        group = group.drop(columns=['weight'])
        cleaned_groups.append(group)
    
    if cleaned_groups:
        cleaned_df = pd.concat(cleaned_groups)
        return cleaned_df
    else:
        return pd.DataFrame()

def main():
    # Check last cleaned date (store in a simple JSON or DB; here using JSON for simplicity)
    last_cleaned_file = 'data/last_cleaned.json'
    today = datetime.now().strftime('%Y-%m-%d')
    fetch_date = today  # Assume cleaning today's fetch
    
    last_cleaned = None
    if os.path.exists(last_cleaned_file):
        with open(last_cleaned_file, 'r') as f:
            last_cleaned_data = json.load(f)
            last_cleaned = last_cleaned_data.get('date', None)
    
    # If last cleaned is yesterday or earlier, clean all of today
    needs_full_clean = last_cleaned != (datetime.now().date() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Connect to DBs
    raw_conn = sqlite3.connect('data/options_raw.db')
    cleaned_conn = sqlite3.connect('data/cleaned_options.db')
    
    # Create cleaned table if not exists
    cursor = cleaned_conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cleaned_options (
            Ticker TEXT,
            Contract_Name TEXT,
            Type TEXT,
            Expiry TEXT,
            Strike REAL,
            Bid REAL,
            Ask REAL,
            Volume INTEGER,
            Open_Interest INTEGER,
            Implied_Volatility REAL,
            Last_Stock_Price REAL,
            Bid_Stock REAL,
            Ask_Stock REAL,
            Moneyness REAL,
            Fetch_Date TEXT,
            PRIMARY KEY (Ticker, Contract_Name, Fetch_Date)
        )
    ''')
    
    # Get unique tickers from today's fetch
    today_query = f"SELECT DISTINCT Ticker FROM raw_options WHERE Fetch_Date = '{fetch_date}'"
    today_tickers = pd.read_sql_query(today_query, raw_conn)['Ticker'].tolist()
    
    if needs_full_clean:
        print(f"Full clean needed for {fetch_date} (tickers: {len(today_tickers)})")
        for ticker in today_tickers:
            cleaned_df = clean_data(raw_conn, fetch_date, ticker)
            if not cleaned_df.empty:
                cleaned_df.to_sql('cleaned_options', cleaned_conn, if_exists='append', index=False)
                print(f"Cleaned {len(cleaned_df)} rows for {ticker}")
    else:
        print(f"Incremental clean: only new data for {fetch_date}")
        # Since we append only new fetches, and clean per day, it's always "new"
        # But if partial, clean all today's as above
        for ticker in today_tickers:
            cleaned_df = clean_data(raw_conn, fetch_date, ticker)
            if not cleaned_df.empty:
                cleaned_df.to_sql('cleaned_options', cleaned_conn, if_exists='append', index=False)
                print(f"Cleaned {len(cleaned_df)} rows for {ticker}")
    
    # Update last_cleaned
    with open(last_cleaned_file, 'w') as f:
        json.dump({'date': today}, f)
    
    raw_conn.close()
    cleaned_conn.close()
    print("Cleaning complete. Cleaned data in data/cleaned_options.db")

if __name__ == "__main__":
    main()
