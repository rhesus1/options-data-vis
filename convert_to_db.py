# convert_to_db.py
import pandas as pd
import sqlite3
import json
import os
import glob
from datetime import datetime

def create_historic_db():
    # Create/connect to historic.db
    conn = sqlite3.connect('data/historic.db')
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historic (
            Ticker TEXT,
            Date TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume INTEGER,
            Realised_Vol_Close_30 REAL,
            Realised_Vol_Close_60 REAL,
            Realised_Vol_Close_100 REAL,
            Realised_Vol_Close_180 REAL,
            Realised_Vol_Close_252 REAL,
            Vol_of_Vol_100d REAL,
            Vol_of_Vol_100d_Percentile REAL,
            Kurtosis_Close_100d REAL,
            Kurtosis_Returns_100d REAL,
            PRIMARY KEY (Ticker, Date)
        )
    ''')
    
    # Persistent history directory
    persistent_history_dir = 'data/history'
    if os.path.exists(persistent_history_dir):
        hist_files = glob.glob(f'{persistent_history_dir}/historic_*.csv')
        for hist_file in hist_files:
            ticker = os.path.basename(hist_file).split('historic_')[1].split('.csv')[0]
            print(f"Converting historic data for {ticker} from {hist_file}")
            try:
                df = pd.read_csv(hist_file)
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                df['Ticker'] = ticker
                # Ensure all columns exist
                required_cols = [
                    'Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100',
                    'Realised_Vol_Close_180', 'Realised_Vol_Close_252',
                    'Vol_of_Vol_100d', 'Vol_of_Vol_100d_Percentile',
                    'Kurtosis_Close_100d', 'Kurtosis_Returns_100d'
                ]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = None
                df = df[required_cols]
                df.to_sql('historic', conn, if_exists='append', index=False)
                print(f"Inserted {len(df)} rows for {ticker}")
            except Exception as e:
                print(f"Error converting {hist_file}: {e}")
    
    conn.commit()
    conn.close()
    print("Historic data conversion complete. Database: data/historic.db")

def create_options_raw_db():
    # Create/connect to options_raw.db
    conn = sqlite3.connect('data/options_raw.db')
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_options (
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
    
    # Load dates.json
    dates_file = 'data/dates.json'
    if not os.path.exists(dates_file):
        print("No dates.json found. Skipping options conversion.")
        conn.close()
        return
    
    with open(dates_file, 'r') as f:
        dates = json.load(f)
    
    for timestamp in dates:
        base_dir = f'data/{timestamp}'
        raw_yfinance_dir = f'{base_dir}/raw_yfinance'
        if os.path.exists(raw_yfinance_dir):
            raw_files = glob.glob(f'{raw_yfinance_dir}/raw_yfinance_*.csv')
            fetch_date = datetime.strptime(timestamp, "%Y%m%d_%H%M").strftime('%Y-%m-%d')
            for raw_file in raw_files:
                ticker = os.path.basename(raw_file).split('raw_yfinance_')[1].split('.csv')[0]
                print(f"Converting raw options for {ticker} from {raw_file} (fetch_date: {fetch_date})")
                try:
                    df = pd.read_csv(raw_file, parse_dates=['Expiry'])
                    df['Fetch_Date'] = fetch_date
                    df = df.rename(columns={'Contract Name': 'Contract_Name'})
                    # Ensure all columns exist (use None for missing)
                    required_cols = [
                        'Ticker', 'Contract_Name', 'Type', 'Expiry', 'Strike', 'Bid', 'Ask',
                        'Volume', 'Open_Interest', 'Implied_Volatility', 'Last_Stock_Price',
                        'Bid_Stock', 'Ask_Stock', 'Moneyness', 'Fetch_Date'
                    ]
                    for col in required_cols:
                        if col not in df.columns:
                            df[col] = None
                    df = df[required_cols]
                    df.to_sql('raw_options', conn, if_exists='append', index=False)
                    print(f"Inserted {len(df)} rows for {ticker} on {fetch_date}")
                except Exception as e:
                    print(f"Error converting {raw_file}: {e}")
    
    conn.commit()
    conn.close()
    print("Raw options conversion complete. Database: data/options_raw.db")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    create_historic_db()
    create_options_raw_db()
    print("Conversion to databases complete. Run this once to migrate existing data.")
