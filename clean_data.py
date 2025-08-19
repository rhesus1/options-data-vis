import pandas as pd
import numpy as np
import glob
import os

def clean_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Expiry'])
    
    # Remove duplicates based on Contract Name
    duplicates = df.duplicated(subset=['Contract Name']).sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['Contract Name'])
        print(f"Removed {duplicates} duplicates from {file_path}")

    # Filter for non-negative values
    df = df[
        (df['Volume'] >= 0) &
        (df['Open Interest'] >= 0) &
        (df['Bid'] >= 0) &
        (df['Ask'] >= 0)
    ]

    # Apply quantile-based filtering for Volume and Open Interest
    volume_threshold = df['Volume'].quantile(0.1)
    oi_threshold = df['Open Interest'].quantile(0.25)

    df = df[
        (df['Volume'] >= volume_threshold) &
        (df['Open Interest'] >= oi_threshold)
    ]

    return df

def main():
    # Find the latest raw files
    raw_files = glob.glob('data/raw_*.csv')
    raw_yfinance_files = glob.glob('data/raw_yfinance_*.csv')

    if not raw_files and not raw_yfinance_files:
        print("No raw data files found")
        return

    # Process Nasdaq raw data
    if raw_files:
        latest_raw = max(raw_files, key=os.path.getctime)
        timestamp = os.path.basename(latest_raw).split('raw_')[1].split('.csv')[0]
        
        print(f"Processing Nasdaq raw data: {latest_raw}")
        cleaned_nasdaq_df = clean_data(latest_raw)
        
        if not cleaned_nasdaq_df.empty:
            clean_filename = f'data/cleaned_{timestamp}.csv'
            cleaned_nasdaq_df.to_csv(clean_filename, index=False)
            print(f"Cleaned Nasdaq data saved to {clean_filename}")
        else:
            print(f"No valid Nasdaq data after cleaning for {latest_raw}")
    else:
        print("No Nasdaq raw data files found")

    # Process yfinance raw data
    if raw_yfinance_files:
        latest_yfinance_raw = max(raw_yfinance_files, key=os.path.getctime)
        timestamp = os.path.basename(latest_yfinance_raw).split('raw_yfinance_')[1].split('.csv')[0]
        
        print(f"Processing yfinance raw data: {latest_yfinance_raw}")
        cleaned_yfinance_df = clean_data(latest_yfinance_raw)
        
        if not cleaned_yfinance_df.empty:
            clean_yfinance_filename = f'data/cleaned_yfinance_{timestamp}.csv'
            cleaned_yfinance_df.to_csv(clean_yfinance_filename, index=False)
            print(f"Cleaned yfinance data saved to {clean_yfinance_filename}")
        else:
            print(f"No valid yfinance data after cleaning for {latest_yfinance_raw}")
    else:
        print("No yfinance raw data files found")

main()
