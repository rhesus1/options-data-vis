import pandas as pd
import numpy as np
import glob
import os
import sys

def clean_data(df):
    # Remove duplicates based on Contract Name
    duplicates = df.duplicated(subset=['Contract Name']).sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['Contract Name'])
        print(f"Removed {duplicates} duplicates")
    
    cleaned_groups = []
    for ticker, group in df.groupby('Ticker'):
        # 1. Remove options with empty Bid or empty Ask
        group = group.dropna(subset=['Bid', 'Ask', 'Strike'])
        
        # 2. Filter moneyness: keep >= 0.3 and <= 3.5
        group = group[(group['Moneyness'] >= 0.3) & (group['Moneyness'] <= 3.5)]
        
        # 3. Remove options where both open interest and volume are 0
        group = group[~((group['Open Interest'] == 0) & (group['Volume'] == 0))]
        
        # 4. Remove the bottom 25% based on open interest (sorted ascending)
        if not group.empty:
            n = len(group)
            threshold_index = int(n * 0.25)
            group = group.sort_values(by='Open Interest').iloc[threshold_index:]
        
        # 5. Calculate ln(1 + open interest) / SM%, where SM% = 100 * (ask - bid) / mid, mid = (bid + ask)/2
        # First, filter to avoid division by zero (mid > 0)
        group = group[(group['Bid'] + group['Ask']) > 0]
        
        mid = (group['Bid'] + group['Ask']) / 2
        spread = group['Ask'] - group['Bid']
        sm_percent = 100 * (spread / mid)
        # Avoid division by zero in sm_percent
        group['weight'] = np.log(1 + group['Open Interest']) / sm_percent
        
        # Remove rows where sm_percent == 0 to avoid inf
        group = group[np.isfinite(group['weight'])]
        
        # Remove the bottom 25% based on weight (sorted ascending)
        if not group.empty:
            n = len(group)
            threshold_index = int(n * 0.25)
            group = group.sort_values(by='weight').iloc[threshold_index:]
        
        # Drop the temporary weight column
        group = group.drop(columns=['weight'])
        
        cleaned_groups.append(group)
    
    if cleaned_groups:
        df = pd.concat(cleaned_groups)
    else:
        df = pd.DataFrame()  # Empty if no groups
    
    return df

def main():
    # Check for timestamp argument
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
        latest_raw = f'data/raw_{timestamp}.csv'
        latest_yfinance_raw = f'data/raw_yfinance_{timestamp}.csv'
    else:
        raw_files = glob.glob('data/raw_[0-9]*.csv')
        raw_yfinance_files = glob.glob('data/raw_yfinance_*.csv')
        if not raw_files and not raw_yfinance_files:
            print("No raw data files found")
            return
        latest_raw = max(raw_files, key=os.path.getctime) if raw_files else None
        latest_yfinance_raw = max(raw_yfinance_files, key=os.path.getctime) if raw_yfinance_files else None
        if latest_raw:
            timestamp = os.path.basename(latest_raw).split('raw_')[1].split('.csv')[0]
        elif latest_yfinance_raw:
            timestamp = os.path.basename(latest_yfinance_raw).split('raw_yfinance_')[1].split('.csv')[0]
        else:
            print("No valid timestamp found")
            return
    
    # Process Nasdaq raw data
    if latest_raw and os.path.exists(latest_raw):
        print(f"Processing Nasdaq raw data: {latest_raw}")
        df = pd.read_csv(latest_raw, parse_dates=['Expiry'])
        cleaned_nasdaq_df = clean_data(df)
        if not cleaned_nasdaq_df.empty:
            clean_filename = f'data/cleaned_{timestamp}.csv'
            cleaned_nasdaq_df.to_csv(clean_filename, index=False)
            print(f"Cleaned Nasdaq data saved to {clean_filename}")
        else:
            print(f"No valid Nasdaq data after cleaning for {latest_raw}")
    
    # Process yfinance raw data
    if latest_yfinance_raw and os.path.exists(latest_yfinance_raw):
        print(f"Processing yfinance raw data: {latest_yfinance_raw}")
        df = pd.read_csv(latest_yfinance_raw, parse_dates=['Expiry'])
        cleaned_yfinance_df = clean_data(df)
        if not cleaned_yfinance_df.empty:
            clean_yfinance_filename = f'data/cleaned_yfinance_{timestamp}.csv'
            cleaned_yfinance_df.to_csv(clean_yfinance_filename, index=False)
            print(f"Cleaned yfinance data saved to {clean_yfinance_filename}")
        else:
            print(f"No valid yfinance data after cleaning for {latest_yfinance_raw}")

main()
