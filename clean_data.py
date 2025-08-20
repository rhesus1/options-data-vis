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
    
    cleaned_groups = []
    for ticker, group in df.groupby('Ticker'):
        # 1. Remove options with empty Bid or empty Ask
        group = group.dropna(subset=['Bid', 'Ask'])
        
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
        # Avoid division by zero in sm_percent, though already filtered mid > 0; if sm_percent == 0, weight would be inf, but handle if needed
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
    # Find the latest raw files
    raw_files = glob.glob('data/raw_[0-9]*.csv')
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
