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
        group = group[(group['Bid'] + group['Ask']) > 0]
        mid = (group['Bid'] + group['Ask']) / 2
        spread = group['Ask'] - group['Bid']
        sm_percent = 100 * (spread / mid)
        group['weight'] = np.log(1 + group['Open Interest']) / sm_percent
        group = group[np.isfinite(group['weight'])]
        if not group.empty:
            n = len(group)
            threshold_index = int(n * 0.25)
            group = group.sort_values(by='weight').iloc[threshold_index:]
        group = group.drop(columns=['weight'])
        cleaned_groups.append(group)
    
    if cleaned_groups:
        df = pd.concat(cleaned_groups)
    else:
        df = pd.DataFrame()
    return df

def main():
    timestamp_dirs = [d for d in glob.glob('data/*') if os.path.isdir(d) and d.split('/')[-1].replace('_', '').isdigit() and len(d.split('/')[-1]) == 13]
    if not timestamp_dirs:
        print("No timestamp folders found")
        return
    latest_timestamp_dir = max(timestamp_dirs, key=os.path.getctime)
    timestamp = os.path.basename(latest_timestamp_dir)
    
    # Process yfinance raw data
    raw_yfinance_dir = f'data/{timestamp}/raw_yfinance'
    if os.path.exists(raw_yfinance_dir):
        raw_yfinance_files = glob.glob(f'{raw_yfinance_dir}/raw_yfinance_*.csv')
        cleaned_yfinance_dir = f'data/{timestamp}/cleaned_yfinance'
        os.makedirs(cleaned_yfinance_dir, exist_ok=True)
        for raw_file in raw_yfinance_files:
            ticker = os.path.basename(raw_file).split('raw_yfinance_')[1].split('.csv')[0]
            print(f"Processing yfinance raw data: {raw_file}")
            cleaned_df = clean_data(raw_file)
            if not cleaned_df.empty:
                clean_filename = f'{cleaned_yfinance_dir}/cleaned_yfinance_{ticker}.csv'
                cleaned_df.to_csv(clean_filename, index=False)
                print(f"Cleaned yfinance data for {ticker} saved to {clean_filename}")
            else:
                print(f"No valid yfinance data after cleaning for {raw_file}")
    else:
        print("No yfinance raw directory found")

main()
