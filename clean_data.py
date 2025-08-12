# clean_data.py
import pandas as pd
import glob
import os

def main():
    raw_files = glob.glob('data/raw_*.csv')
    if not raw_files:
        print("No raw data files found")
        return
    latest_raw = max(raw_files, key=os.path.getctime)
    df = pd.read_csv(latest_raw, parse_dates=['Expiry'])
    
    volume_threshold = df['Volume'].quantile(0.25)
    oi_threshold = df['Open Interest'].quantile(0.25)
    
    df = df[
        (df['Volume'] >= volume_threshold) &
        (df['Open Interest'] >= oi_threshold) &
        (df['Bid'] >= 0) &
        (df['Ask'] >= 0) &
        (df['Bid'] <= df['Ask'])
    ]
    
    timestamp = os.path.basename(latest_raw).split('raw_')[1].split('.csv')[0]
    clean_filename = f'data/cleaned_{timestamp}.csv'
    df.to_csv(clean_filename, index=False)
    print(f"Cleaned data saved to {clean_filename}")

main()
