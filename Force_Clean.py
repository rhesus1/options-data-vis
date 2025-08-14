import pandas as pd
import numpy as np
import glob
import os
import sys

def main():
    # Check for timestamp argument
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
        latest_raw = f'data/raw_{timestamp}.csv'
        if not os.path.exists(latest_raw):
            print(f"Raw data file {latest_raw} not found")
            return
    else:
        raw_files = glob.glob('data/raw_*.csv')
        if not raw_files:
            print("No raw data files found")
            return
        latest_raw = max(raw_files, key=os.path.getctime)
        timestamp = os.path.basename(latest_raw).split('raw_')[1].split('.csv')[0]

    # Read the latest raw data file
    df = pd.read_csv(latest_raw, parse_dates=['Expiry'])

    # Remove duplicates based on Contract Name
    duplicates = df.duplicated(subset=['Contract Name']).sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['Contract Name'])

    # Filter data based on conditions
    df = df[
        (df['Volume'] >= 0) &
        (df['Open Interest'] >= 0) &
        (df['Bid'] >= 0) &
        (df['Ask'] >= 0)
    ]

    # Apply volume and open interest thresholds
    volume_threshold = df['Volume'].quantile(0.1)
    oi_threshold = df['Open Interest'].quantile(0.3)

    df = df[
        (df['Volume'] >= volume_threshold) &
        (df['Open Interest'] >= oi_threshold)
    ]

    # Save cleaned data
    clean_filename = f'data/cleaned_{timestamp}.csv'
    df.to_csv(clean_filename, index=False)
    print(f"Cleaned data saved to {clean_filename}")

if __name__ == "__main__":
    main()
