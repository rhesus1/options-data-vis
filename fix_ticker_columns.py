import pandas as pd
import os
import glob
import sys

def fix_ticker_column(file_path, ticker):
    try:
        df = pd.read_csv(file_path)
        if 'Ticker' not in df.columns:
            df['Ticker'] = ticker
            df.to_csv(file_path, index=False)
            print(f"Added 'Ticker' column to {file_path}")
        elif df['Ticker'].isna().any() or not all(df['Ticker'] == ticker):
            df['Ticker'] = ticker
            df.to_csv(file_path, index=False)
            print(f"Corrected 'Ticker' column in {file_path}")
        else:
            print(f"'Ticker' column is correct in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Get timestamp from command-line argument or use default
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
    else:
        timestamp = '20250909_2233'  # Default timestamp
    print(f"Using timestamp: {timestamp}")

    # Process data/history
    history_dir = 'data/history'
    if os.path.exists(history_dir):
        for file_path in glob.glob(os.path.join(history_dir, 'historic_*.csv')):
            filename = os.path.basename(file_path)
            if filename.startswith('historic_') and filename.endswith('.csv'):
                ticker = filename.split('historic_')[1].split('.csv')[0]
                fix_ticker_column(file_path, ticker)
            else:
                print(f"Skipping invalid file: {file_path}")

    # Process data/{timestamp}/historic
    timestamp_historic_dir = f'data/{timestamp}/historic'
    if os.path.exists(timestamp_historic_dir):
        for file_path in glob.glob(os.path.join(timestamp_historic_dir, 'historic_*.csv')):
            filename = os.path.basename(file_path)
            if filename.startswith('historic_') and filename.endswith('.csv'):
                ticker = filename.split('historic_')[1].split('.csv')[0]
                fix_ticker_column(file_path, ticker)
            else:
                print(f"Skipping invalid file: {file_path}")
    else:
        print(f"Timestamp directory {timestamp_historic_dir} does not exist")

  main()
