import pandas as pd
import os
import glob
import json
from datetime import datetime

def split_csv_by_ticker(input_file, output_dir, prefix, file_type):
    """
    Split a CSV file by ticker and save to the new folder structure.
    
    Args:
        input_file (str): Path to the input CSV file (e.g., 'data/raw_yfinance_20250827_0929.csv').
        output_dir (str): Base directory for output (e.g., 'data/20250827_0929/raw_yfinance/').
        prefix (str): Prefix for file naming ('_yfinance' or '').
        file_type (str): Type of file ('raw', 'raw_yfinance', 'cleaned', etc.).
    """
    print(f"Processing file: {input_file}")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist, skipping")
        return
    
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully read {input_file}, rows: {len(df)}")
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    if 'Ticker' not in df.columns:
        print(f"No 'Ticker' column in {input_file}, skipping")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker in df['Ticker'].unique():
        if pd.isna(ticker):
            print(f"Skipping invalid ticker (NaN) in {input_file}")
            continue
        ticker_df = df[df['Ticker'] == ticker]
        output_file = f"{output_dir}/{file_type}{prefix}_{ticker}.csv"
        try:
            ticker_df.to_csv(output_file, index=False)
            print(f"Saved {file_type}{prefix}_{ticker}.csv to {output_dir}, rows: {len(ticker_df)}")
        except Exception as e:
            print(f"Error saving {output_file}: {e}")

def main():
    # Initialize dates.json if it doesn't exist
    dates_file = 'data/dates.json'
    if os.path.exists(dates_file):
        try:
            with open(dates_file, 'r') as f:
                dates = json.load(f)
            print(f"Loaded existing dates.json: {dates}")
        except Exception as e:
            print(f"Error reading dates.json: {e}, initializing empty list")
            dates = []
    else:
        print("dates.json not found, initializing empty list")
        dates = []
    
    # Define all file patterns, including yfinance
    file_patterns = [
        ('data/raw_[0-9]*.csv', 'raw', ''),
        ('data/raw_yfinance_[0-9]*.csv', 'raw_yfinance', '_yfinance'),
        ('data/cleaned_[0-9]*.csv', 'cleaned', ''),
        ('data/cleaned_yfinance_[0-9]*.csv', 'cleaned_yfinance', '_yfinance'),
        ('data/processed_[0-9]*.csv', 'processed', ''),
        ('data/processed_yfinance_[0-9]*.csv', 'processed_yfinance', '_yfinance'),
        ('data/skew_metrics_[0-9]*.csv', 'skew_metrics', ''),
        ('data/skew_metrics_yfinance_[0-9]*.csv', 'skew_metrics_yfinance', '_yfinance'),
        ('data/slope_metrics_[0-9]*.csv', 'slope_metrics', ''),
        ('data/slope_metrics_yfinance_[0-9]*.csv', 'slope_metrics_yfinance', '_yfinance'),
        ('data/historic_[0-9]*.csv', 'historic', '')
    ]
    
    # Extract timestamps from files
    timestamps = set()
    for pattern, file_type, prefix in file_patterns:
        files = glob.glob(pattern)
        print(f"Found {len(files)} files for pattern {pattern}: {files}")
        for file in files:
            filename = os.path.basename(file)
            try:
                # Extract timestamp (e.g., '20250827_0929' from 'raw_yfinance_20250827_0929.csv')
                if prefix:
                    timestamp = filename.split(f'{file_type}{prefix}_')[1].split('.csv')[0]
                else:
                    timestamp = filename.split(f'{file_type}_')[1].split('.csv')[0]
                timestamps.add(timestamp)
            except IndexError:
                print(f"Could not extract timestamp from {filename}, skipping")
    
    # Update dates.json
    for timestamp in timestamps:
        if timestamp not in dates:
            dates.append(timestamp)
    if timestamps:
        dates.sort(reverse=True)
        try:
            with open(dates_file, 'w') as f:
                json.dump(dates, f)
            print(f"Updated dates.json with timestamps: {timestamps}")
        except Exception as e:
            print(f"Error writing to dates.json: {e}")
    
    # Process each timestamp
    for timestamp in timestamps:
        print(f"Processing timestamp: {timestamp}")
        base_dir = f'data/{timestamp}'
        
        # Define output directories
        dir_map = {
            'raw': f'{base_dir}/raw',
            'raw_yfinance': f'{base_dir}/raw_yfinance',
            'historic': f'{base_dir}/historic',
            'cleaned': f'{base_dir}/cleaned',
            'cleaned_yfinance': f'{base_dir}/cleaned_yfinance',
            'processed': f'{base_dir}/processed',
            'processed_yfinance': f'{base_dir}/processed_yfinance',
            'skew_metrics': f'{base_dir}/skew_metrics',
            'skew_metrics_yfinance': f'{base_dir}/skew_metrics_yfinance',
            'slope_metrics': f'{base_dir}/slope_metrics',
            'slope_metrics_yfinance': f'{base_dir}/slope_metrics_yfinance'
        }
        
        # Process each file type
        for pattern, file_type, prefix in file_patterns:
            files = glob.glob(f'data/{file_type}_{prefix}{timestamp}.csv')
            for file in files:
                split_csv_by_ticker(file, dir_map[file_type], prefix, file_type)
        
        # Handle ranking data (move without splitting)
        ranking_dir = f'{base_dir}/ranking'
        os.makedirs(ranking_dir, exist_ok=True)
        for ranking_file in glob.glob(f'data/ranking_{prefix}{timestamp}.csv'):
            new_ranking_file = f'{ranking_dir}/ranking{prefix.replace("_", "")}.csv'
            try:
                os.rename(ranking_file, new_ranking_file)
                print(f"Moved {ranking_file} to {new_ranking_file}")
            except Exception as e:
                print(f"Error moving {ranking_file}: {e}")

main()
