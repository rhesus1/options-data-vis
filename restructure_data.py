import pandas as pd
import os
import glob
import json
from datetime import datetime

def split_csv_by_ticker(input_file, output_dir, prefix, file_type):
    """
    Split a CSV file by ticker and save to the new folder structure.
    
    Args:
        input_file (str): Path to the input CSV file (e.g., 'data/raw_20250812_2111.csv').
        output_dir (str): Base directory for output (e.g., 'data/20250812_2111/raw/').
        prefix (str): Prefix for file naming ('_yfinance' or '').
        file_type (str): Type of file ('raw', 'cleaned', 'processed', 'skew_metrics', 'slope_metrics').
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found")
        return
    
    df = pd.read_csv(input_file)
    if 'Ticker' not in df.columns:
        print(f"No 'Ticker' column in {input_file}, skipping")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker in df['Ticker'].unique():
        ticker_df = df[df['Ticker'] == ticker]
        output_file = f"{output_dir}/{file_type}{prefix}_{ticker}.csv"
        ticker_df.to_csv(output_file, index=False)
        print(f"Saved {file_type}{prefix}_{ticker}.csv to {output_dir}")

def main():
    # Initialize dates.json if it doesn't exist
    dates_file = 'data/dates.json'
    if os.path.exists(dates_file):
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    else:
        dates = []
    
    # Find all timestamped files in data/
    file_patterns = [
        'data/raw_[0-9]*.csv',
        'data/raw_yfinance_*.csv',
        'data/cleaned_[0-9]*.csv',
        'data/cleaned_yfinance_*.csv',
        'data/processed_[0-9]*.csv',
        'data/processed_yfinance_*.csv',
        'data/skew_metrics_[0-9]*.csv',
        'data/skew_metrics_yfinance_*.csv',
        'data/slope_metrics_[0-9]*.csv',
        'data/slope_metrics_yfinance_*.csv'
    ]
    
    timestamps = set()
    for pattern in file_patterns:
        for file in glob.glob(pattern):
            # Extract timestamp from filename
            filename = os.path.basename(file)
            if 'yfinance' in filename:
                timestamp = filename.split('yfinance_')[1].split('.csv')[0]
            else:
                timestamp = filename.split('_')[1].split('.csv')[0]
            timestamps.add(timestamp)
    
    # Update dates.json
    for timestamp in timestamps:
        if timestamp not in dates:
            dates.append(timestamp)
    if timestamps:
        dates.sort(reverse=True)
        with open(dates_file, 'w') as f:
            json.dump(dates, f)
        print(f"Updated dates.json with timestamps: {timestamps}")
    
    # Process each timestamp
    for timestamp in timestamps:
        base_dir = f'data/{timestamp}'
        
        # Define output directories
        dir_map = {
            'raw': f'{base_dir}/raw',
            'raw_yfinance': f'{base_dir}/raw_yfinance',
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
        file_types = [
            ('raw_[0-9]*.csv', 'raw', ''),
            ('raw_yfinance_*.csv', 'raw_yfinance', '_yfinance'),
            ('cleaned_[0-9]*.csv', 'cleaned', ''),
            ('cleaned_yfinance_*.csv', 'cleaned_yfinance', '_yfinance'),
            ('processed_[0-9]*.csv', 'processed', ''),
            ('processed_yfinance_*.csv', 'processed_yfinance', '_yfinance'),
            ('skew_metrics_[0-9]*.csv', 'skew_metrics', ''),
            ('skew_metrics_yfinance_*.csv', 'skew_metrics_yfinance', '_yfinance'),
            ('slope_metrics_[0-9]*.csv', 'slope_metrics', ''),
            ('slope_metrics_yfinance_*.csv', 'slope_metrics_yfinance', '_yfinance')
        ]
        
        for pattern, file_type, prefix in file_types:
            files = glob.glob(f'data/{file_type}_{prefix}{timestamp}.csv')
            for file in files:
                split_csv_by_ticker(file, dir_map[file_type], prefix, file_type)
        
        # Handle historic data
        historic_file = f'data/historic_{timestamp}.csv'
        if os.path.exists(historic_file):
            split_csv_by_ticker(historic_file, f'{base_dir}/historic', '', 'historic')
        
        # Handle ranking data (no splitting, just move to new directory)
        ranking_dir = f'{base_dir}/ranking'
        os.makedirs(ranking_dir, exist_ok=True)
        for ranking_file in glob.glob(f'data/ranking_{prefix}{timestamp}.csv'):
            new_ranking_file = f'{ranking_dir}/ranking_{prefix.replace("_", "")}.csv'
            os.rename(ranking_file, new_ranking_file)
            print(f"Moved {ranking_file} to {new_ranking_file}")

if __name__ == "__main__":
    main()
