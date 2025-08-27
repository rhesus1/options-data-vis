import pandas as pd
import os
import glob
import json
from datetime import datetime

def is_valid_timestamp(timestamp):
    """Validate timestamp format (YYYYMMDD_HHMM)."""
    if len(timestamp) != 13 or '_' not in timestamp:
        return False
    date_part, time_part = timestamp.split('_')
    if len(date_part) != 8 or len(time_part) != 4:
        return False
    try:
        datetime.strptime(date_part, '%Y%m%d')
        datetime.strptime(time_part, '%H%M')
        return True
    except ValueError:
        return False

def split_csv_by_ticker(input_file, output_dir, file_type):
    """
    Split a CSV file by ticker and save to the new folder structure.
    
    Args:
        input_file (str): Path to the input CSV file (e.g., 'data/raw_yfinance_20250825_2124.csv').
        output_dir (str): Base directory for output (e.g., 'data/20250825_2124/raw_yfinance/').
        file_type (str): Type of file ('raw_yfinance', 'cleaned', etc.).
    """
    log_message = f"Attempting to process file: {input_file}\n"
    print(log_message)
    with open('restructure_log.txt', 'a') as f:
        f.write(log_message)
    
    if not os.path.exists(input_file):
        log_message = f"Input file {input_file} does not exist, skipping\n"
        print(log_message)
        with open('restructure_log.txt', 'a') as f:
            f.write(log_message)
        return False
    
    try:
        df = pd.read_csv(input_file)
        log_message = f"Successfully read {input_file}, rows: {len(df)}, columns: {list(df.columns)}\n"
        print(log_message)
        with open('restructure_log.txt', 'a') as f:
            f.write(log_message)
    except Exception as e:
        log_message = f"Error reading {input_file}: {e}\n"
        print(log_message)
        with open('restructure_log.txt', 'a') as f:
            f.write(log_message)
        return False
    
    if 'Ticker' not in df.columns:
        log_message = f"No 'Ticker' column in {input_file}, skipping\n"
        print(log_message)
        with open('restructure_log.txt', 'a') as f:
            f.write(log_message)
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker in df['Ticker'].unique():
        if pd.isna(ticker) or not ticker:
            log_message = f"Skipping invalid ticker (NaN or empty) in {input_file}\n"
            print(log_message)
            with open('restructure_log.txt', 'a') as f:
                f.write(log_message)
            continue
        ticker_df = df[df['Ticker'] == ticker]
        output_file = f"{output_dir}/{file_type}_{ticker}.csv"
        try:
            ticker_df.to_csv(output_file, index=False)
            log_message = f"Saved {file_type}_{ticker}.csv to {output_dir}, rows: {len(ticker_df)}\n"
            print(log_message)
            with open('restructure_log.txt', 'a') as f:
                f.write(log_message)
        except Exception as e:
            log_message = f"Error saving {output_file}: {e}\n"
            print(log_message)
            with open('restructure_log.txt', 'a') as f:
                f.write(log_message)
            return False
    return True

def clean_incorrect_files(base_dir):
    """Remove files with incorrect '_yfinance_yfinance' naming."""
    log_message = f"Cleaning incorrect files in {base_dir}\n"
    print(log_message)
    with open('restructure_log.txt', 'a') as f:
        f.write(log_message)
    
    incorrect_pattern = f"{base_dir}/*/*_yfinance_yfinance_*.csv"
    incorrect_files = glob.glob(incorrect_pattern, recursive=True)
    for file in incorrect_files:
        try:
            os.remove(file)
            log_message = f"Deleted incorrect file: {file}\n"
            print(log_message)
            with open('restructure_log.txt', 'a') as f:
                f.write(log_message)
        except Exception as e:
            log_message = f"Error deleting {file}: {e}\n"
            print(log_message)
            with open('restructure_log.txt', 'a') as f:
                f.write(log_message)

def main():
    # Create a log file for debugging
    log_file = 'restructure_log.txt'
    with open(log_file, 'w') as f:
        f.write(f"Starting restructure at {datetime.now()}\n")
    
    # Initialize dates.json
    dates_file = 'data/dates.json'
    if os.path.exists(dates_file):
        try:
            with open(dates_file, 'r') as f:
                dates = json.load(f)
            # Clean invalid entries
            dates = [d for d in dates if is_valid_timestamp(d)]
            log_message = f"Loaded and cleaned dates.json: {dates}\n"
            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message)
        except Exception as e:
            log_message = f"Error reading dates.json: {e}, initializing empty list\n"
            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message)
            dates = []
    else:
        log_message = "dates.json not found, initializing empty list\n"
        print(log_message)
        with open(log_file, 'a') as f:
            f.write(log_message)
        dates = []
    
    # Define file patterns for yfinance, Nasdaq, and ranking (excluding historic)
    file_patterns = [
        ('data/raw_yfinance_[0-9]*.csv', 'raw_yfinance', ''),
        ('data/cleaned_yfinance_[0-9]*.csv', 'cleaned_yfinance', ''),
        ('data/processed_yfinance_[0-9]*.csv', 'processed_yfinance', ''),
        ('data/skew_metrics_yfinance_[0-9]*.csv', 'skew_metrics_yfinance', ''),
        ('data/slope_metrics_yfinance_[0-9]*.csv', 'slope_metrics_yfinance', ''),
        ('data/ranking_yfinance_[0-9]*.csv', 'ranking', '_yfinance'),
        ('data/raw_[0-9]*.csv', 'raw', ''),
        ('data/cleaned_[0-9]*.csv', 'cleaned', ''),
        ('data/processed_[0-9]*.csv', 'processed', ''),
        ('data/skew_metrics_[0-9]*.csv', 'skew_metrics', ''),
        ('data/slope_metrics_[0-9]*.csv', 'slope_metrics', ''),
        ('data/ranking_[0-9]*.csv', 'ranking', '')
    ]
    
    # Collect all files and their timestamps
    file_timestamp_map = {}
    timestamps = set()
    for pattern, file_type, ticker_suffix in file_patterns:
        files = glob.glob(pattern)
        log_message = f"Found {len(files)} files for pattern {pattern}: {files}\n"
        print(log_message)
        with open(log_file, 'a') as f:
            f.write(log_message)
        for file in files:
            filename = os.path.basename(file)
            try:
                # Extract timestamp (e.g., '20250825_2124' from 'ranking_yfinance_20250825_2124.csv')
                if file_type == 'ranking' and 'yfinance' in filename:
                    timestamp = filename.split('ranking_yfinance_')[1].split('.csv')[0]
                else:
                    timestamp = filename.split(f'{file_type}_')[1].split('.csv')[0]
                if not is_valid_timestamp(timestamp):
                    log_message = f"Invalid timestamp format in {filename}: {timestamp}, skipping\n"
                    print(log_message)
                    with open(log_file, 'a') as f:
                        f.write(log_message)
                    continue
                timestamps.add(timestamp)
                file_timestamp_map[file] = (timestamp, file_type, ticker_suffix)
                log_message = f"Extracted timestamp {timestamp} from {filename}\n"
                print(log_message)
                with open(log_file, 'a') as f:
                    f.write(log_message)
            except IndexError:
                log_message = f"Could not extract timestamp from {filename}, skipping\n"
                print(log_message)
                with open(log_file, 'a') as f:
                    f.write(log_message)
    
    # Update dates.json
    if timestamps:
        dates = list(set(dates + list(timestamps)))
        dates = [d for d in dates if is_valid_timestamp(d)]
        dates.sort(reverse=True)
        try:
            with open(dates_file, 'w') as f:
                json.dump(dates, f)
            log_message = f"Updated dates.json with timestamps: {dates}\n"
            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message)
        except Exception as e:
            log_message = f"Error writing to dates.json: {e}\n"
            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message)
    
    # Clean incorrect files before processing
    for timestamp in timestamps:
        base_dir = f'data/{timestamp}'
        clean_incorrect_files(base_dir)
    
    # Process each file
    for file, (timestamp, file_type, ticker_suffix) in file_timestamp_map.items():
        log_message = f"Processing file: {file} for timestamp: {timestamp}\n"
        print(log_message)
        with open(log_file, 'a') as f:
            f.write(log_message)
        base_dir = f'data/{timestamp}'
        dir_map = {
            'raw_yfinance': f'{base_dir}/raw_yfinance',
            'cleaned_yfinance': f'{base_dir}/cleaned_yfinance',
            'processed_yfinance': f'{base_dir}/processed_yfinance',
            'skew_metrics_yfinance': f'{base_dir}/skew_metrics_yfinance',
            'slope_metrics_yfinance': f'{base_dir}/slope_metrics_yfinance',
            'raw': f'{base_dir}/raw',
            'cleaned': f'{base_dir}/cleaned',
            'processed': f'{base_dir}/processed',
            'skew_metrics': f'{base_dir}/skew_metrics',
            'slope_metrics': f'{base_dir}/slope_metrics',
            'ranking': f'{base_dir}/ranking'
        }
        if file_type == 'ranking':
            # Move ranking file without splitting
            os.makedirs(dir_map[file_type], exist_ok=True)
            new_ranking_file = f'{dir_map[file_type]}/ranking{ticker_suffix}.csv'
            try:
                os.rename(file, new_ranking_file)
                log_message = f"Moved {file} to {new_ranking_file}\n"
                print(log_message)
                with open(log_file, 'a') as f:
                    f.write(log_message)
            except Exception as e:
                log_message = f"Error moving {file}: {e}\n"
                print(log_message)
                with open(log_file, 'a') as f:
                    f.write(log_message)
        else:
            # Split other files by ticker
            split_csv_by_ticker(file, dir_map[file_type], file_type)

main()
