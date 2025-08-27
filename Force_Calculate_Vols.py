import sys
import glob
import os
import json
from calculate_data import process_data

def main():
    # Check for timestamp argument
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
    else:
        timestamp_dirs = [d for d in glob.glob('data/*') if os.path.isdir(d) and d.split('/')[-1].replace('_', '').isdigit() and len(d.split('/')[-1]) == 13]
        if not timestamp_dirs:
            print("No timestamp folders found")
            return
        latest_timestamp_dir = max(timestamp_dirs, key=os.path.getctime)
        timestamp = os.path.basename(latest_timestamp_dir)
    
    # Initialize dates.json
    dates_file = 'data/dates.json'
    if os.path.exists(dates_file):
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    else:
        dates = []
    
    # Add timestamp if not present
    if timestamp not in dates:
        dates.append(timestamp)
        dates.sort(reverse=True)
        with open(dates_file, 'w') as f:
            json.dump(dates, f)
        print(f"Updated dates list in {dates_file} with timestamp: {timestamp}")
    
    # Process Nasdaq data
    print(f"Processing Nasdaq data for {timestamp}")
    process_data(timestamp, prefix="")
    
    # Process yfinance data
    print(f"Processing yfinance data for {timestamp}")
    process_data(timestamp, prefix="_yfinance")

main()
