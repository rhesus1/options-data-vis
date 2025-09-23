import sys
import time
import os
from preprocess_data import save_tables, generate_returns_summary

def main():
    start_time = time.time()
    base_path = 'data'

    if len(sys.argv) < 2:
        print("Usage: python force_preprocess_data.py <timestamp>", flush=True)
        sys.exit(1)

    timestamp = sys.argv[1]
    dates_file = os.path.join(base_path, 'dates.json')
    try:
        with open(dates_file, 'r') as f:
            dates = json.load(f)
        if timestamp not in dates:
            print(f"Error: Provided timestamp {timestamp} not found in {dates_file}", flush=True)
            sys.exit(1)
        print(f"Processing timestamp {timestamp}", flush=True)
    except Exception as e:
        print(f"Error loading {dates_file}: {e}", flush=True)
        sys.exit(1)

    sources = ['yfinance']
    for source in sources:
        print(f"Starting table generation for source {source}...", flush=True)
        try:
            save_tables(timestamp, source, base_path)
            print(f"Precomputed tables generated for {timestamp}, source {source}", flush=True)
        except Exception as e:
            print(f"Error generating tables for {source}: {e}", flush=True)

    print(f"Starting generate_returns_summary for BH_HF_Ret_Sept_25.csv...", flush=True)
    try:
        generate_returns_summary(base_path)
        print("generate_returns_summary completed successfully", flush=True)
    except Exception as e:
        print(f"Error executing generate_returns_summary: {e}", flush=True)

    print(f"Total execution time: {time.time() - start_time:.2f} seconds", flush=True)

if __name__ == "__main__":
    main()
