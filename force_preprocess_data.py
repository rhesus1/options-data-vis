import sys
import time
from preprocess_data import save_tables, generate_returns_summary

def main():
    start_time = time.time()
    base_path = 'data'
    
    if len(sys.argv) < 2:
        print("Usage: python force_preprocess_data.py <timestamp>", flush=True)
        sys.exit(1)
    
    timestamp = sys.argv[1]
    sources = ['yfinance']  # Consistent with Force_Calculate_Ranking.py
    for source in sources:
        print(f"Starting table generation for source {source}...", flush=True)
        save_tables(timestamp, source, base_path)
        print(f"Precomputed tables generated for {timestamp}, source {source}", flush=True)
    
    print(f"Starting generate_returns_summary for BH_HF_Ret_Sept_25.csv...", flush=True)
    try:
        generate_returns_summary(base_path)
    except Exception as e:
        print(f"Error executing generate_returns_summary: {e}", flush=True)
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds", flush=True)

if __name__ == "__main__":
    main()
