import sys
from preprocess_data import save_tables

def main():
    if len(sys.argv) < 2:
        print("Usage: python force_preprocess_data.py <timestamp>")
        sys.exit(1)
    
    timestamp = sys.argv[1]
    sources = ['yfinance']  # Consistent with Force_Calculate_Ranking.py
    for source in sources:
        save_tables(timestamp, source)
        print(f"Precomputed tables generated for {timestamp}, source {source}")

main()
