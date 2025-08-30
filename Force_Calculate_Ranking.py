import sys
from calculate_ranking_metrics import calculate_ranking_metrics

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    
    timestamp = sys.argv[1]
    sources = ['yfinance', 'nasdaq']
    calculate_ranking_metrics(timestamp, sources)

main()
