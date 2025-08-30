import sys
from calculate_ranking_metrics import calculate_ranking_metrics

def main():
    if len(sys.argv) < 2:
        with open('ranking_error.log', 'a') as f:
            f.write("No timestamp provided to Force_Calculate_Ranking.py\n")
        return
    
    timestamp = sys.argv[1]
    try:
        sources = ['yfinance']
        calculate_ranking_metrics(timestamp, sources)
    except Exception as e:
        with open('ranking_error.log', 'a') as f:
            f.write(f"Error in calculate_ranking_metrics for timestamp {timestamp}: {str(e)}\n")
        raise

main()
