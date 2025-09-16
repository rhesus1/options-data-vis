import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python force_calculate_vols.py <timestamp>")
        sys.exit(1)
    timestamp = sys.argv[1]
    from calculate_vols import process_volumes
    process_volumes(timestamp)

if __name__ == '__main__':
    main()
