# clean_data.py
import pandas as pd
import glob
import os

def main():
    raw_files = glob.glob('data/raw_*.csv')
    if not raw_files:
        print("No raw data files found")
        return
    latest_raw = max(raw_files, key=os.path.getctime)
    df = pd.read_csv(latest_raw, parse_dates=['Expiry'])

    duplicates = df.duplicated(subset=['Contract Name']).sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['Contract Name'])

    key_cols = ['Ticker', 'Type', 'Strike', 'Moneyness', 'Bid', 'Ask', 'Volume', 
                'Open Interest', 'Last Option Price', 'Implied Volatility', 'Last Stock Price']
    na_before = df[key_cols].isna().any(axis=1).sum()
    df = df.dropna(subset=key_cols)

    df = df[
        (df['Volume'] >= 0) &
        (df['Open Interest'] >= 0) &
        (df['Bid'] >= 0) &
        (df['Ask'] >= 0) &
        #(df['Bid'] < df['Ask']) &  # Strict < to ensure positive spread
        #(df['Bid'] <= df['Last Option Price']) &
        #(df['Last Option Price'] <= df['Ask']) &
        (df['Implied Volatility'] > 0.01)  # Remove zero IV
    ]
    
    volume_threshold = df['Volume'].quantile(0.25)
    oi_threshold = df['Open Interest'].quantile(0.25)
    
    df = df[
        (df['Volume'] >= volume_threshold) &
        (df['Open Interest'] >= oi_threshold)
    ]

    df['Spread'] = df['Ask'] - df['Bid']
    df['Mid Price'] = (df['Ask'] + df['Bid']) / 2
    df = df[df['Spread'] / df['Mid Price'] <= 0.2]

    for col in ['Volume', 'Open Interest', 'Implied Volatility']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    
    timestamp = os.path.basename(latest_raw).split('raw_')[1].split('.csv')[0]
    clean_filename = f'data/cleaned_{timestamp}.csv'
    df.to_csv(clean_filename, index=False)
    print(f"Cleaned data saved to {clean_filename}")

main()
