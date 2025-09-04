import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import json
from scipy.stats import kurtosis

def fetch_option_data_yfinance(ticker):
    print(f"Fetching option data for {ticker} from yfinance...")
    stock = yf.Ticker(ticker)
    option_data = []
    expiries = stock.options
    last_stock_price = stock.info.get('regularMarketPrice', stock.info.get('lastPrice', None))
    bid_stock = stock.info.get('bid', None)
    ask_stock = stock.info.get('ask', None)
    for expiry in expiries:
        chain = stock.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts
        calls['Type'] = 'Call'
        puts['Type'] = 'Put'
        calls['Expiry'] = expiry
        puts['Expiry'] = expiry
        calls['Ticker'] = ticker
        puts['Ticker'] = ticker
        calls['Last Stock Price'] = last_stock_price
        puts['Last Stock Price'] = last_stock_price
        calls['Bid Stock'] = bid_stock
        puts['Bid Stock'] = bid_stock
        calls['Ask Stock'] = ask_stock
        puts['Ask Stock'] = ask_stock
        calls['Moneyness'] = last_stock_price / calls['strike'] if last_stock_price else np.nan
        puts['Moneyness'] = puts['strike'] / last_stock_price if last_stock_price else np.nan
        option_data.append(calls)
        option_data.append(puts)
    if not option_data:
        return pd.DataFrame()
    yfinance_df = pd.concat(option_data, ignore_index=True)
    columns = ['Ticker', 'Contract Name', 'Type', 'Expiry', 'Strike', 'Bid', 'Ask', 'Volume', 'Open Interest', 'Implied Volatility', 'Last Stock Price', 'Bid Stock', 'Ask Stock', 'Moneyness']
    yfinance_df = yfinance_df.rename(columns={
        'contractSymbol': 'Contract Name',
        'strike': 'Strike',
        'bid': 'Bid',
        'ask': 'Ask',
        'volume': 'Volume',
        'openInterest': 'Open Interest',
        'impliedVolatility': 'Implied Volatility'
    })
    yfinance_df['Strike'] = (yfinance_df['Strike'] / 0.01) * 0.01
    yfinance_df = yfinance_df[columns]
    return yfinance_df

def fetch_historic_data(ticker):
    print(f"Updating data for {ticker}...")
    file_path = f'data/history/history_{ticker}.csv'
    columns = [
        'Ticker', 'Date', 'Open', 'High', 'Low', 'Close',
        'Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100',
        'Realised_Vol_Close_180', 'Realised_Vol_Close_252',
        'Vol_of_Vol_100d', 'Vol_of_Vol_100d_Percentile', 'Kurtosis_100d'
    ]

    # Check if the CSV file exists
    file_exists = os.path.isfile(file_path)

    # Load historical data from CSV if it exists
    if file_exists:
        hist = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    else:
        # If no CSV exists, fetch enough historical data for calculations (e.g., 252 days)
        print(f"No existing file found for {ticker}. Fetching initial historical data...")
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')  # Fetch 1 year to cover 252-day window
        if hist.empty:
            print(f"No data found for {ticker}")
            return pd.DataFrame()
        hist = hist[['Open', 'High', 'Low', 'Close']]

    # Fetch today's data
    stock = yf.Ticker(ticker)
    today_data = stock.history(period='1d')
    if today_data.empty:
        print(f"No data available for {ticker} today")
        return hist

    # Prepare today's data
    today_data = today_data[['Open', 'High', 'Low', 'Close']]
    today_data['Date'] = today_data.index.strftime('%Y-%m-%d')
    today_data['Ticker'] = ticker

    # If CSV is empty or new, compute all metrics for historical data
    if not file_exists or hist.empty:
        hist = pd.concat([hist, today_data], axis=0)
        hist['Log_Return_Close'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hist['Realised_Vol_Close_30'] = hist['Log_Return_Close'].rolling(window=30).std() * np.sqrt(252) * 100
        hist['Realised_Vol_Close_60'] = hist['Log_Return_Close'].rolling(window=60).std() * np.sqrt(252) * 100
        hist['Realised_Vol_Close_100'] = hist['Log_Return_Close'].rolling(window=100).std() * np.sqrt(252) * 100
        hist['Realised_Vol_Close_180'] = hist['Log_Return_Close'].rolling(window=180).std() * np.sqrt(252) * 100
        hist['Realised_Vol_Close_252'] = hist['Log_Return_Close'].rolling(window=252).std() * np.sqrt(252) * 100
        vol_series = hist['Realised_Vol_Close_100'].copy()
        hist['Vol_of_Vol_100d'] = vol_series.rolling(window=100, min_periods=100).std().round(2)
        hist['Vol_of_Vol_100d_Percentile'] = vol_series.rolling(window=100, min_periods=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x.dropna()) >= 100 else np.nan, raw=False
        ).round(2)
        hist['Kurtosis_100d'] = vol_series.rolling(window=100, min_periods=100).apply(
            lambda x: kurtosis(x.dropna(), nan_policy='omit') if len(x.dropna()) >= 100 else np.nan, raw=False
        ).round(2)
    else:
        # Use existing metrics from CSV and compute only for today
        # Get the last day's Close for Log_Return_Close calculation
        last_close = hist['Close'].iloc[-1] if not hist.empty else None
        today_close = today_data['Close'].iloc[0]
        today_data['Log_Return_Close'] = np.log(today_close / last_close) if last_close else np.nan

        # Combine historical Log_Return_Close with today's for rolling calculations
        log_returns = pd.concat([
            hist['Log_Return_Close'][-251:],  # Get up to 252 days for longest window
            pd.Series(today_data['Log_Return_Close'].iloc[0], index=today_data.index)
        ])

        # Calculate today's realized volatilities
        today_data['Realised_Vol_Close_30'] = log_returns.rolling(window=30).std()[-1] * np.sqrt(252) * 100
        today_data['Realised_Vol_Close_60'] = log_returns.rolling(window=60).std()[-1] * np.sqrt(252) * 100
        today_data['Realised_Vol_Close_100'] = log_returns.rolling(window=100).std()[-1] * np.sqrt(252) * 100
        today_data['Realised_Vol_Close_180'] = log_returns.rolling(window=180).std()[-1] * np.sqrt(252) * 100
        today_data['Realised_Vol_Close_252'] = log_returns.rolling(window=252).std()[-1] * np.sqrt(252) * 100

        # Calculate Vol of Vol, Percentile, and Kurtosis for 100-day Realised Volatility
        vol_series = pd.concat([
            hist['Realised_Vol_Close_100'][-99:],  # Get up to 99 days for 100-day window
            pd.Series(today_data['Realised_Vol_Close_100'].iloc[0], index=today_data.index)
        ])
        today_data['Vol_of_Vol_100d'] = vol_series.rolling(window=100, min_periods=100).std()[-1].round(2)
        today_data['Vol_of_Vol_100d_Percentile'] = vol_series.rolling(window=100, min_periods=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x.dropna()) >= 100 else np.nan, raw=False
        )[-1].round(2)
        today_data['Kurtosis_100d'] = vol_series.rolling(window=100, min_periods=100).apply(
            lambda x: kurtosis(x.dropna(), nan_policy='omit') if len(x.dropna()) >= 100 else np.nan, raw=False
        )[-1].round(2)

        # Combine historical and today's data
        hist = pd.concat([hist, today_data[columns]], axis=0)

    # Ensure columns are in the correct order
    hist = hist[columns]

    # Append today's row to the CSV
    today_data = hist.tail(1)
    today_data.to_csv(
        file_path,
        mode='a',
        index=False,
        header=not file_exists  # Write headers only if file doesn't exist
    )

    print(f"Appended today's data for {ticker} to {file_path}")
    return hist

def main():
    with open('tickers.txt', 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]
    print(f"Number of tickers: {len(tickers)}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_dir = f'data/{timestamp}'
    raw_yfinance_dir = f'{base_dir}/raw_yfinance'
    historic_dir = f'{base_dir}/historic'
    os.makedirs(raw_yfinance_dir, exist_ok=True)
    os.makedirs(historic_dir, exist_ok=True)
    for ticker in tickers:
        yfinance_df = fetch_option_data_yfinance(ticker)
        if not yfinance_df.empty:
            yfinance_filename = f'{raw_yfinance_dir}/raw_yfinance_{ticker}.csv'
            yfinance_df.to_csv(yfinance_filename, index=False)
            print(f"yfinance raw data for {ticker} saved to {yfinance_filename}")
        df_hist = fetch_historic_data(ticker)
        if not df_hist.empty:
            hist_filename = f'{historic_dir}/historic_{ticker}.csv'
            df_hist.to_csv(hist_filename, index=False)
            print(f"Historic data for {ticker} saved to {hist_filename}")
        time.sleep(1)
    # Update dates.json with the new timestamp
    dates_file = 'data/dates.json'
    if os.path.exists(dates_file):
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    else:
        dates = []
    if timestamp not in dates:
        dates.append(timestamp)
        dates.sort(reverse=True)
        with open(dates_file, 'w') as f:
            json.dump(dates, f)

main()
