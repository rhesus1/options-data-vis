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
    print(f"Fetching historic data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period='max')
    if hist.empty:
        return pd.DataFrame()
    hist = hist[['Open', 'High', 'Low', 'Close']]
    hist['Log_Return_Close'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist['Realised_Vol_Close_30'] = hist['Log_Return_Close'].rolling(window=30).std() * np.sqrt(252) * 100
    hist['Realised_Vol_Close_60'] = hist['Log_Return_Close'].rolling(window=60).std() * np.sqrt(252) * 100
    hist['Realised_Vol_Close_100'] = hist['Log_Return_Close'].rolling(window=100).std() * np.sqrt(252) * 100
    hist['Realised_Vol_Close_180'] = hist['Log_Return_Close'].rolling(window=180).std() * np.sqrt(252) * 100
    hist['Realised_Vol_Close_252'] = hist['Log_Return_Close'].rolling(window=252).std() * np.sqrt(252) * 100
    # Calculate Vol of Vol, Percentile, and Kurtosis for 100-day Realised Volatility
    vol_series = hist['Realised_Vol_Close_100'].copy()
    hist['Vol_of_Vol_100d'] = vol_series.rolling(window=100, min_periods=100).std().round(2)
    hist['Vol_of_Vol_100d_Percentile'] = vol_series.rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x.dropna()) >= 100 else np.nan, raw=False
    ).round(2)
    hist['Kurtosis_100d'] = vol_series.rolling(window=100, min_periods=100).apply(
        lambda x: kurtosis(x.dropna(), nan_policy='omit') if len(x.dropna()) >= 100 else np.nan, raw=False
    ).round(2)
    hist = hist.dropna()
    hist['Date'] = hist.index.strftime('%Y-%m-%d')
    hist['Ticker'] = ticker
    columns = [
        'Ticker', 'Date', 'Open', 'High', 'Low', 'Close',
        'Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100',
        'Realised_Vol_Close_180', 'Realised_Vol_Close_252',
        'Vol_of_Vol_100d', 'Vol_of_Vol_100d_Percentile', 'Kurtosis_100d'
    ]
    return hist[columns]

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
