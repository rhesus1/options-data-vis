import pandas as pd
import numpy as np
import os
import sys
import json
import glob
from datetime import datetime
import time
from tqdm import tqdm
from scipy.stats import kurtosis

def load_data(timestamp, source, base_path="data"):
    """Load raw data files for a given timestamp and source."""
    start_time = time.time()
    prefix = "_yfinance" if source == "yfinance" else ""
    company_names = None
    ranking = None
    historic = pd.DataFrame()
    events = pd.DataFrame()
    try:
        # Load company names
        if os.path.exists("company_names.txt"):
            company_names = pd.read_csv("company_names.txt", sep="\t")
        else:
            print("Warning: company_names.txt not found.")
        # Load ranking
        ranking_path = f"{base_path}/{timestamp}/ranking/ranking{prefix}.csv"
        if os.path.exists(ranking_path):
            ranking = pd.read_csv(ranking_path)
            numeric_cols = ['Latest Close', 'Realised Volatility 30d (%)', 'Realised Volatility 100d (%)',
                           'Realised Volatility 100d 1d (%)', 'Realised Volatility 100d 1w (%)',
                           'Min Realised Volatility 100d (1y)', 'Max Realised Volatility 100d (1y)',
                           'Mean Realised Volatility 100d (1y)', 'Rvol 100d Percentile (%)',
                           'Rvol 100d Z-Score Percentile (%)', 'Realised Volatility 180d (%)',
                           'Realised Volatility 252d (%)', 'Weighted IV (%)', 'Weighted IV 1d (%)',
                           'Weighted IV 1w (%)', 'Weighted IV 3m (%)', 'Weighted IV 3m 1d (%)',
                           'Weighted IV 3m 1w (%)', 'ATM IV 3m (%)', 'ATM IV 3m 1d (%)',
                           'ATM IV 3m 1w (%)', 'Rvol100d - Weighted IV', 'Volume', 'Volume 1d (%)',
                           'Volume 1w (%)', 'Open Interest', 'OI 1d (%)', 'OI 1w (%)']
            for col in numeric_cols:
                if col in ranking.columns:
                    ranking[col] = pd.to_numeric(ranking[col], errors='coerce')
        else:
            raise FileNotFoundError(f"Ranking file not found: {ranking_path}")
        # Load historic data
        historic_dir = f"{base_path}/{timestamp}/historic"
        if os.path.exists(historic_dir):
            historic_files = glob.glob(f"{historic_dir}/historic_*.csv")
            if historic_files:
                dfs = []
                for file in historic_files:
                    try:
                        df = pd.read_csv(file)
                        if 'Ticker' not in df.columns:
                            ticker = os.path.basename(file).split('historic_')[1].split('.csv')[0].upper()
                            df['Ticker'] = ticker
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        numeric_cols = ['High', 'Low', 'Close', 'Realised_Vol_Close_30', 'Realised_Vol_Close_60',
                                       'Realised_Vol_Close_100', 'Realised_Vol_Close_180', 'Realised_Vol_Close_252']
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Calculate Vol of Vol, Percentile, and Kurtosis for 100-day Realised Volatility
                        if 'Realised_Vol_Close_100' in df.columns:
                            # Ensure the column is a Pandas Series
                            vol_series = df['Realised_Vol_Close_100'].copy()
                            # Vol of Vol: Standard deviation of 100-day realised volatility over a 252-day window
                            df['Vol_of_Vol_100d'] = vol_series.rolling(window=252, min_periods=100).std().round(2)
                            # Percentile of Vol of Vol
                            df['Vol_of_Vol_100d_Percentile'] = vol_series.rolling(window=252, min_periods=100).apply(
                                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x.dropna()) >= 100 else np.nan, raw=False
                            ).round(2)
                            # Kurtosis of 100-day Realised Volatility
                            df['Kurtosis_100d'] = vol_series.rolling(window=252, min_periods=100).apply(
                                lambda x: kurtosis(x.dropna(), nan_policy='omit') if len(x.dropna()) >= 100 else np.nan, raw=False
                            ).round(2)
                        dfs.append(df)
                    except Exception as e:
                        print(f"Warning: Error loading historic file {file}: {e}")
                if dfs:
                    historic = pd.concat(dfs, ignore_index=True)
                    historic = historic.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
                    # Save updated historic files
                    for ticker in historic['Ticker'].unique():
                        ticker_df = historic[historic['Ticker'] == ticker]
                        output_file = f"{historic_dir}/historic_{ticker}.csv"
                        ticker_df.to_csv(output_file, index=False)
                        print(f"Saved updated historic file for {ticker}: {output_file}")
                print(f"Loaded historic data for {len(historic_files)} tickers.")
            else:
                print(f"Warning: No historic_*.csv files found in {historic_dir}")
        else:
            print(f"Warning: Historic directory {historic_dir} not found.")
        # Load events
        events_path = f"{base_path}/Events.csv"
        if os.path.exists(events_path):
            events = pd.read_csv(events_path)
        else:
            print("Warning: Events.csv not found.")
        print(f"Loaded all data in {time.time() - start_time:.2f} seconds")
        return company_names, ranking, historic, events
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        return None, None, None, None

def load_ticker_data(ticker, timestamp, source, base_path="data"):
    """Load ticker-specific data."""
    start_time = time.time()
    prefix = "_yfinance" if source == "yfinance" else ""
    processed = pd.DataFrame()
    cleaned = pd.DataFrame()
    skew = pd.DataFrame()
    try:
        processed_path = f"{base_path}/{timestamp}/processed{prefix}/processed{prefix}_{ticker}.csv"
        if os.path.exists(processed_path):
            processed = pd.read_csv(processed_path)
       
        cleaned_path = f"{base_path}/{timestamp}/cleaned_yfinance/cleaned_yfinance_{ticker}.csv"
        if os.path.exists(cleaned_path):
            cleaned = pd.read_csv(cleaned_path)
       
        skew_path = f"{base_path}/{timestamp}/skew_metrics{prefix}/skew_metrics{prefix}_{ticker}.csv"
        if os.path.exists(skew_path):
            skew = pd.read_csv(skew_path)
       
        print(f"Loaded data for ticker {ticker} in {time.time() - start_time:.2f} seconds")
        return ticker, processed, cleaned, skew
    except Exception as e:
        print(f"Error loading data for ticker {ticker}: {e}")
        return ticker, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def generate_ranking_table(ranking, company_names):
    """Generate ranking table with formatted values and colors."""
    start_time = time.time()
    if ranking is None or ranking.empty:
        print(f"No ranking data in {time.time() - start_time:.2f} seconds")
        return pd.DataFrame(), pd.DataFrame()
    ranking = ranking.copy()
    if company_names is not None and not company_names.empty:
        ranking["Company Name"] = ranking["Ticker"].map(
            company_names.set_index("Ticker")["Company Name"].to_dict()
        ).fillna("N/A")
    else:
        ranking["Company Name"] = "N/A"
   
    columns = [
        "Rank", "Ticker", "Company Name", "Latest Close", "Realised Volatility 30d (%)",
        "Realised Volatility 100d (%)", "Realised Volatility 100d 1d (%)",
        "Realised Volatility 100d 1w (%)", "Min Realised Volatility 100d (1y)",
        "Max Realised Volatility 100d (1y)", "Mean Realised Volatility 100d (1y)",
        "Rvol 100d Percentile (%)", "Rvol 100d Z-Score Percentile (%)",
        "Realised Volatility 180d (%)", "Realised Volatility 252d (%)",
        "Weighted IV (%)", "Weighted IV 1d (%)", "Weighted IV 1w (%)",
        "Weighted IV 3m (%)", "Weighted IV 3m 1d (%)", "Weighted IV 3m 1w (%)",
        "ATM IV 3m (%)", "ATM IV 3m 1d (%)", "ATM IV 3m 1w (%)",
        "Rvol100d - Weighted IV", "Volume", "Volume 1d (%)", "Volume 1w (%)",
        "Open Interest", "OI 1d (%)", "OI 1w (%)"
    ]
   
    for col in columns:
        if col not in ranking.columns:
            ranking[col] = pd.NA if col not in ["Rank", "Ticker", "Company Name"] else "N/A"
   
    ranking['Open Interest Numeric'] = pd.to_numeric(ranking['Open Interest'], errors='coerce').fillna(0)
    ranking["Rank"] = ranking['Open Interest Numeric'].rank(ascending=False, na_option="bottom").astype(int)
    ranking = ranking.drop('Open Interest Numeric', axis=1)
   
    for col in columns:
        if col in ["Volume", "Open Interest"]:
            ranking[col] = np.where(
                ranking[col].notna() & (pd.to_numeric(ranking[col], errors='coerce') > 0) & (pd.to_numeric(ranking[col], errors='coerce').notna()),
                pd.to_numeric(ranking[col], errors='coerce').map(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"),
                "N/A"
            )
        elif col not in ["Rank", "Ticker", "Company Name"]:
            ranking[col] = np.where(
                ranking[col].notna() & (pd.to_numeric(ranking[col], errors='coerce') == pd.to_numeric(ranking[col], errors='coerce')),
                pd.to_numeric(ranking[col], errors='coerce').round(2),
                pd.NA
            )
       
        color_col = f"{col}_Color"
        ranking[color_col] = "#FFFFFF"
        if col in ["Realised Volatility 100d 1d (%)", "Realised Volatility 100d 1w (%)",
                   "Weighted IV 1d (%)", "Weighted IV 1w (%)", "Weighted IV 3m 1d (%)",
                   "Weighted IV 3m 1w (%)", "ATM IV 3m 1d (%)", "ATM IV 3m 1w (%)",
                   "Rvol100d - Weighted IV", "Volume 1d (%)", "Volume 1w (%)",
                   "OI 1d (%)", "OI 1w (%)"]:
            ranking[color_col] = np.where(
                pd.to_numeric(ranking[col], errors='coerce').notna(),
                np.where(pd.to_numeric(ranking[col], errors='coerce') < 0, "#F87171",
                         np.where(pd.to_numeric(ranking[col], errors='coerce') > 0, "#10B981", "#FFFFFF")),
                "#FFFFFF"
            )
   
    color_columns = [f"{col}_Color" for col in columns if col not in ["Rank", "Ticker", "Company Name"]]
    ranking_no_colors = ranking[columns]
    print(f"Generated ranking table in {time.time() - start_time:.2f} seconds")
    return ranking[columns + color_columns], ranking_no_colors

def generate_stock_table(ranking, company_names):
    """Generate stock table with formatted values and colors."""
    start_time = time.time()
    if ranking is None or ranking.empty:
        print(f"No stock data in {time.time() - start_time:.2f} seconds")
        return pd.DataFrame(), pd.DataFrame()
    stock_data = ranking.copy()
    if company_names is not None and not company_names.empty:
        stock_data["Company Name"] = stock_data["Ticker"].map(
            company_names.set_index("Ticker")["Company Name"].to_dict()
        ).fillna("N/A")
    else:
        stock_data["Company Name"] = "N/A"
   
    columns = [
        "Ticker", "Company Name", "Latest Open", "Latest Close", "Latest High",
        "Latest Low", "Open 1d (%)", "Open 1w (%)", "Close 1d (%)", "Close 1w (%)",
        "High 1d (%)", "High 1w (%)", "Low 1d (%)", "Low 1w (%)"
    ]
   
    for col in columns:
        if col not in stock_data.columns:
            stock_data[col] = pd.NA if col not in ["Ticker", "Company Name"] else "N/A"
   
    stock_data = stock_data.sort_values("Latest Close", ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce'))
    for col in columns:
        if col not in ["Ticker", "Company Name"]:
            stock_data[col] = np.where(
                stock_data[col].notna() & (pd.to_numeric(stock_data[col], errors='coerce') == pd.to_numeric(stock_data[col], errors='coerce')),
                pd.to_numeric(stock_data[col], errors='coerce').round(2),
                pd.NA
            )
       
        color_col = f"{col}_Color"
        stock_data[color_col] = "#FFFFFF"
        if col in ["Open 1d (%)", "Open 1w (%)", "Close 1d (%)", "Close 1w (%)",
                   "High 1d (%)", "High 1w (%)", "Low 1d (%)", "Low 1w (%)"]:
            stock_data[color_col] = np.where(
                pd.to_numeric(stock_data[col], errors='coerce').notna(),
                np.where(pd.to_numeric(stock_data[col], errors='coerce') < 0, "#F87171",
                         np.where(pd.to_numeric(stock_data[col], errors='coerce') > 0, "#10B981", "#FFFFFF")),
                "#FFFFFF"
            )
   
    color_columns = [f"{col}_Color" for col in columns if col not in ["Ticker", "Company Name"]]
    stock_data_no_colors = stock_data[columns]
    print(f"Generated stock table in {time.time() - start_time:.2f} seconds")
    return stock_data[columns + color_columns], stock_data_no_colors

def generate_summary_table(ranking, skew_data, tickers):
    """Generate aggregated summary table for all tickers."""
    start_time = time.time()
    if ranking is None or ranking.empty:
        print(f"No summary data in {time.time() - start_time:.2f} seconds")
        return pd.DataFrame(), pd.DataFrame()
   
    total_tickers = len(tickers)
    volume_rank = ranking['Volume'].rank(ascending=False, na_option="bottom").astype(int)
    open_interest_rank = ranking['Open Interest'].rank(ascending=False, na_option="bottom").astype(int)
    rank_map = {ticker: {'Volume Rank': f"{int(v)} of {total_tickers}", 'Open Interest Rank': f"{int(o)} of {total_tickers}"}
                for ticker, v, o in zip(ranking['Ticker'], volume_rank, open_interest_rank)}
    
    metrics = [
        {"name": "Latest Close ($)", "key": "Latest Close"},
        {"name": "Open ($)", "key": "Latest Open"},
        {"name": "Low ($)", "key": "Latest Low"},
        {"name": "High ($)", "key": "Latest High"},
        {"name": "Daily Volume", "key": "Volume"},
        {"name": "Close 1d (%)", "key": "Close 1d (%)"},
        {"name": "Close 1w (%)", "key": "Close 1w (%)"},
        {"name": "Realised Volatility 100d (%)", "key": "Realised Volatility 100d (%)"},
        {"name": "Weighted IV (%)", "key": "Weighted IV (%)"},
        {"name": "Weighted IV 1d (%)", "key": "Weighted IV 1d (%)"},
        {"name": "Weighted IV 1w (%)", "key": "Weighted IV 1w (%)"},
        {"name": "Volume 1d (%)", "key": "Volume 1d (%)"},
        {"name": "Volume 1w (%)", "key": "Volume 1w (%)"},
        {"name": "Open Interest", "key": "Open Interest"},
        {"name": "OI 1d (%)", "key": "OI 1d (%)"},
        {"name": "OI 1w (%)", "key": "OI 1w (%)"},
        {"name": "ATM 12m/3m Ratio", "key": "ATM_12m_3m_Ratio"},
        {"name": "Volume Rank", "key": "Volume Rank"},
        {"name": "Open Interest Rank", "key": "Open Interest Rank"}
    ]
    
    summary_data = []
    for ticker in tickers:
        filtered_ranking = ranking[ranking["Ticker"] == ticker]
        if filtered_ranking.empty:
            continue
        row = {"Ticker": ticker}
        for metric in metrics:
            if metric["key"] in ["Volume Rank", "Open Interest Rank"]:
                value = rank_map.get(ticker, {}).get(metric["key"], "N/A")
            else:
                value = filtered_ranking[metric["key"]].iloc[0] if metric["key"] in filtered_ranking.columns and not filtered_ranking[metric["key"]].empty else pd.NA
                if metric["key"] in ["Volume", "Open Interest"]:
                    num_val = pd.to_numeric(value, errors='coerce')
                    value = f"{int(num_val):,}" if pd.notna(num_val) and num_val > 0 else "N/A"
                else:
                    num_val = pd.to_numeric(value, errors='coerce')
                    value = round(num_val, 2) if pd.notna(num_val) else pd.NA
            row[metric["name"]] = value
       
        # Add ATM ratio from skew data
        atm_ratio = pd.NA
        if not skew_data.empty and skew_data["Ticker"].isin([ticker]).any():
            atm_val = skew_data[skew_data["Ticker"] == ticker]["ATM_12m_3m_Ratio"].iloc[0]
            atm_ratio = round(pd.to_numeric(atm_val, errors='coerce'), 4) if pd.notna(atm_val) else pd.NA
        row["ATM 12m/3m Ratio"] = atm_ratio
       
        # Add colors for percentage columns
        for metric in metrics:
            color_key = f"{metric['name']}_Color"
            row[color_key] = "#FFFFFF"
            if metric["key"] in ["Close 1d (%)", "Close 1w (%)", "Weighted IV 1d (%)",
                                "Weighted IV 1w (%)", "Volume 1d (%)", "Volume 1w (%)",
                                "OI 1d (%)", "OI 1w (%)"]:
                num_val = pd.to_numeric(row[metric["name"]], errors='coerce')
                if pd.notna(num_val):
                    row[color_key] = "#F87171" if num_val < 0 else "#10B981" if num_val > 0 else "#FFFFFF"
       
        summary_data.append(row)
    
    columns = ["Ticker"] + [metric["name"] for metric in metrics]
    color_columns = [f"{metric['name']}_Color" for metric in metrics if metric["key"] in ["Close 1d (%)", "Close 1w (%)", "Weighted IV 1d (%)",
                                                                                      "Weighted IV 1w (%)", "Volume 1d (%)", "Volume 1w (%)",
                                                                                      "OI 1d (%)", "OI 1w (%)"]]
    result = pd.DataFrame(summary_data, columns=columns + color_columns)
    result_no_colors = pd.DataFrame(summary_data, columns=columns)
    print(f"Generated summary table in {time.time() - start_time:.2f} seconds")
    return result, result_no_colors

def generate_top_contracts_tables(processed_data, tickers):
    """Generate aggregated top 10 volume and open interest tables for each ticker in a single file."""
    start_time = time.time()
    if processed_data.empty or 'Ticker' not in processed_data.columns:
        print(f"No contracts data in {time.time() - start_time:.2f} seconds")
        return pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"])
   
    top_volume_list = []
    top_open_interest_list = []
   
    for ticker in tickers:
        filtered = processed_data[processed_data["Ticker"] == ticker]
        if filtered.empty:
            continue
        top_volume = filtered[filtered["Volume"].notna()].sort_values("Volume", ascending=False).head(10)
        top_open_interest = filtered[filtered["Open Interest"].notna()].sort_values("Open Interest", ascending=False).head(10)
        if not top_volume.empty:
            top_volume_list.append(top_volume)
        if not top_open_interest.empty:
            top_open_interest_list.append(top_open_interest)
   
    top_volume_table = pd.concat(top_volume_list, ignore_index=True) if top_volume_list else pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"])
    top_open_interest_table = pd.concat(top_open_interest_list, ignore_index=True) if top_open_interest_list else pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"])
   
    def format_table(df):
        if df.empty:
            return pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"])
        required_cols = ["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]
        df = df[required_cols].copy() if all(col in df.columns for col in required_cols) else pd.DataFrame(columns=required_cols)
        df["Strike"] = np.where(df["Strike"].notna(), pd.to_numeric(df["Strike"], errors='coerce').round(2), pd.NA)
        df["Expiry"] = np.where(df["Expiry"].notna(), pd.to_datetime(df["Expiry"]).dt.strftime("%d/%m/%Y"), "N/A")
        df["Type"] = df["Type"].fillna("N/A")
        df["Bid"] = np.where(df["Bid"].notna(), pd.to_numeric(df["Bid"], errors='coerce').round(2), pd.NA)
        df["Ask"] = np.where(df["Ask"].notna(), pd.to_numeric(df["Ask"], errors='coerce').round(2), pd.NA)
        df["Volume"] = np.where(df["Volume"].notna() & pd.to_numeric(df["Volume"], errors='coerce').notna(),
                               pd.to_numeric(df["Volume"], errors='coerce').map(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "N/A"),
                               "N/A")
        df["Open Interest"] = np.where(df["Open Interest"].notna() & pd.to_numeric(df["Open Interest"], errors='coerce').notna(),
                                     pd.to_numeric(df["Open Interest"], errors='coerce').map(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "N/A"),
                                     "N/A")
        return df
   
    top_volume_table = format_table(top_volume_table)
    top_open_interest_table = format_table(top_open_interest_table)
   
    print(f"Generated contracts tables in {time.time() - start_time:.2f} seconds")
    return top_volume_table, top_open_interest_table, top_volume_table, top_open_interest_table

def save_tables(timestamp, source, base_path="data"):
    """Generate and save all precomputed tables."""
    start_time = time.time()
    prefix = "_yfinance" if source == "yfinance" else ""
    company_names, ranking, historic, events = load_data(timestamp, source, base_path)
    if ranking is None or ranking.empty:
        print(f"Failed to load required data files (ranking is missing or empty). Skipping table generation in {time.time() - start_time:.2f} seconds")
        return
   
    # Load all ticker-specific data
    tickers = ranking["Ticker"].unique()
    processed_data = pd.DataFrame()
    cleaned_data = pd.DataFrame()
    skew_data = pd.DataFrame()
    for ticker in tqdm(tickers, desc="Loading ticker data"):
        _, processed, cleaned, skew = load_ticker_data(ticker, timestamp, source, base_path)
        if not processed.empty:
            processed_data = pd.concat([processed_data, processed], ignore_index=True)
        if not cleaned.empty:
            cleaned_data = pd.concat([cleaned_data, cleaned], ignore_index=True)
        if not skew.empty:
            skew_data = pd.concat([skew_data, skew], ignore_index=True)
   
    # Generate tables
    ranking_table, ranking_table_no_colors = generate_ranking_table(ranking, company_names)
    stock_table, stock_table_no_colors = generate_stock_table(ranking, company_names)
    summary_table, summary_table_no_colors = generate_summary_table(ranking, skew_data, tickers)
    top_volume, top_open_interest, top_volume_no_colors, top_open_interest_no_colors = generate_top_contracts_tables(processed_data, tickers)
   
    # Create directories
    os.makedirs(f"{base_path}/{timestamp}/tables/ranking", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/stock", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/summary", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/contracts", exist_ok=True)
   
    # Save tables
    if not ranking_table.empty:
        ranking_table.to_csv(f"{base_path}/{timestamp}/tables/ranking/ranking_table{prefix}.csv", index=False)
        ranking_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/ranking/ranking_table_no_colors{prefix}.csv", index=False)
    if not stock_table.empty:
        stock_table.to_csv(f"{base_path}/{timestamp}/tables/stock/stock_table{prefix}.csv", index=False)
        stock_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/stock/stock_table_no_colors{prefix}.csv", index=False)
    if not summary_table.empty:
        summary_table.to_csv(f"{base_path}/{timestamp}/tables/summary/summary_table{prefix}.csv", index=False)
        summary_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/summary/summary_table_no_colors{prefix}.csv", index=False)
    if not top_volume.empty:
        top_volume.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_volume_table{prefix}.csv", index=False)
        top_volume_no_colors.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_volume_table_no_colors{prefix}.csv", index=False)
    if not top_open_interest.empty:
        top_open_interest.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_open_interest_table{prefix}.csv", index=False)
        top_open_interest_no_colors.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_open_interest_table_no_colors{prefix}.csv", index=False)
   
    print(f"Precomputed tables generation completed for {timestamp}, source {source} in {time.time() - start_time:.2f} seconds")

def main():
    start_time = time.time()
    base_path = 'data'
    dates_file = os.path.join(base_path, 'dates.json')
    try:
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    except Exception as e:
        print(f"Error loading dates.json: {e}")
        return
   
    timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
        if timestamp not in timestamps:
            print(f"Provided timestamp {timestamp} not found in dates.json")
            return
    else:
        timestamp = timestamps[-1]
        print(f"No timestamp provided, using latest: {timestamp}")
   
    sources = ['yfinance']
    for source in sources:
        save_tables(timestamp, source, base_path)
   
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

main()
