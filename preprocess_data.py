import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import time
import pyxlsb
from tqdm import tqdm

def load_data(timestamp, source, base_path="data"):
    """Load raw data files for a given timestamp and source."""
    start_time = time.time()
    prefix = "_yfinance" if source == "yfinance" else ""
    company_names = None
    ranking = None
    barclays = None
    try:
        # Load company names
        if os.path.exists("company_names.txt"):
            print("Loading company_names.txt...", flush=True)
            company_names = pd.read_csv("company_names.txt", sep="\t")
            print(f"Loaded company_names.txt in {time.time() - start_time:.2f} seconds", flush=True)
        else:
            print("Warning: company_names.txt not found.", flush=True)
        # Load ranking
        ranking_path = f"{base_path}/{timestamp}/ranking/ranking{prefix}.csv"
        if os.path.exists(ranking_path):
            print(f"Loading ranking file: {ranking_path}...", flush=True)
            ranking = pd.read_csv(ranking_path)
            numeric_cols = ['Latest Close', 'Realised Volatility 30d (%)', 'Realised Volatility 100d (%)',
                           'Realised Volatility 100d 1d (%)', 'Realised Volatility 100d 1w (%)',
                           'Min Realised Volatility 100d (2y)', 'Max Realised Volatility 100d (2y)',
                           'Mean Realised Volatility 100d (2y)', 'Rvol 100d Percentile 2y (%)',
                           'Rvol 100d Z-Score Percentile 2y (%)', 'Realised Volatility 180d (%)',
                           'Realised Volatility 252d (%)', 'Weighted IV (%)', 'Weighted IV 1d (%)',
                           'Weighted IV 1w (%)', 'Weighted IV 3m (%)', 'Weighted IV 3m 1d (%)',
                           'Weighted IV 3m 1w (%)', 'ATM IV 3m (%)', 'ATM IV 3m 1d (%)',
                           'ATM IV 3m 1w (%)', 'Rvol100d - Weighted IV', 'Volume', 'Volume 1d (%)',
                           'Volume 1w (%)', 'Open Interest', 'OI 1d (%)', 'OI 1w (%)', 'Num Contracts',
                           'Normalized 3m', 'Normalized 6m', 'Normalized 1y']
            for col in numeric_cols:
                if col in ranking.columns:
                    ranking[col] = pd.to_numeric(ranking[col], errors='coerce')
            print(f"Loaded ranking file in {time.time() - start_time:.2f} seconds", flush=True)
        else:
            raise FileNotFoundError(f"Ranking file not found: {ranking_path}")
        # Load Barclays data
        barclays_path = f"{base_path}/BASE3-Credit & Equity Volatility Term Structures.xlsb"
        if os.path.exists(barclays_path):
            print(f"Loading Barclays data: {barclays_path}...", flush=True)
            barclays = pd.read_excel(barclays_path, engine='pyxlsb')
            required_cols = ['Ticker', 'Company Name', 'Debt Class', 'Spread 1Y', 'Spread 3Y', 'Spread 5Y']
            if all(col in barclays.columns for col in required_cols):
                barclays = barclays[required_cols]
                for col in ['Spread 1Y', 'Spread 3Y', 'Spread 5Y']:
                    barclays[col] = pd.to_numeric(barclays[col], errors='coerce')
            else:
                print(f"Warning: Barclays data missing required columns: {required_cols}", flush=True)
                barclays = None
            print(f"Loaded Barclays data in {time.time() - start_time:.2f} seconds", flush=True)
        else:
            print(f"Warning: Barclays data file not found: {barclays_path}", flush=True)
        print(f"Loaded all data in {time.time() - start_time:.2f} seconds", flush=True)
        return company_names, ranking, barclays
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}", flush=True)
        return None, None, None

def load_ticker_data(ticker, timestamp, source, base_path="data"):
    """Load ticker-specific data."""
    start_time = time.time()
    prefix = "_yfinance" if source == "yfinance" else ""
    processed = pd.DataFrame()
    skew = pd.DataFrame()
    try:
        processed_path = f"{base_path}/{timestamp}/processed{prefix}/processed{prefix}_{ticker}.csv"
        if os.path.exists(processed_path):
            processed = pd.read_csv(processed_path)
        skew_path = f"{base_path}/{timestamp}/skew_metrics{prefix}/skew_metrics{prefix}_{ticker}.csv"
        if os.path.exists(skew_path):
            skew = pd.read_csv(skew_path)
        print(f"Loaded data for ticker {ticker} in {time.time() - start_time:.2f} seconds")
        return ticker, processed, skew
    except Exception as e:
        print(f"Error loading data for ticker {ticker}: {e}")
        return ticker, pd.DataFrame(), pd.DataFrame()

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
        "Realised Volatility 100d 1w (%)", "Min Realised Volatility 100d (2y)",
        "Max Realised Volatility 100d (2y)", "Mean Realised Volatility 100d (2y)",
        "Rvol 100d Percentile 2y (%)", "Rvol 100d Z-Score Percentile 2y (%)",
        "Realised Volatility 180d (%)", "Realised Volatility 252d (%)",
        "Weighted IV (%)", "Weighted IV 1d (%)", "Weighted IV 1w (%)",
        "Weighted IV 3m (%)", "Weighted IV 3m 1d (%)", "Weighted IV 3m 1w (%)",
        "ATM IV 3m (%)", "ATM IV 3m 1d (%)", "ATM IV 3m 1w (%)",
        "Rvol100d - Weighted IV", "Volume", "Volume 1d (%)", "Volume 1w (%)",
        "Open Interest", "OI 1d (%)", "OI 1w (%)", "Num Contracts"
    ]
    for col in columns:
        if col not in ranking.columns:
            ranking[col] = pd.NA if col not in ["Rank", "Ticker", "Company Name"] else "N/A"
    ranking['Open Interest Numeric'] = pd.to_numeric(ranking['Open Interest'], errors='coerce').fillna(0)
    ranking["Rank"] = ranking['Open Interest Numeric'].rank(ascending=False, na_option="bottom").astype(int)
    ranking = ranking.drop('Open Interest Numeric', axis=1)
    for col in columns:
        if col in ["Volume", "Open Interest", "Num Contracts"]:
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

def generate_stock_table(ranking, company_names, barclays):
    """Generate stock table with formatted values, colors, and Barclays data."""
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
    # Merge Barclays data
    if barclays is not None and not barclays.empty:
        stock_data = stock_data.merge(
            barclays[['Ticker', 'Debt Class', 'Spread 1Y', 'Spread 3Y', 'Spread 5Y']],
            on='Ticker',
            how='left'
        )
    columns = [
        "Ticker", "Company Name", "Latest Open", "Latest Close", "Latest High",
        "Latest Low", "Open 1d (%)", "Open 1w (%)", "Close 1d (%)", "Close 1w (%)",
        "High 1d (%)", "High 1w (%)", "Low 1d (%)", "Low 1w (%)",
        "Debt Class", "Spread 1Y", "Spread 3Y", "Spread 5Y"
    ]
    for col in columns:
        if col not in stock_data.columns:
            stock_data[col] = pd.NA if col not in ["Ticker", "Company Name", "Debt Class"] else "N/A"
    stock_data = stock_data.sort_values("Latest Close", ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce'))
    for col in columns:
        if col not in ["Ticker", "Company Name", "Debt Class"]:
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
    color_columns = [f"{col}_Color" for col in columns if col not in ["Ticker", "Company Name", "Debt Class"]]
    stock_data_no_colors = stock_data[columns]
    print(f"Generated stock table in {time.time() - start_time:.2f} seconds")
    return stock_data[columns + color_columns], stock_data_no_colors

def generate_normalized_table(ranking):
    """Generate normalized price table with formatted values."""
    start_time = time.time()
    if ranking is None or ranking.empty:
        print(f"No normalized data in {time.time() - start_time:.2f} seconds")
        return pd.DataFrame(), pd.DataFrame()
    normalized_data = ranking.copy()
    columns = ["Ticker", "Normalized 3m", "Normalized 6m", "Normalized 1y"]
    for col in columns:
        if col not in normalized_data.columns:
            normalized_data[col] = pd.NA if col != "Ticker" else "N/A"
    for col in columns[1:]: # Skip Ticker
        normalized_data[col] = np.where(
            normalized_data[col].notna() & (pd.to_numeric(normalized_data[col], errors='coerce') == pd.to_numeric(normalized_data[col], errors='coerce')),
            pd.to_numeric(normalized_data[col], errors='coerce').round(2),
            pd.NA
        )
        color_col = f"{col}_Color"
        normalized_data[color_col] = "#FFFFFF"
    color_columns = [f"{col}_Color" for col in columns if col != "Ticker"]
    normalized_data_no_colors = normalized_data[columns]
    print(f"Generated normalized table in {time.time() - start_time:.2f} seconds")
    return normalized_data[columns + color_columns], normalized_data_no_colors

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

def generate_top_contracts_tables(processed_data, tickers, timestamp):
    """Generate aggregated top 10 volume and open interest tables for each ticker in a single file."""
    start_time = time.time()
    if processed_data.empty or 'Ticker' not in processed_data.columns:
        print(f"No contracts data in {time.time() - start_time:.2f} seconds")
        return pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"])
    timestamp_dt = datetime.strptime(timestamp, "%Y%m%d_%H%M")
    min_expiry_dt = (timestamp_dt + pd.DateOffset(months=3)).date()
    top_volume_list = []
    top_open_interest_list = []
    for ticker in tickers:
        filtered = processed_data[processed_data["Ticker"] == ticker]
        if filtered.empty:
            continue
        filtered['Expiry_dt'] = pd.to_datetime(filtered['Expiry'], errors='coerce').dt.date
        long_term = filtered[filtered['Expiry_dt'] >= min_expiry_dt].drop(columns=['Expiry_dt'])
        top_volume = long_term[long_term["Volume"].notna()].sort_values("Volume", ascending=False).head(10)
        top_open_interest = long_term[long_term["Open Interest"].notna()].sort_values("Open Interest", ascending=False).head(10)
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
    """Generate and save all precomputed tables, including normalized table, and append IVOLs to historical data."""
    start_time = time.time()
    prefix = "_yfinance" if source == "yfinance" else ""
    company_names, ranking, barclays = load_data(timestamp, source, base_path)
    if ranking is None or ranking.empty:
        print(f"Failed to load required data files (ranking is missing or empty). Skipping table generation in {time.time() - start_time:.2f} seconds")
        return
    # Load all ticker-specific data
    tickers = ranking["Ticker"].unique()
    processed_data = pd.DataFrame()
    skew_data = pd.DataFrame()
    for ticker in tqdm(tickers, desc="Loading ticker data"):
        _, processed, skew = load_ticker_data(ticker, timestamp, source, base_path)
        if not processed.empty:
            processed_data = pd.concat([processed_data, processed], ignore_index=True)
        if not skew.empty:
            skew_data = pd.concat([skew_data, skew], ignore_index=True)
    # Generate tables
    ranking_table, ranking_table_no_colors = generate_ranking_table(ranking, company_names)
    stock_table, stock_table_no_colors = generate_stock_table(ranking, company_names, barclays)
    normalized_table, normalized_table_no_colors = generate_normalized_table(ranking)
    summary_table, summary_table_no_colors = generate_summary_table(ranking, skew_data, tickers)
    top_volume, top_open_interest, top_volume_no_colors, top_open_interest_no_colors = generate_top_contracts_tables(processed_data, tickers, timestamp)
    # Create directories
    os.makedirs(f"{base_path}/{timestamp}/tables/ranking", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/stock", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/normalized", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/summary", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/contracts", exist_ok=True)
    # Save tables
    if not ranking_table.empty:
        ranking_table.to_csv(f"{base_path}/{timestamp}/tables/ranking/ranking_table{prefix}.csv", index=False)
        ranking_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/ranking/ranking_table.csv", index=False)
    if not stock_table.empty:
        stock_table.to_csv(f"{base_path}/{timestamp}/tables/stock/stock_table{prefix}.csv", index=False)
        stock_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/stock/stock_table.csv", index=False)
    if not normalized_table.empty:
        normalized_table.to_csv(f"{base_path}/{timestamp}/tables/normalized/normalized_table{prefix}.csv", index=False)
        normalized_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/normalized/normalized_table.csv", index=False)
    if not summary_table.empty:
        summary_table.to_csv(f"{base_path}/{timestamp}/tables/summary/summary_table{prefix}.csv", index=False)
        summary_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/summary/summary_table.csv", index=False)
    if not top_volume.empty:
        top_volume.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_volume_table{prefix}.csv", index=False)
        top_volume_no_colors.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_volume_table.csv", index=False)
    if not top_open_interest.empty:
        top_open_interest.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_open_interest_table{prefix}.csv", index=False)
        top_open_interest_no_colors.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_open_interest_table.csv", index=False)
    # Append IVOLs to historical data
    ivol_columns = [
        'Weighted IV (%)', 'Weighted IV 3m (%)', 'ATM IV 3m (%)'
    ]
    historical_columns = ['Timestamp', 'Ticker'] + ivol_columns
    for ticker in tickers:
        ticker_data = ranking[ranking['Ticker'] == ticker]
        if ticker_data.empty:
            continue
        row = {'Timestamp': timestamp, 'Ticker': ticker}
        for col in ivol_columns:
            value = ticker_data[col].iloc[0] if col in ticker_data.columns and not ticker_data[col].empty else pd.NA
            row[col] = round(pd.to_numeric(value, errors='coerce'), 2) if pd.notna(value) else pd.NA
        historical_df = pd.DataFrame([row], columns=historical_columns)
        historical_file = f"{base_path}/history/historic_{ticker}.csv"
        os.makedirs(os.path.dirname(historical_file), exist_ok=True)
        if os.path.exists(historical_file):
            # Append to existing file without header
            historical_df.to_csv(historical_file, mode='a', header=False, index=False)
        else:
            # Create new file with header
            historical_df.to_csv(historical_file, mode='w', header=True, index=False)
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
