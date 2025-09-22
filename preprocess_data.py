import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import time
import pyxlsb
from tqdm import tqdm
import yfinance as yf  # For fetching SPX data

def load_data(timestamp, source, base_path="data"):
    """Load raw data files for a given timestamp and source."""
    start_time = time.time()
    prefix = "_yfinance" if source == "yfinance" else ""
    company_names = None
    ranking = None
    barclays = None
    try:
        if os.path.exists("company_names.txt"):
            print("Loading company_names.txt...", flush=True)
            company_names = pd.read_csv("company_names.txt", sep="\t")
            print(f"Loaded company_names.txt in {time.time() - start_time:.2f} seconds", flush=True)
        else:
            print("Warning: company_names.txt not found.", flush=True)
        
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
            print(f"Warning: Ranking file not found: {ranking_path}. Returning empty DataFrame.", flush=True)
            ranking = pd.DataFrame()

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
    except Exception as e:
        print(f"Error loading data: {e}", flush=True)
        return company_names, ranking, barclays

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
        else:
            print(f"Warning: Processed data file not found for {ticker}: {processed_path}", flush=True)
        skew_path = f"{base_path}/{timestamp}/skew_metrics{prefix}/skew_metrics{prefix}_{ticker}.csv"
        if os.path.exists(skew_path):
            skew = pd.read_csv(skew_path)
        else:
            print(f"Warning: Skew metrics file not found for {ticker}: {skew_path}", flush=True)
        print(f"Loaded data for ticker {ticker} in {time.time() - start_time:.2f} seconds", flush=True)
        return ticker, processed, skew
    except Exception as e:
        print(f"Error loading data for ticker {ticker}: {e}", flush=True)
        return ticker, pd.DataFrame(), pd.DataFrame()

def generate_ranking_table(ranking, company_names):
    """Generate ranking table with formatted values and colors."""
    start_time = time.time()
    if ranking is None or ranking.empty:
        print(f"No ranking data in {time.time() - start_time:.2f} seconds", flush=True)
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
        "Open Interest", "OI 1d (%)", "OI 1w (%)", "Num Contracts",
        "One_Yr_ATM_Rel_Error_Call (%)", "P90_Rel_Error_Call (%)", "Restricted_P90_Rel_Error_Call (%)",
        "One_Yr_ATM_Rel_Error_Put (%)", "P90_Rel_Error_Put (%)", "Restricted_P90_Rel_Error_Put (%)"
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
                   "OI 1d (%)", "OI 1w (%)", "One_Yr_ATM_Rel_Error_Call (%)",
                   "P90_Rel_Error_Call (%)", "Restricted_P90_Rel_Error_Call (%)",
                   "One_Yr_ATM_Rel_Error_Put (%)", "P90_Rel_Error_Put (%)",
                   "Restricted_P90_Rel_Error_Put (%)"]:
            ranking[color_col] = np.where(
                pd.to_numeric(ranking[col], errors='coerce').notna(),
                np.where(pd.to_numeric(ranking[col], errors='coerce') < 0, "#F87171",
                         np.where(pd.to_numeric(ranking[col], errors='coerce') > 0, "#10B981", "#FFFFFF")),
                "#FFFFFF"
            )
    color_columns = [f"{col}_Color" for col in columns if col not in ["Rank", "Ticker", "Company Name"]]
    ranking_no_colors = ranking[columns]
    print(f"Generated ranking table in {time.time() - start_time:.2f} seconds", flush=True)
    return ranking[columns + color_columns], ranking_no_colors

def generate_stock_table(ranking, company_names, barclays):
    """Generate stock table with formatted values, colors, and Barclays data."""
    start_time = time.time()
    if ranking is None or ranking.empty:
        print(f"No stock data in {time.time() - start_time:.2f} seconds", flush=True)
        return pd.DataFrame(), pd.DataFrame()
    stock_data = ranking.copy()
    if company_names is not None and not company_names.empty:
        stock_data["Company Name"] = stock_data["Ticker"].map(
            company_names.set_index("Ticker")["Company Name"].to_dict()
        ).fillna("N/A")
    else:
        stock_data["Company Name"] = "N/A"
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
    print(f"Generated stock table in {time.time() - start_time:.2f} seconds", flush=True)
    return stock_data[columns + color_columns], stock_data_no_colors

def generate_normalized_table(ranking):
    """Generate normalized price table with formatted values."""
    start_time = time.time()
    if ranking is None or ranking.empty:
        print(f"No normalized data in {time.time() - start_time:.2f} seconds", flush=True)
        return pd.DataFrame(), pd.DataFrame()
    normalized_data = ranking.copy()
    columns = ["Ticker", "Normalized 3m", "Normalized 6m", "Normalized 1y"]
    for col in columns:
        if col not in normalized_data.columns:
            normalized_data[col] = pd.NA if col != "Ticker" else "N/A"
    for col in columns[1:]:
        normalized_data[col] = np.where(
            normalized_data[col].notna() & (pd.to_numeric(normalized_data[col], errors='coerce') == pd.to_numeric(normalized_data[col], errors='coerce')),
            pd.to_numeric(normalized_data[col], errors='coerce').round(2),
            pd.NA
        )
        color_col = f"{col}_Color"
        normalized_data[color_col] = "#FFFFFF"
    color_columns = [f"{col}_Color" for col in columns if col != "Ticker"]
    normalized_data_no_colors = normalized_data[columns]
    print(f"Generated normalized table in {time.time() - start_time:.2f} seconds", flush=True)
    return normalized_data[columns + color_columns], normalized_data_no_colors

def generate_summary_table(ranking, skew_data, tickers):
    """Generate aggregated summary table for all tickers."""
    start_time = time.time()
    if ranking is None or ranking.empty:
        print(f"No summary data in {time.time() - start_time:.2f} seconds", flush=True)
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
            print(f"Warning: No ranking data for ticker {ticker}", flush=True)
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
        atm_ratio = pd.NA
        if not skew_data.empty and skew_data["Ticker"].isin([ticker]).any():
            atm_val = skew_data[skew_data["Ticker"] == ticker]["ATM_12m_3m_Ratio"].iloc[0]
            atm_ratio = round(pd.to_numeric(atm_val, errors='coerce'), 4) if pd.notna(atm_val) else pd.NA
        row["ATM 12m/3m Ratio"] = atm_ratio
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
    print(f"Generated summary table in {time.time() - start_time:.2f} seconds", flush=True)
    return result, result_no_colors

def generate_top_contracts_tables(processed_data, tickers, timestamp):
    """Generate aggregated top 10 volume and open interest tables for each ticker in a single file."""
    start_time = time.time()
    if processed_data.empty or 'Ticker' not in processed_data.columns:
        print(f"No contracts data in {time.time() - start_time:.2f} seconds", flush=True)
        return pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Ticker", "Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"])
    timestamp_dt = datetime.strptime(timestamp, "%Y%m%d_%H%M")
    min_expiry_dt = (timestamp_dt + pd.DateOffset(months=3)).date()
    top_volume_list = []
    top_open_interest_list = []
    for ticker in tickers:
        filtered = processed_data[processed_data["Ticker"] == ticker].copy()  # Create a copy to avoid SettingWithCopyWarning
        if filtered.empty:
            print(f"Warning: No processed data for ticker {ticker}", flush=True)
            continue
        filtered.loc[:, 'Expiry_dt'] = pd.to_datetime(filtered['Expiry'], errors='coerce').dt.date
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
    print(f"Generated contracts tables in {time.time() - start_time:.2f} seconds", flush=True)
    return top_volume_table, top_open_interest_table, top_volume_table, top_open_interest_table

def generate_returns_summary(base_path="data"):
    """Generate summary table and correlation table from BH_HF_Ret_Sept_25.csv, adding SPX returns, Sharpe, and Sortino ratios."""
    start_time = time.time()
    file_path = os.path.join(base_path, "BH_HF_Ret_Sept_25.csv")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", flush=True)
        return
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        if 'Date' not in df.columns:
            print("Error: CSV must have a 'Date' column.", flush=True)
            return
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
        df.set_index('Date', inplace=True)
        # Convert percentage strings to decimals (e.g., "1.05%" to 0.0105)
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].str.rstrip('%').astype(float) / 100
        # Fetch SPX data
        min_date = df.index.min()
        max_date = df.index.max()
        try:
            spx = yf.download('^GSPC', start=min_date - pd.DateOffset(months=1), end=max_date + pd.DateOffset(months=1))
            spx_monthly = spx['Adj Close'].resample('ME').last().pct_change()
            spx_monthly.index = pd.to_datetime(spx_monthly.index)
            spx_monthly = spx_monthly.reindex(df.index, method='nearest')
            df['SPX'] = spx_monthly
        except Exception as e:
            print(f"Error fetching SPX data: {e}. Proceeding without SPX.", flush=True)
            df['SPX'] = np.nan
        # Fetch 3-month T-Bill yield (^IRX) as risk-free rate
        try:
            tbill = yf.download('^IRX', start=min_date - pd.DateOffset(months=1), end=max_date + pd.DateOffset(months=1))
            # ^IRX is annualized yield in percent; convert to monthly decimal
            rf_monthly = tbill['Close'] / 100 / 12
            rf_monthly = rf_monthly.resample('ME').last()
            rf_monthly = rf_monthly.reindex(df.index, method='nearest').fillna(method='ffill')
            df['RiskFree'] = rf_monthly
        except Exception as e:
            print(f"Error fetching T-Bill data: {e}. Using fallback risk-free rate of 4% annual.", flush=True)
            df['RiskFree'] = 0.04 / 12  # Fallback: 4% annual = 0.00333 monthly
        # Ensure numeric data
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        strategies = df.columns.drop('RiskFree')  # Exclude RiskFree from strategies
        summary = []
        for strat in strategies:
            rets = df[strat].dropna()
            rf = df['RiskFree'].reindex(rets.index).dropna()
            if len(rets) == 0 or len(rf) == 0:
                print(f"Warning: No valid returns data for strategy {strat}", flush=True)
                continue
            # Excess returns
            excess_rets = rets - rf
            # Basic statistics
            min_r = round(rets.min(), 4)
            max_r = round(rets.max(), 4)
            mean_r = round(rets.mean(), 4)
            # Positive and negative months
            pos = rets[rets > 0]
            neg = rets[rets < 0]
            pos_neg_ratio = round(len(pos) / len(neg), 2) if len(neg) > 0 else np.inf
            mean_pos = round(pos.mean(), 4) if len(pos) > 0 else np.nan
            mean_neg = round(neg.mean(), 4) if len(neg) > 0 else np.nan
            # Drawdown calculations
            cumret = (1 + rets).cumprod()
            cummax = cumret.cummax()
            dd = (cumret / cummax) - 1
            max_dd = round(dd.min(), 4)
            trough_idx = dd.idxmin()
            peak_idx = cummax.loc[:trough_idx].idxmax()
            dd_length = (trough_idx - peak_idx).days // 30  # Approximate months
            # Recovery time
            peak_val = cummax.loc[peak_idx]
            post_trough = cumret.loc[trough_idx:]
            recovery_mask = post_trough >= peak_val
            recovery_time = np.nan
            if recovery_mask.any():
                recovery_idx = post_trough[recovery_mask].index[0]
                recovery_time = (recovery_idx - trough_idx).days // 30  # Approximate months
            # Sharpe Ratio: (Mean excess return / Std dev of excess returns) * sqrt(12)
            mean_excess = excess_rets.mean()
            std_excess = excess_rets.std()
            sharpe_ratio = round((mean_excess / std_excess) * np.sqrt(12), 4) if std_excess > 0 else np.nan
            # Sortino Ratio: (Mean excess return / Downside deviation) * sqrt(12)
            downside_rets = excess_rets[excess_rets < 0]
            downside_std = downside_rets.std() if len(downside_rets) > 0 else np.nan
            sortino_ratio = round((mean_excess / downside_std) * np.sqrt(12), 4) if downside_std > 0 else np.nan
            summary.append({
                'Strategy': strat,
                'Min Return': min_r,
                'Max Return': max_r,
                'Mean Return': mean_r,
                'Max Drawdown': max_dd,
                'Drawdown Length (months)': dd_length,
                'Recovery Time (months)': recovery_time,
                'Positive/Negative Ratio': pos_neg_ratio if np.isfinite(pos_neg_ratio) else 'inf',
                'Mean Positive Return': mean_pos,
                'Mean Negative Return': mean_neg,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio
            })
        summary_df = pd.DataFrame(summary)
        # Save summary table
        os.makedirs(os.path.join(base_path, "tables/returns"), exist_ok=True)
        summary_df.to_csv(os.path.join(base_path, "tables/returns/summary_table.csv"), index=False)
        # Correlation table
        corr = df[strategies].corr().round(4)  # Exclude RiskFree from correlation
        corr.to_csv(os.path.join(base_path, "tables/returns/correlation_table.csv"))
        print(f"Generated returns summary and correlation tables in {time.time() - start_time:.2f} seconds", flush=True)
    except Exception as e:
        print(f"Error in generate_returns_summary: {e}", flush=True)

def save_tables(timestamp, source, base_path="data"):
    """Generate and save all precomputed tables, including normalized table, and append IVOLs to historical data."""
    start_time = time.time()
    prefix = "_yfinance" if source == "yfinance" else ""
    company_names, ranking, barclays = load_data(timestamp, source, base_path)
    if ranking is None or ranking.empty:
        print(f"Warning: No ranking data available for {timestamp}. Generating empty tables.", flush=True)
        tickers = []
    else:
        tickers = ranking["Ticker"].unique()
    processed_data = pd.DataFrame()
    skew_data = pd.DataFrame()
    for ticker in tqdm(tickers, desc="Loading ticker data"):
        _, processed, skew = load_ticker_data(ticker, timestamp, source, base_path)
        if not processed.empty:
            processed_data = pd.concat([processed_data, processed], ignore_index=True)
        if not skew.empty:
            skew_data = pd.concat([skew_data, skew], ignore_index=True)
    ranking_table, ranking_table_no_colors = generate_ranking_table(ranking, company_names)
    stock_table, stock_table_no_colors = generate_stock_table(ranking, company_names, barclays)
    normalized_table, normalized_table_no_colors = generate_normalized_table(ranking)
    summary_table, summary_table_no_colors = generate_summary_table(ranking, skew_data, tickers)
    top_volume, top_open_interest, top_volume_no_colors, top_open_interest_no_colors = generate_top_contracts_tables(processed_data, tickers, timestamp)
    os.makedirs(f"{base_path}/{timestamp}/tables/ranking", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/stock", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/normalized", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/summary", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/contracts", exist_ok=True)
    if not ranking_table.empty:
        ranking_table.to_csv(f"{base_path}/{timestamp}/tables/ranking/ranking_table{prefix}.csv", index=False)
        ranking_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/ranking/ranking_table.csv", index=False)
        print(f"Saved ranking tables to {base_path}/{timestamp}/tables/ranking/", flush=True)
    else:
        print(f"Warning: Ranking table is empty for {timestamp}", flush=True)
    if not stock_table.empty:
        stock_table.to_csv(f"{base_path}/{timestamp}/tables/stock/stock_table{prefix}.csv", index=False)
        stock_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/stock/stock_table.csv", index=False)
        print(f"Saved stock tables to {base_path}/{timestamp}/tables/stock/", flush=True)
    else:
        print(f"Warning: Stock table is empty for {timestamp}", flush=True)
    if not normalized_table.empty:
        normalized_table.to_csv(f"{base_path}/{timestamp}/tables/normalized/normalized_table{prefix}.csv", index=False)
        normalized_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/normalized/normalized_table.csv", index=False)
        print(f"Saved normalized tables to {base_path}/{timestamp}/tables/normalized/", flush=True)
    else:
        print(f"Warning: Normalized table is empty for {timestamp}", flush=True)
    if not summary_table.empty:
        summary_table.to_csv(f"{base_path}/{timestamp}/tables/summary/summary_table{prefix}.csv", index=False)
        summary_table_no_colors.to_csv(f"{base_path}/{timestamp}/tables/summary/summary_table.csv", index=False)
        print(f"Saved summary tables to {base_path}/{timestamp}/tables/summary/", flush=True)
    else:
        print(f"Warning: Summary table is empty for {timestamp}", flush=True)
    if not top_volume.empty:
        top_volume.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_volume_table{prefix}.csv", index=False)
        top_volume_no_colors.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_volume_table.csv", index=False)
        print(f"Saved top volume tables to {base_path}/{timestamp}/tables/contracts/", flush=True)
    else:
        print(f"Warning: Top volume table is empty for {timestamp}", flush=True)
    if not top_open_interest.empty:
        top_open_interest.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_open_interest_table{prefix}.csv", index=False)
        top_open_interest_no_colors.to_csv(f"{base_path}/{timestamp}/tables/contracts/top_open_interest_table.csv", index=False)
        print(f"Saved top open interest tables to {base_path}/{timestamp}/tables/contracts/", flush=True)
    else:
        print(f"Warning: Top open interest table is empty for {timestamp}", flush=True)
    ivol_columns = [
        'Weighted IV (%)', 'Weighted IV 3m (%)', 'ATM IV 3m (%)'
    ]
    historical_columns = ['Timestamp', 'Ticker'] + ivol_columns
    for ticker in tickers:
        ticker_data = ranking[ranking['Ticker'] == ticker] if ranking is not None and not ranking.empty else pd.DataFrame()
        if ticker_data.empty:
            print(f"Warning: No ranking data for ticker {ticker} when appending IVOLs", flush=True)
            continue
        row = {'Timestamp': timestamp, 'Ticker': ticker}
        for col in ivol_columns:
            value = ticker_data[col].iloc[0] if col in ticker_data.columns and not ticker_data[col].empty else pd.NA
            row[col] = round(pd.to_numeric(value, errors='coerce'), 2) if pd.notna(value) else pd.NA
        historical_df = pd.DataFrame([row], columns=historical_columns)
        historical_file = f"{base_path}/history/historic_{ticker}.csv"
        os.makedirs(os.path.dirname(historical_file), exist_ok=True)
        if os.path.exists(historical_file):
            historical_df.to_csv(historical_file, mode='a', header=False, index=False)
        else:
            historical_df.to_csv(historical_file, mode='w', header=True, index=False)
        print(f"Appended IVOLs for {ticker} to {historical_file}", flush=True)
    print(f"Precomputed tables generation completed for {timestamp}, source {source} in {time.time() - start_time:.2f} seconds", flush=True)

def main():
    start_time = time.time()
    base_path = 'data'
    dates_file = os.path.join(base_path, 'dates.json')
    try:
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    except Exception as e:
        print(f"Error loading dates.json: {e}", flush=True)
        return
    timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
        if timestamp not in timestamps:
            print(f"Provided timestamp {timestamp} not found in dates.json", flush=True)
            return
    else:
        timestamp = timestamps[-1]
        print(f"No timestamp provided, using latest: {timestamp}", flush=True)
    sources = ['yfinance']
    for source in sources:
        save_tables(timestamp, source, base_path)
    generate_returns_summary(base_path)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds", flush=True)

if __name__ == "__main__":
    main()
