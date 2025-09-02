import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
import sys
import json
import glob
from datetime import datetime

def load_data(timestamp, source, base_path="data"):
    """Load raw data files for a given timestamp and source."""
    prefix = "_yfinance" if source == "yfinance" else ""
    company_names = None
    ranking = None
    historic = pd.DataFrame()  # Default to empty
    events = pd.DataFrame()
    try:
        # Load company names (assuming it's in current directory)
        if os.path.exists("company_names.txt"):
            company_names = pd.read_csv("company_names.txt", sep="\t")
        else:
            print("Warning: company_names.txt not found in current directory.")

        # Load ranking
        ranking_path = f"{base_path}/{timestamp}/ranking/ranking{prefix}.csv"
        if os.path.exists(ranking_path):
            ranking = pd.read_csv(ranking_path)
        else:
            raise FileNotFoundError(f"Ranking file not found: {ranking_path}")

        # Load historic data using glob to find all historic_*.csv files and concatenate
        historic_dir = f"{base_path}/{timestamp}/historic"
        if os.path.exists(historic_dir):
            historic_files = glob.glob(f"{historic_dir}/historic_*.csv")
            if historic_files:
                dfs = []
                for file in historic_files:
                    try:
                        df = pd.read_csv(file)
                        # Ensure 'Ticker' column exists (extract from filename if not)
                        if 'Ticker' not in df.columns:
                            ticker = os.path.basename(file).split('historic_')[1].split('.csv')[0].upper()
                            df['Ticker'] = ticker
                        # Ensure 'Date' is datetime
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        # Convert numeric columns
                        numeric_cols = ['High', 'Low', 'Close', 'Realised_Vol_Close_30', 'Realised_Vol_Close_60',
                                        'Realised_Vol_Close_100', 'Realised_Vol_Close_180', 'Realised_Vol_Close_252']
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        dfs.append(df)
                    except Exception as e:
                        print(f"Warning: Error loading historic file {file}: {e}")
                if dfs:
                    historic = pd.concat(dfs, ignore_index=True)
                    historic = historic.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
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

        return company_names, ranking, historic, events
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        return None, None, None, None

def calculate_market_spread(cleaned_data, ticker):
    """Calculate market spread for a ticker."""
    if cleaned_data.empty or 'Ticker' not in cleaned_data.columns:
        return "N/A"
    stock_rows = cleaned_data[(cleaned_data["Ticker"] == ticker) &
                             cleaned_data["Bid Stock"].notna() &
                             cleaned_data["Ask Stock"].notna()]
    if stock_rows.empty:
        return "N/A"
    bid = stock_rows.iloc[0]["Bid Stock"]
    ask = stock_rows.iloc[0]["Ask Stock"]
    return round(ask - bid, 2) if pd.notna(bid) and pd.notna(ask) else "N/A"

def calculate_historical_spreads(historic_data, ticker):
    """Calculate historical spreads (High - Low) for 1y, 3y, 5y periods."""
    if historic_data.empty or 'Ticker' not in historic_data.columns:
        return {"1y Spread": "N/A", "3y Spread": "N/A", "5y Spread": "N/A"}
    
    filtered = historic_data[(historic_data["Ticker"] == ticker) &
                             historic_data["High"].notna() &
                             historic_data["Low"].notna()]
    if filtered.empty:
        return {"1y Spread": "N/A", "3y Spread": "N/A", "5y Spread": "N/A"}
    
    filtered = filtered.sort_values("Date")
    spreads = filtered["High"] - filtered["Low"]
    spreads = spreads[spreads.notna()]
    
    if spreads.empty:
        return {"1y Spread": "N/A", "3y Spread": "N/A", "5y Spread": "N/A"}
    
    recent_spreads = spreads[-1260:]  # Up to 5 years (1260 trading days)
    one_year = recent_spreads[-252:]  # 1 year
    three_year = recent_spreads[-756:]  # 3 years
    five_year = recent_spreads  # 5 years
    
    def avg_spread(arr):
        return round(arr.mean(), 2) if not arr.empty else "N/A"
    
    return {
        "1y Spread": avg_spread(one_year),
        "3y Spread": avg_spread(three_year),
        "5y Spread": avg_spread(five_year)
    }

def interpolate_vol_surface(data, ticker, dataset_date, moneyness_min=0.6, moneyness_max=2.5, expiry_t_min=0.2, expiry_t_max=5.0):
    """Interpolate volatility surface for a ticker."""
    if data.empty or 'Ticker' not in data.columns or 'Smoothed_IV' not in data.columns:
        return pd.DataFrame(columns=["Expiry", "Moneyness", "Volatility", "Expiry_T"])
    
    filtered = data[(data["Ticker"] == ticker) & (data["Type"] == "Call") &
                    data["Moneyness"].notna() & data["Smoothed_IV"].notna()]
    
    if filtered.empty:
        return pd.DataFrame(columns=["Expiry", "Moneyness", "Volatility", "Expiry_T"])
    
    dataset_date = pd.to_datetime(dataset_date)
    filtered = filtered.copy()
    filtered["Expiry_T"] = filtered["Expiry"].apply(
        lambda x: (pd.to_datetime(x) - dataset_date).days / 365.0 if pd.notna(x) else np.nan
    )
    
    filtered = filtered[
        (filtered["Moneyness"] >= moneyness_min) &
        (filtered["Moneyness"] <= moneyness_max) &
        (filtered["Expiry_T"] >= expiry_t_min) &
        (filtered["Expiry_T"] <= expiry_t_max) &
        filtered["Expiry_T"].notna()
    ]
    
    if filtered.empty:
        return pd.DataFrame(columns=["Expiry", "Moneyness", "Volatility", "Expiry_T"])
    
    moneyness_values = np.arange(moneyness_min, moneyness_max + 0.01, 0.01)
    expiry_values = sorted(filtered["Expiry"].unique())
    expiry_times = sorted(filtered["Expiry_T"].unique())
    
    if len(expiry_times) == 0 or len(moneyness_values) == 0:
        return pd.DataFrame(columns=["Expiry", "Moneyness", "Volatility", "Expiry_T"])
    
    grid_x, grid_y = np.meshgrid(expiry_times, moneyness_values)
    points = filtered[["Expiry_T", "Moneyness"]].values
    values = filtered["Smoothed_IV"].values * 100
    
    z = griddata(points, values, (grid_x, grid_y), method="linear")
    
    surface_data = []
    for i, m in enumerate(moneyness_values):
        for j, e_t in enumerate(expiry_times):
            # Find corresponding expiry date for this expiry_time
            corresponding_expiry = filtered[filtered["Expiry_T"] == e_t]["Expiry"].iloc[0] if not filtered[filtered["Expiry_T"] == e_t].empty else expiry_values[j]
            surface_data.append({
                "Expiry": corresponding_expiry,
                "Moneyness": m,
                "Volatility": z[i, j] if not np.isnan(z[i, j]) else None,
                "Expiry_T": e_t
            })
    
    return pd.DataFrame(surface_data).dropna()

def generate_ranking_table(ranking, company_names, cleaned_data, historic_data):
    """Generate ranking table with formatted values and colors."""
    if ranking is None or ranking.empty:
        return pd.DataFrame()
    ranking = ranking.copy()
    if company_names is not None and not company_names.empty:
        ranking["Company Name"] = ranking["Ticker"].map(
            company_names.set_index("Ticker")["Company Name"].to_dict()
        ).fillna("N/A")
    else:
        ranking["Company Name"] = "N/A"
    
    for ticker in ranking["Ticker"].unique():
        if not cleaned_data.empty:
            ranking.loc[ranking["Ticker"] == ticker, "Market Spread"] = calculate_market_spread(cleaned_data, ticker)
        else:
            ranking.loc[ranking["Ticker"] == ticker, "Market Spread"] = "N/A"
        if not historic_data.empty:
            spreads = calculate_historical_spreads(historic_data, ticker)
            for key, value in spreads.items():
                ranking.loc[ranking["Ticker"] == ticker, key] = value
        else:
            for key in ["1y Spread", "3y Spread", "5y Spread"]:
                ranking.loc[ranking["Ticker"] == ticker, key] = "N/A"
    
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
        "Open Interest", "OI 1d (%)", "OI 1w (%)", "Market Spread", "1y Spread",
        "3y Spread", "5y Spread"
    ]
    
    # Ensure all columns exist
    for col in columns:
        if col not in ranking.columns:
            ranking[col] = "N/A"
    
    # Rank by Open Interest (handle non-numeric)
    ranking['Open Interest Numeric'] = pd.to_numeric(ranking['Open Interest'], errors='coerce').fillna(0)
    ranking["Rank"] = ranking['Open Interest Numeric'].rank(ascending=False, na_option="bottom").astype(int)
    ranking = ranking.drop('Open Interest Numeric', axis=1)
    
    for col in columns:
        if col in ["Volume", "Open Interest"]:
            ranking[col] = ranking[col].apply(lambda x: f"{int(pd.to_numeric(x, errors='coerce')):,}" if pd.notna(x) and pd.to_numeric(x, errors='coerce') > 0 else "N/A")
        elif col not in ["Rank", "Ticker", "Company Name"]:
            ranking[col] = ranking[col].apply(lambda x: round(pd.to_numeric(x, errors='coerce'), 2) if pd.notna(x) and pd.to_numeric(x, errors='coerce') == pd.to_numeric(x, errors='coerce') else "N/A")
        
        color_col = f"{col}_Color"
        ranking[color_col] = "#FFFFFF"
        if col in ["Realised Volatility 100d 1d (%)", "Realised Volatility 100d 1w (%)",
                   "Weighted IV 1d (%)", "Weighted IV 1w (%)", "Weighted IV 3m 1d (%)",
                   "Weighted IV 3m 1w (%)", "ATM IV 3m 1d (%)", "ATM IV 3m 1w (%)",
                   "Rvol100d - Weighted IV", "Volume 1d (%)", "Volume 1w (%)",
                   "OI 1d (%)", "OI 1w (%)"]:
            def get_color(val):
                try:
                    num_val = pd.to_numeric(val, errors='coerce')
                    if pd.isna(num_val):
                        return "#FFFFFF"
                    return "#F87171" if num_val < 0 else "#10B981" if num_val > 0 else "#FFFFFF"
                except:
                    return "#FFFFFF"
            ranking[color_col] = ranking[col].apply(get_color)
    
    color_columns = [f"{col}_Color" for col in columns if col not in ["Rank", "Ticker", "Company Name"]]
    return ranking[columns + color_columns]

def generate_stock_table(ranking, company_names, cleaned_data, historic_data):
    """Generate stock table with formatted values and colors."""
    if ranking is None or ranking.empty:
        return pd.DataFrame()
    stock_data = ranking.copy()
    if company_names is not None and not company_names.empty:
        stock_data["Company Name"] = stock_data["Ticker"].map(
            company_names.set_index("Ticker")["Company Name"].to_dict()
        ).fillna("N/A")
    else:
        stock_data["Company Name"] = "N/A"
    
    for ticker in stock_data["Ticker"].unique():
        if not cleaned_data.empty:
            stock_data.loc[stock_data["Ticker"] == ticker, "Market Spread"] = calculate_market_spread(cleaned_data, ticker)
        else:
            stock_data.loc[stock_data["Ticker"] == ticker, "Market Spread"] = "N/A"
        if not historic_data.empty:
            spreads = calculate_historical_spreads(historic_data, ticker)
            for key, value in spreads.items():
                stock_data.loc[stock_data["Ticker"] == ticker, key] = value
        else:
            for key in ["1y Spread", "3y Spread", "5y Spread"]:
                stock_data.loc[stock_data["Ticker"] == ticker, key] = "N/A"
    
    columns = [
        "Ticker", "Company Name", "Latest Open", "Latest Close", "Latest High",
        "Latest Low", "Open 1d (%)", "Open 1w (%)", "Close 1d (%)", "Close 1w (%)",
        "High 1d (%)", "High 1w (%)", "Low 1d (%)", "Low 1w (%)", "Market Spread",
        "1y Spread", "3y Spread", "5y Spread"
    ]
    
    # Ensure all columns exist
    for col in columns:
        if col not in stock_data.columns:
            stock_data[col] = "N/A"
    
    stock_data = stock_data.sort_values("Latest Close", ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce'))
    for col in columns:
        if col not in ["Ticker", "Company Name"]:
            stock_data[col] = stock_data[col].apply(lambda x: round(pd.to_numeric(x, errors='coerce'), 2) if pd.notna(x) and pd.to_numeric(x, errors='coerce') == pd.to_numeric(x, errors='coerce') else "N/A")
        
        color_col = f"{col}_Color"
        stock_data[color_col] = "#FFFFFF"
        if col in ["Open 1d (%)", "Open 1w (%)", "Close 1d (%)", "Close 1w (%)",
                   "High 1d (%)", "High 1w (%)", "Low 1d (%)", "Low 1w (%)"]:
            def get_color(val):
                try:
                    num_val = pd.to_numeric(val, errors='coerce')
                    if pd.isna(num_val):
                        return "#FFFFFF"
                    return "#F87171" if num_val < 0 else "#10B981" if num_val > 0 else "#FFFFFF"
                except:
                    return "#FFFFFF"
            stock_data[color_col] = stock_data[col].apply(get_color)
    
    color_columns = [f"{col}_Color" for col in columns if col not in ["Ticker", "Company Name"]]
    return stock_data[columns + color_columns]

def generate_summary_table(ranking, skew_data, ticker):
    """Generate summary table for a ticker with formatted values and colors."""
    if ranking is None or ranking.empty:
        return pd.DataFrame(columns=["Metric", "Value", "Color"])
    
    filtered_ranking = ranking[ranking["Ticker"] == ticker]
    if filtered_ranking.empty:
        return pd.DataFrame(columns=["Metric", "Value", "Color"])
    
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
        {"name": "OI 1w (%)", "key": "OI 1w (%)"}
    ]
    
    summary_data = []
    for metric in metrics:
        value = filtered_ranking[metric["key"]].iloc[0] if metric["key"] in filtered_ranking.columns and not filtered_ranking[metric["key"]].empty else "N/A"
        if metric["key"] in ["Volume", "Open Interest"]:
            num_val = pd.to_numeric(value, errors='coerce')
            value = f"{int(num_val):,}" if pd.notna(num_val) and num_val > 0 else "N/A"
        else:
            num_val = pd.to_numeric(value, errors='coerce')
            value = round(num_val, 2) if pd.notna(num_val) else "N/A"
        
        color = "#FFFFFF"
        if metric["key"] in ["Close 1d (%)", "Close 1w (%)", "Weighted IV 1d (%)",
                            "Weighted IV 1w (%)", "Volume 1d (%)", "Volume 1w (%)",
                            "OI 1d (%)", "OI 1w (%)"]:
            num_val = pd.to_numeric(value, errors='coerce')
            if pd.notna(num_val):
                color = "#F87171" if num_val < 0 else "#10B981" if num_val > 0 else "#FFFFFF"
        
        summary_data.append({"Metric": metric["name"], "Value": value, "Color": color})
    
    atm_ratio = "N/A"
    if not skew_data.empty and skew_data["Ticker"].isin([ticker]).any():
        atm_val = skew_data[skew_data["Ticker"] == ticker]["ATM_12m_3m_Ratio"].iloc[0]
        atm_ratio = round(pd.to_numeric(atm_val, errors='coerce'), 4) if pd.notna(atm_val) else "N/A"
    summary_data.append({"Metric": "ATM 12m/3m Ratio", "Value": atm_ratio, "Color": "#FFFFFF"})
    
    return pd.DataFrame(summary_data)

def generate_top_contracts_tables(data, ticker):
    """Generate top 10 volume and open interest contracts tables for a ticker."""
    if data.empty or 'Ticker' not in data.columns:
        return pd.DataFrame(columns=["Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]), pd.DataFrame(columns=["Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"])
    
    filtered = data[data["Ticker"] == ticker].copy()
    
    top_volume = filtered[filtered["Volume"].notna()].sort_values("Volume", ascending=False).head(10) if not filtered.empty else pd.DataFrame()
    top_open_interest = filtered[filtered["Open Interest"].notna()].sort_values("Open Interest", ascending=False).head(10) if not filtered.empty else pd.DataFrame()
    
    def format_table(df):
        if df.empty:
            return pd.DataFrame(columns=["Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"])
        required_cols = ["Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]
        df = df[required_cols].copy() if all(col in df.columns for col in required_cols) else pd.DataFrame(columns=required_cols)
        df["Strike"] = df["Strike"].apply(lambda x: round(pd.to_numeric(x, errors='coerce'), 2) if pd.notna(x) else "N/A")
        df["Expiry"] = df["Expiry"].apply(lambda x: pd.to_datetime(x).strftime("%d/%m/%Y") if pd.notna(x) else "N/A")
        df["Type"] = df["Type"].fillna("N/A")
        df["Bid"] = df["Bid"].apply(lambda x: round(pd.to_numeric(x, errors='coerce'), 2) if pd.notna(x) else "N/A")
        df["Ask"] = df["Ask"].apply(lambda x: round(pd.to_numeric(x, errors='coerce'), 2) if pd.notna(x) else "N/A")
        df["Volume"] = df["Volume"].apply(lambda x: f"{int(pd.to_numeric(x, errors='coerce')):,}" if pd.notna(x) else "N/A")
        df["Open Interest"] = df["Open Interest"].apply(lambda x: f"{int(pd.to_numeric(x, errors='coerce')):,}" if pd.notna(x) else "N/A")
        return df
    
    return format_table(top_volume), format_table(top_open_interest)

def save_tables(timestamp, source, base_path="data"):
    """Generate and save all precomputed tables."""
    prefix = "_yfinance" if source == "yfinance" else ""
    company_names, ranking, historic, events = load_data(timestamp, source, base_path)
    if ranking is None or ranking.empty:
        print("Failed to load required data files (ranking is missing or empty). Skipping table generation.")
        return
    
    # Create directories
    os.makedirs(f"{base_path}/{timestamp}/tables/ranking", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/stock", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/summary", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/contracts", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/vol_surface", exist_ok=True)
    
    # Get unique tickers from ranking
    tickers = ranking["Ticker"].unique()
    dataset_date = pd.to_datetime(timestamp[:8], format="%Y%m%d")
    
    processed_dir = f"{base_path}/{timestamp}/processed{prefix}"
    cleaned_dir = f"{base_path}/{timestamp}/cleaned_yfinance"
    skew_dir = f"{base_path}/{timestamp}/skew_metrics{prefix}"
    
    for ticker in tickers:
        try:
            # Load ticker-specific data with fallbacks to empty DataFrames
            processed = pd.DataFrame()
            if os.path.exists(f"{processed_dir}/processed{prefix}_{ticker}.csv"):
                processed = pd.read_csv(f"{processed_dir}/processed{prefix}_{ticker}.csv")
            
            cleaned = pd.DataFrame()
            if os.path.exists(f"{cleaned_dir}/cleaned_yfinance_{ticker}.csv"):
                cleaned = pd.read_csv(f"{cleaned_dir}/cleaned_yfinance_{ticker}.csv")
            
            skew = pd.DataFrame()
            if os.path.exists(f"{skew_dir}/skew_metrics{prefix}_{ticker}.csv"):
                skew = pd.read_csv(f"{skew_dir}/skew_metrics{prefix}_{ticker}.csv")
            
            # Generate tables (will use fallbacks if data is empty)
            ranking_table = generate_ranking_table(ranking, company_names, cleaned, historic)
            stock_table = generate_stock_table(ranking, company_names, cleaned, historic)
            summary_table = generate_summary_table(ranking, skew, ticker)
            top_volume, top_open_interest = generate_top_contracts_tables(processed, ticker)
            vol_surface = interpolate_vol_surface(processed, ticker, dataset_date)
            
            # Save tables (only if they have data)
            ranking_table.to_csv(f"{base_path}/{timestamp}/tables/ranking/ranking_table{prefix}.csv", index=False)
            stock_table.to_csv(f"{base_path}/{timestamp}/tables/stock/stock_table{prefix}.csv", index=False)
            summary_table.to_csv(f"{base_path}/{timestamp}/tables/summary/summary{prefix}_{ticker}.csv", index=False)
            contracts_dir = f"{base_path}/{timestamp}/tables/contracts/{ticker}"
            os.makedirs(contracts_dir, exist_ok=True)
            top_volume.to_csv(f"{contracts_dir}/top_volume{prefix}_{ticker}.csv", index=False)
            top_open_interest.to_csv(f"{contracts_dir}/top_open_interest{prefix}_{ticker}.csv", index=False)
            vol_surface.to_csv(f"{base_path}/{timestamp}/tables/vol_surface/vol_surface{prefix}_{ticker}.csv", index=False)
            
            print(f"Generated tables for ticker: {ticker}")
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
    
    print(f"Precomputed tables generation completed for {timestamp}, source {source}")

def main():
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
        timestamp = timestamps[-1]  # Use the latest timestamp if none provided
        print(f"No timestamp provided, using latest: {timestamp}")
    
    sources = ['yfinance']
    for source in sources:
        save_tables(timestamp, source, base_path)
        print(f"Precomputed tables generated for {timestamp}, source {source}")

main()
