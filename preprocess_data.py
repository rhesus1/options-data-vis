import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
import sys
import json
from datetime import datetime

def load_data(timestamp, source, base_path="data"):
    """Load raw data files for a given timestamp and source."""
    prefix = "_yfinance" if source == "yfinance" else ""
    try:
        # Load data files
        company_names = pd.read_csv("company_names.txt", sep="\t")
        ranking = pd.read_csv(f"{base_path}/{timestamp}/ranking/ranking{prefix}.csv")
        historic = pd.read_csv(f"{base_path}/{timestamp}/historic/historic_*.csv")  # Note: Adjust for specific ticker if needed
        events = pd.read_csv(f"{base_path}/Events.csv") if os.path.exists(f"{base_path}/Events.csv") else pd.DataFrame()
        return company_names, ranking, historic, events
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        return None, None, None, None

def calculate_market_spread(cleaned_data, ticker):
    """Calculate market spread for a ticker."""
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
    filtered = data[(data["Ticker"] == ticker) & (data["Type"] == "Call") & 
                    data["Moneyness"].notna() & data["Smoothed_IV"].notna()]
    
    if filtered.empty:
        return pd.DataFrame(columns=["Expiry", "Moneyness", "Volatility", "Expiry_T"])
    
    dataset_date = pd.to_datetime(dataset_date)
    filtered["Expiry_T"] = filtered["Expiry"].apply(
        lambda x: (pd.to_datetime(x) - dataset_date).days / 365.0
    )
    
    filtered = filtered[
        (filtered["Moneyness"] >= moneyness_min) & 
        (filtered["Moneyness"] <= moneyness_max) &
        (filtered["Expiry_T"] >= expiry_t_min) & 
        (filtered["Expiry_T"] <= expiry_t_max)
    ]
    
    if filtered.empty:
        return pd.DataFrame(columns=["Expiry", "Moneyness", "Volatility", "Expiry_T"])
    
    moneyness_values = np.arange(moneyness_min, moneyness_max + 0.01, 0.01)
    expiry_values = sorted(filtered["Expiry"].unique())
    expiry_times = filtered["Expiry_T"].unique()
    
    grid_x, grid_y = np.meshgrid(expiry_times, moneyness_values)
    points = filtered[["Expiry_T", "Moneyness"]].values
    values = filtered["Smoothed_IV"].values * 100
    
    z = griddata(points, values, (grid_x, grid_y), method="linear")
    
    surface_data = []
    for i, m in enumerate(moneyness_values):
        for j, e in enumerate(expiry_values):
            surface_data.append({
                "Expiry": e,
                "Moneyness": m,
                "Volatility": z[i, j] if not np.isnan(z[i, j]) else None,
                "Expiry_T": expiry_times[j]
            })
    
    return pd.DataFrame(surface_data).dropna()

def generate_ranking_table(ranking, company_names, cleaned_data, historic_data):
    """Generate ranking table with formatted values and colors."""
    ranking = ranking.copy()
    ranking["Company Name"] = ranking["Ticker"].map(
        company_names.set_index("Ticker")["Company Name"].to_dict()
    ).fillna("N/A")
    
    for ticker in ranking["Ticker"]:
        ranking.loc[ranking["Ticker"] == ticker, "Market Spread"] = calculate_market_spread(cleaned_data, ticker)
        spreads = calculate_historical_spreads(historic_data, ticker)
        for key, value in spreads.items():
            ranking.loc[ranking["Ticker"] == ticker, key] = value
    
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
    
    ranking["Rank"] = ranking["Open Interest"].rank(ascending=False, na_option="bottom").astype(int)
    for col in columns:
        if col in ["Volume", "Open Interest"]:
            ranking[col] = ranking[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
        elif col not in ["Rank", "Ticker", "Company Name"]:
            ranking[col] = ranking[col].apply(lambda x: round(x, 2) if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
        
        color_col = f"{col}_Color"
        ranking[color_col] = "#FFFFFF"
        if col in ["Realised Volatility 100d 1d (%)", "Realised Volatility 100d 1w (%)",
                   "Weighted IV 1d (%)", "Weighted IV 1w (%)", "Weighted IV 3m 1d (%)",
                   "Weighted IV 3m 1w (%)", "ATM IV 3m 1d (%)", "ATM IV 3m 1w (%)",
                   "Rvol100d - Weighted IV", "Volume 1d (%)", "Volume 1w (%)",
                   "OI 1d (%)", "OI 1w (%)"]:
            ranking[color_col] = ranking[col].apply(
                lambda x: "#F87171" if pd.notna(x) and isinstance(x, (int, float)) and x < 0 else
                          "#10B981" if pd.notna(x) and isinstance(x, (int, float)) and x > 0 else "#FFFFFF"
            )
    
    return ranking[columns + [f"{col}_Color" for col in columns if col not in ["Rank", "Ticker", "Company Name"]]]

def generate_stock_table(ranking, company_names, cleaned_data, historic_data):
    """Generate stock table with formatted values and colors."""
    stock_data = ranking.copy()
    stock_data["Company Name"] = stock_data["Ticker"].map(
        company_names.set_index("Ticker")["Company Name"].to_dict()
    ).fillna("N/A")
    
    for ticker in stock_data["Ticker"]:
        stock_data.loc[stock_data["Ticker"] == ticker, "Market Spread"] = calculate_market_spread(cleaned_data, ticker)
        spreads = calculate_historical_spreads(historic_data, ticker)
        for key, value in spreads.items():
            stock_data.loc[stock_data["Ticker"] == ticker, key] = value
    
    columns = [
        "Ticker", "Company Name", "Latest Open", "Latest Close", "Latest High",
        "Latest Low", "Open 1d (%)", "Open 1w (%)", "Close 1d (%)", "Close 1w (%)",
        "High 1d (%)", "High 1w (%)", "Low 1d (%)", "Low 1w (%)", "Market Spread",
        "1y Spread", "3y Spread", "5y Spread"
    ]
    
    stock_data = stock_data.sort_values("Latest Close", ascending=False)
    for col in columns:
        if col not in ["Ticker", "Company Name"]:
            stock_data[col] = stock_data[col].apply(lambda x: round(x, 2) if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
        
        color_col = f"{col}_Color"
        stock_data[color_col] = "#FFFFFF"
        if col in ["Open 1d (%)", "Open 1w (%)", "Close 1d (%)", "Close 1w (%)",
                   "High 1d (%)", "High 1w (%)", "Low 1d (%)", "Low 1w (%)"]:
            stock_data[color_col] = stock_data[col].apply(
                lambda x: "#F87171" if pd.notna(x) and isinstance(x, (int, float)) and x < 0 else
                          "#10B981" if pd.notna(x) and isinstance(x, (int, float)) and x > 0 else "#FFFFFF"
            )
    
    return stock_data[columns + [f"{col}_Color" for col in columns if col not in ["Ticker", "Company Name"]]]

def generate_summary_table(ranking, skew_data, ticker):
    """Generate summary table for a ticker with formatted values and colors."""
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
        value = filtered_ranking[metric["key"]].iloc[0] if not filtered_ranking[metric["key"]].empty else "N/A"
        if metric["key"] in ["Volume", "Open Interest"]:
            value = f"{int(value):,}" if pd.notna(value) and isinstance(value, (int, float)) else "N/A"
        elif metric["key"] not in ["Volume", "Open Interest"]:
            value = round(value, 2) if pd.notna(value) and isinstance(value, (int, float)) else "N/A"
        
        color = "#FFFFFF"
        if metric["key"] in ["Close 1d (%)", "Close 1w (%)", "Weighted IV 1d (%)", 
                            "Weighted IV 1w (%)", "Volume 1d (%)", "Volume 1w (%)", 
                            "OI 1d (%)", "OI 1w (%)"]:
            color = "#F87171" if pd.notna(value) and isinstance(value, (int, float)) and value < 0 else \
                    "#10B981" if pd.notna(value) and isinstance(value, (int, float)) and value > 0 else "#FFFFFF"
        
        summary_data.append({"Metric": metric["name"], "Value": value, "Color": color})
    
    atm_ratio = skew_data[skew_data["Ticker"] == ticker]["ATM_12m_3m_Ratio"].iloc[0] if not skew_data[skew_data["Ticker"] == ticker].empty else "N/A"
    atm_ratio = round(atm_ratio, 4) if pd.notna(atm_ratio) and isinstance(atm_ratio, (int, float)) else "N/A"
    summary_data.append({"Metric": "ATM 12m/3m Ratio", "Value": atm_ratio, "Color": "#FFFFFF"})
    
    return pd.DataFrame(summary_data)

def generate_top_contracts_tables(data, ticker):
    """Generate top 10 volume and open interest contracts tables for a ticker."""
    filtered = data[data["Ticker"] == ticker]
    
    top_volume = filtered[filtered["Volume"].notna()].sort_values("Volume", ascending=False).head(10)
    top_open_interest = filtered[filtered["Open Interest"].notna()].sort_values("Open Interest", ascending=False).head(10)
    
    def format_table(df):
        if df.empty:
            return pd.DataFrame(columns=["Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"])
        df = df[["Strike", "Expiry", "Type", "Bid", "Ask", "Volume", "Open Interest"]].copy()
        df["Strike"] = df["Strike"].apply(lambda x: round(x, 2) if pd.notna(x) else "N/A")
        df["Expiry"] = df["Expiry"].apply(lambda x: pd.to_datetime(x).strftime("%d/%m/%Y") if pd.notna(x) else "N/A")
        df["Type"] = df["Type"].fillna("N/A")
        df["Bid"] = df["Bid"].apply(lambda x: round(x, 2) if pd.notna(x) else "N/A")
        df["Ask"] = df["Ask"].apply(lambda x: round(x, 2) if pd.notna(x) else "N/A")
        df["Volume"] = df["Volume"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
        df["Open Interest"] = df["Open Interest"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
        return df
    
    return format_table(top_volume), format_table(top_open_interest)

def save_tables(timestamp, source, base_path="data"):
    """Generate and save all precomputed tables."""
    prefix = "_yfinance" if source == "yfinance" else ""
    company_names, ranking, historic, events = load_data(timestamp, source, base_path)
    if any(x is None for x in [company_names, ranking, historic]):
        print("Failed to load required data files.")
        return
    
    # Create directories
    os.makedirs(f"{base_path}/{timestamp}/tables/ranking", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/stock", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/summary", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/contracts", exist_ok=True)
    os.makedirs(f"{base_path}/{timestamp}/tables/vol_surface", exist_ok=True)
    
    # Load ticker-specific data
    tickers = ranking["Ticker"].unique()
    dataset_date = pd.to_datetime(timestamp[:8], format="%Y%m%d")
    
    for ticker in tickers:
        try:
            processed = pd.read_csv(f"{base_path}/{timestamp}/processed{prefix}/processed{prefix}_{ticker}.csv")
            cleaned = pd.read_csv(f"{base_path}/{timestamp}/cleaned_yfinance/cleaned_yfinance_{ticker}.csv")
            skew = pd.read_csv(f"{base_path}/{timestamp}/skew_metrics{prefix}/skew_metrics{prefix}_{ticker}.csv")
            
            # Generate tables
            ranking_table = generate_ranking_table(ranking, company_names, cleaned, historic)
            stock_table = generate_stock_table(ranking, company_names, cleaned, historic)
            summary_table = generate_summary_table(ranking, skew, ticker)
            top_volume, top_open_interest = generate_top_contracts_tables(processed, ticker)
            vol_surface = interpolate_vol_surface(processed, ticker, dataset_date)
            
            # Save tables
            ranking_table.to_csv(f"{base_path}/{timestamp}/tables/ranking/ranking_table{prefix}.csv", index=False)
            stock_table.to_csv(f"{base_path}/{timestamp}/tables/stock/stock_table{prefix}.csv", index=False)
            summary_table.to_csv(f"{base_path}/{timestamp}/tables/summary/summary{prefix}_{ticker}.csv", index=False)
            os.makedirs(f"{base_path}/{timestamp}/tables/contracts/{ticker}", exist_ok=True)
            top_volume.to_csv(f"{base_path}/{timestamp}/tables/contracts/{ticker}/top_volume{prefix}_{ticker}.csv", index=False)
            top_open_interest.to_csv(f"{base_path}/{timestamp}/tables/contracts/{ticker}/top_open_interest{prefix}_{ticker}.csv", index=False)
            vol_surface.to_csv(f"{base_path}/{timestamp}/tables/vol_surface/vol_surface{prefix}_{ticker}.csv", index=False)
            
            print(f"Generated tables for ticker: {ticker}")
        except FileNotFoundError as e:
            print(f"Error processing ticker {ticker}: {e}")

def main():
    data_dir = 'data'
    dates_file = os.path.join(data_dir, 'dates.json')
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
        save_tables(timestamp, source)
        print(f"Precomputed tables generated for {timestamp}, source {source}")

main()
