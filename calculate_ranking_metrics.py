import sys
import pandas as pd
import json
from datetime import datetime, timedelta
import os

if len(sys.argv) < 3:
    print("Usage: python calculate_ranking_metrics.py <timestamp> <source>")
    sys.exit(1)

current_ts = sys.argv[1]
source = sys.argv[2]
prefix = 'yfinance_' if source == 'yfinance' else ''
data_dir = 'data'
dates_file = os.path.join(data_dir, 'dates.json')

with open(dates_file, 'r') as f:
    dates = json.load(f)

timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))

if current_ts not in timestamps:
    print("Timestamp not in dates.json")
    sys.exit(1)

current_index = timestamps.index(current_ts)
current_dt = datetime.strptime(current_ts[:8], "%Y%m%d")
prev_day_dt = current_dt - timedelta(days=1)
prev_week_dt = current_dt - timedelta(days=7)

prev_day_ts = None
for ts in timestamps[:current_index][::-1]:
    ts_dt = datetime.strptime(ts[:8], "%Y%m%d")
    if ts_dt <= prev_day_dt:
        prev_day_ts = ts
        break

prev_week_ts = None
for ts in timestamps[:current_index][::-1]:
    ts_dt = datetime.strptime(ts[:8], "%Y%m%d")
    if ts_dt <= prev_week_dt:
        prev_week_ts = ts
        break

def get_option_totals(ts, prefix):
    if ts is None:
        return pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
    file = os.path.join(data_dir, f"raw_{prefix}{ts}.csv")
    if not os.path.exists(file):
        return pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
    df = pd.read_csv(file)
    totals = df.groupby('Ticker').agg({
        'Open Interest': 'sum',
        'Volume': 'sum'
    }).reset_index()
    totals.rename(columns={'Open Interest': 'OI', 'Volume': 'Vol'}, inplace=True)
    return totals

current_option = get_option_totals(current_ts, prefix)
prev_day_option = get_option_totals(prev_day_ts, prefix)
prev_week_option = get_option_totals(prev_week_ts, prefix)

historic_file = os.path.join(data_dir, f"historic_{current_ts}.csv")
df_historic = pd.read_csv(historic_file)
df_historic['Date'] = pd.to_datetime(df_historic['Date'])

latest_historic = df_historic.loc[df_historic.groupby('Ticker')['Date'].idxmax()]

def get_prev_value(ticker, target_date, col):
    g = df_historic[df_historic['Ticker'] == ticker]
    prev = g[g['Date'] <= target_date]
    if prev.empty:
        return None
    return prev.loc[prev['Date'].idxmax(), col]

ranking = []
for row in current_option.itertuples():
    ticker = row.Ticker
    oi = row.OI
    vol = row.Vol
    prev_oi = prev_day_option.loc[prev_day_option['Ticker'] == ticker, 'OI'].values[0] if not prev_day_option[prev_day_option['Ticker'] == ticker].empty else 0
    oi_1d_pct = (oi - prev_oi) / prev_oi * 100 if prev_oi != 0 else 'N/A'
    prev_week_oi = prev_week_option.loc[prev_week_option['Ticker'] == ticker, 'OI'].values[0] if not prev_week_option[prev_week_option['Ticker'] == ticker].empty else 0
    oi_1w_pct = (oi - prev_week_oi) / prev_week_oi * 100 if prev_week_oi != 0 else 'N/A'
    prev_vol = prev_day_option.loc[prev_day_option['Ticker'] == ticker, 'Vol'].values[0] if not prev_day_option[prev_day_option['Ticker'] == ticker].empty else 0
    vol_1d_pct = (vol - prev_vol) / prev_vol * 100 if prev_vol != 0 else 'N/A'
    prev_week_vol = prev_week_option.loc[prev_week_option['Ticker'] == ticker, 'Vol'].values[0] if not prev_week_option[prev_week_option['Ticker'] == ticker].empty else 0
    vol_1w_pct = (vol - prev_week_vol) / prev_week_vol * 100 if prev_week_vol != 0 else 'N/A'
    latest = latest_historic[latest_historic['Ticker'] == ticker]
    if latest.empty:
        vol30 = 'N/A'
        vol60 = 'N/A'
        vol100 = 'N/A'
        vol180 = 'N/A'
        vol252 = 'N/A'
        close_1d_pct = 'N/A'
        close_1w_pct = 'N/A'
    else:
        vol30 = latest['Realised_Vol_30'].values[0] if 'Realised_Vol_30' in latest else 'N/A'
        vol60 = latest['Realised_Vol_60'].values[0] if 'Realised_Vol_60' in latest else 'N/A'
        vol100 = latest['Realised_Vol_100'].values[0] if 'Realised_Vol_100' in latest else 'N/A'
        vol180 = latest['Realised_Vol_180'].values[0] if 'Realised_Vol_180' in latest else 'N/A'
        vol252 = latest['Realised_Vol_252'].values[0] if 'Realised_Vol_252' in latest else 'N/A'
        latest_close = latest['Close'].values[0]
        prev_close = get_prev_value(ticker, prev_day_dt, 'Close')
        close_1d_pct = (latest_close - prev_close) / prev_close * 100 if prev_close is not None and prev_close != 0 else 'N/A'
        prev_week_close = get_prev_value(ticker, prev_week_dt, 'Close')
        close_1w_pct = (latest_close - prev_week_close) / prev_week_close * 100 if prev_week_close is not None and prev_week_close != 0 else 'N/A'
    ranking.append({
        'Ticker': ticker,
        'Open Interest': oi,
        'Volume': vol,
        'OI 1d (%)': oi_1d_pct,
        'OI 1w (%)': oi_1w_pct,
        'Volume 1d (%)': vol_1d_pct,
        'Volume 1w (%)': vol_1w_pct,
        'Close 1d (%)': close_1d_pct,
        'Close 1w (%)': close_1w_pct,
        'Realised Volatility 30d (%)': vol30,
        'Realised Volatility 60d (%)': vol60,
        'Realised Volatility 100d (%)': vol100,
        'Realised Volatility 180d (%)': vol180,
        'Realised Volatility 252d (%)': vol252,
    })

df_ranking = pd.DataFrame(ranking)
output_file = os.path.join(data_dir, f"ranking_{prefix}{current_ts}.csv")
df_ranking.to_csv(output_file, index=False)
