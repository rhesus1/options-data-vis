import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

def calculate_ranking_metrics(timestamp, sources, data_dir='data'):
    dates_file = os.path.join(data_dir, 'dates.json')
    
    with open(dates_file, 'r') as f:
        dates = json.load(f)
    
    timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
    
    if timestamp not in timestamps:
        print(f"Timestamp {timestamp} not in dates.json")
        return
    
    for source in sources:
        prefix = 'yfinance_' if source == 'yfinance' else ''
        
        current_index = timestamps.index(timestamp)
        current_dt = datetime.strptime(timestamp[:8], "%Y%m%d")
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
        
        current_option = get_option_totals(timestamp, prefix)
        prev_day_option = get_option_totals(prev_day_ts, prefix)
        prev_week_option = get_option_totals(prev_week_ts, prefix)
        
        historic_file = os.path.join(data_dir, f"historic_{timestamp}.csv")
        if not os.path.exists(historic_file):
            print(f"Historic file {historic_file} not found")
            continue
        df_historic = pd.read_csv(historic_file)
        df_historic['Date'] = pd.to_datetime(df_historic['Date'])
        
        processed_file = os.path.join(data_dir, f"processed_{prefix}{timestamp}.csv")
        if not os.path.exists(processed_file):
            print(f"Processed file {processed_file} not found")
            continue
        df_processed = pd.read_csv(processed_file)
        
        latest_historic = df_historic.loc[df_historic.groupby('Ticker')['Date'].idxmax()]
        
        def get_prev_value(ticker, target_date, col):
            g = df_historic[df_historic['Ticker'] == ticker]
            prev = g[g['Date'] <= target_date]
            if prev.empty:
                return None
            return prev.loc[prev['Date'].idxmax(), col]
        
        def calculate_weighted_iv(ticker_df):
            if ticker_df.empty or 'IV_mid' not in ticker_df.columns or ticker_df['IV_mid'].isna().all():
                return np.nan
            mid = (ticker_df['Bid'] + ticker_df['Ask']) / 2
            spread = ticker_df['Ask'] - ticker_df['Bid']
            smi = 100 * (spread / mid)
            smi = smi.replace(0, np.nan)  # Avoid division by zero
            weights = np.log(1 + ticker_df['Open Interest']) / smi
            valid = (~weights.isna()) & (~ticker_df['IV_mid'].isna())
            if not valid.any():
                return np.nan
            weighted_iv = np.average(ticker_df.loc[valid, 'IV_mid'], weights=weights[valid])
            return weighted_iv
        
        ranking = []
        past_year_start = current_dt - timedelta(days=365)
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
            ticker_processed = df_processed[df_processed['Ticker'] == ticker]
            weighted_iv = calculate_weighted_iv(ticker_processed) if not ticker_processed.empty else np.nan
            latest = latest_historic[latest_historic['Ticker'] == ticker]
            rank_dict = {
                'Ticker': ticker,
                'Open Interest': oi,
                'Volume': vol,
                'OI 1d (%)': oi_1d_pct,
                'OI 1w (%)': oi_1w_pct,
                'Volume 1d (%)': vol_1d_pct,
                'Volume 1w (%)': vol_1w_pct,
                'Weighted IV (%)': weighted_iv * 100 if not np.isnan(weighted_iv) else 'N/A'
            }
            if latest.empty:
                rank_dict['Latest Close'] = 'N/A'
                rank_dict['Close 1d (%)'] = 'N/A'
                rank_dict['Close 1w (%)'] = 'N/A'
                rank_dict['Realised Volatility 100d (%)'] = 'N/A'
                rank_dict['Realised Volatility 100d 1d (%)'] = 'N/A'
                rank_dict['Realised Volatility 100d 1w (%)'] = 'N/A'
                rank_dict['Min Realised Volatility 100d (1y)'] = 'N/A'
                rank_dict['Max Realised Volatility 100d (1y)'] = 'N/A'
                rank_dict['Mean Realised Volatility 100d (1y)'] = 'N/A'
            else:
                latest_close = latest['Close'].values[0]
                rank_dict['Latest Close'] = latest_close
                prev_close = get_prev_value(ticker, prev_day_dt, 'Close')
                close_1d_pct = (latest_close - prev_close) / prev_close * 100 if prev_close is not None and prev_close != 0 else 'N/A'
                rank_dict['Close 1d (%)'] = close_1d_pct
                prev_week_close = get_prev_value(ticker, prev_week_dt, 'Close')
                close_1w_pct = (latest_close - prev_week_close) / prev_week_close * 100 if prev_week_close is not None and prev_week_close != 0 else 'N/A'
                rank_dict['Close 1w (%)'] = close_1w_pct
                current_vol = latest['Realised_Vol_100'].values[0] if 'Realised_Vol_100' in latest.columns else 'N/A'
                rank_dict['Realised Volatility 100d (%)'] = current_vol
                prev_day_vol = get_prev_value(ticker, prev_day_dt, 'Realised_Vol_100')
                vol_1d_pct = (current_vol - prev_day_vol) / prev_day_vol * 100 if prev_day_vol is not None and prev_day_vol != 0 else 'N/A'
                rank_dict['Realised Volatility 100d 1d (%)'] = vol_1d_pct
                prev_week_vol = get_prev_value(ticker, prev_week_dt, 'Realised_Vol_100')
                vol_1w_pct = (current_vol - prev_week_vol) / prev_week_vol * 100 if prev_week_vol is not None and prev_week_vol != 0 else 'N/A'
                rank_dict['Realised Volatility 100d 1w (%)'] = vol_1w_pct
                past_year = df_historic[(df_historic['Ticker'] == ticker) & (df_historic['Date'] >= past_year_start) & (df_historic['Date'] <= current_dt)]
                vols = past_year['Realised_Vol_100'].dropna()
                if not vols.empty:
                    min_vol = vols.min()
                    max_vol = vols.max()
                    mean_vol = vols.mean()
                else:
                    min_vol = max_vol = mean_vol = 'N/A'
                rank_dict['Min Realised Volatility 100d (1y)'] = min_vol
                rank_dict['Max Realised Volatility 100d (1y)'] = max_vol
                rank_dict['Mean Realised Volatility 100d (1y)'] = mean_vol
            ranking.append(rank_dict)
        
        df_ranking = pd.DataFrame(ranking)
        output_file = os.path.join(data_dir, f"ranking_{prefix}{timestamp}.csv")
        df_ranking.to_csv(output_file, index=False)
        print(f"Ranking metrics saved to {output_file}")

data_dir = 'data'
dates_file = os.path.join(data_dir, 'dates.json')

with open(dates_file, 'r') as f:
    dates = json.load(f)

timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))

if len(sys.argv) > 1:
    timestamp = sys.argv[1]
    sources = ['yfinance', 'nasdaq']
    calculate_ranking_metrics(timestamp, sources)
else:
    sources = ['yfinance', 'nasdaq']
    for timestamp in timestamps:
        calculate_ranking_metrics(timestamp, sources)
