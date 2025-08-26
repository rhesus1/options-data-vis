import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from scipy.stats import norm

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
       
        processed_prev_day_file = os.path.join(data_dir, f"processed_{prefix}{prev_day_ts}.csv") if prev_day_ts else None
        df_processed_prev_day = pd.read_csv(processed_prev_day_file) if processed_prev_day_file and os.path.exists(processed_prev_day_file) else pd.DataFrame(columns=['Ticker'])
       
        processed_prev_week_file = os.path.join(data_dir, f"processed_{prefix}{prev_week_ts}.csv") if prev_week_ts else None
        df_processed_prev_week = pd.read_csv(processed_prev_week_file) if processed_prev_week_file and os.path.exists(processed_prev_week_file) else pd.DataFrame(columns=['Ticker'])
       
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
            smi = smi.replace(0, np.nan) # Avoid division by zero
            weights = np.log(1 + ticker_df['Open Interest']) / smi
            valid = (~weights.isna()) & (~ticker_df['IV_mid'].isna())
            if not valid.any():
                return np.nan
            weighted_iv = np.average(ticker_df.loc[valid, 'IV_mid'], weights=weights[valid])
            return weighted_iv
       
        def calculate_rvol_percentile(ticker, current_vol, df_historic, past_year_start, current_dt):
            past_year = df_historic[(df_historic['Ticker'] == ticker) & (df_historic['Date'] >= past_year_start) & (df_historic['Date'] <= current_dt)]
            vols = past_year['Realised_Vol_Close_100'].dropna()
            if vols.empty or len(vols) < 2 or current_vol == 'N/A':
                return 'N/A'
            percentile = (vols < current_vol).sum() / len(vols) * 100
            return percentile
       
        def calculate_rvol_z_score_percentile(ticker, current_vol, df_historic, past_year_start, current_dt):
            past_year = df_historic[(df_historic['Ticker'] == ticker) & (df_historic['Date'] >= past_year_start) & (df_historic['Date'] <= current_dt)]
            vols = past_year['Realised_Vol_Close_100'].dropna()
            if vols.empty or len(vols) < 2 or current_vol == 'N/A':
                return 'N/A'
            mean_vol = vols.mean()
            std_vol = vols.std()
            if std_vol == 0:
                return 'N/A' # Avoid division by zero
            z_score = (current_vol - mean_vol) / std_vol
            percentile = norm.cdf(z_score) * 100
            return percentile
       
        ranking = []
        rvol_types = ['Close_30', 'Close_60', 'Close_100', 'Close_180', 'Close_252']
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
            ticker_processed_prev_day = df_processed_prev_day[df_processed_prev_day['Ticker'] == ticker] if 'Ticker' in df_processed_prev_day.columns else pd.DataFrame()
            weighted_iv_prev_day = calculate_weighted_iv(ticker_processed_prev_day) if not ticker_processed_prev_day.empty else np.nan
            weighted_iv_1d_pct = (weighted_iv - weighted_iv_prev_day) / weighted_iv_prev_day * 100 if not np.isnan(weighted_iv_prev_day) and weighted_iv_prev_day != 0 else 'N/A'
            ticker_processed_prev_week = df_processed_prev_week[df_processed_prev_week['Ticker'] == ticker] if 'Ticker' in df_processed_prev_week.columns else pd.DataFrame()
            weighted_iv_prev_week = calculate_weighted_iv(ticker_processed_prev_week) if not ticker_processed_prev_week.empty else np.nan
            weighted_iv_1w_pct = (weighted_iv - weighted_iv_prev_week) / weighted_iv_prev_week * 100 if not np.isnan(weighted_iv_prev_week) and weighted_iv_prev_week != 0 else 'N/A'
            latest = latest_historic[latest_historic['Ticker'] == ticker]
            rank_dict = {
                'Ticker': ticker,
                'Latest Close': 'N/A',
                'Latest High': 'N/A',
                'Latest Low': 'N/A',
                'Close 1d (%)': 'N/A',
                'Close 1w (%)': 'N/A',
                'High 1d (%)': 'N/A',
                'High 1w (%)': 'N/A',
                'Low 1d (%)': 'N/A',
                'Low 1w (%)': 'N/A',
                'Realised Volatility Close 30d (%)': 'N/A',
                'Realised Volatility Close 60d (%)': 'N/A',
                'Realised Volatility Close 100d (%)': 'N/A',
                'Realised Volatility Close 100d 1d (%)': 'N/A',
                'Realised Volatility Close 100d 1w (%)': 'N/A',
                'Min Realised Volatility Close 100d (1y)': 'N/A',
                'Max Realised Volatility Close 100d (1y)': 'N/A',
                'Mean Realised Volatility Close 100d (1y)': 'N/A',
                'Rvol Close 100d Percentile (%)': 'N/A',
                'Rvol Close 100d Z-Score Percentile (%)': 'N/A',
                'Realised Volatility Close 180d (%)': 'N/A',
                'Realised Volatility Close 252d (%)': 'N/A',
                'Weighted IV (%)': weighted_iv * 100 if not np.isnan(weighted_iv) else 'N/A',
                'Weighted IV 1d (%)': weighted_iv_1d_pct,
                'Weighted IV 1w (%)': weighted_iv_1w_pct,
                'Rvol Close 100d - Weighted IV': 'N/A',
                'Volume': vol,
                'Volume 1d (%)': vol_1d_pct,
                'Volume 1w (%)': vol_1w_pct,
                'Open Interest': oi,
                'OI 1d (%)': oi_1d_pct,
                'OI 1w (%)': oi_1w_pct
            }
            if not latest.empty:
                latest_close = latest['Close'].values[0]
                rank_dict['Latest Close'] = latest_close
                latest_high = latest['High'].values[0]
                rank_dict['Latest High'] = latest_high
                latest_low = latest['Low'].values[0]
                rank_dict['Latest Low'] = latest_low
                prev_close = get_prev_value(ticker, prev_day_dt, 'Close')
                close_1d_pct = (latest_close - prev_close) / prev_close * 100 if prev_close is not None and prev_close != 0 else 'N/A'
                rank_dict['Close 1d (%)'] = close_1d_pct
                prev_week_close = get_prev_value(ticker, prev_week_dt, 'Close')
                close_1w_pct = (latest_close - prev_week_close) / prev_week_close * 100 if prev_week_close is not None and prev_week_close != 0 else 'N/A'
                rank_dict['Close 1w (%)'] = close_1w_pct
                prev_high = get_prev_value(ticker, prev_day_dt, 'High')
                high_1d_pct = (latest_high - prev_high) / prev_high * 100 if prev_high is not None and prev_high != 0 else 'N/A'
                rank_dict['High 1d (%)'] = high_1d_pct
                prev_week_high = get_prev_value(ticker, prev_week_dt, 'High')
                high_1w_pct = (latest_high - prev_week_high) / prev_week_high * 100 if prev_week_high is not None and prev_week_high != 0 else 'N/A'
                rank_dict['High 1w (%)'] = high_1w_pct
                prev_low = get_prev_value(ticker, prev_day_dt, 'Low')
                low_1d_pct = (latest_low - prev_low) / prev_low * 100 if prev_low is not None and prev_low != 0 else 'N/A'
                rank_dict['Low 1d (%)'] = low_1d_pct
                prev_week_low = get_prev_value(ticker, prev_week_dt, 'Low')
                low_1w_pct = (latest_low - prev_week_low) / prev_week_low * 100 if prev_week_low is not None and prev_week_low != 0 else 'N/A'
                rank_dict['Low 1w (%)'] = low_1w_pct
                for rvol_type in rvol_types:
                    col = f'Realised_Vol_{rvol_type}'
                    current_vol = latest[col].values[0] if col in latest.columns else 'N/A'
                    rank_dict[f'Realised Volatility {rvol_type}d (%)'] = current_vol
                    if rvol_type == 'Close_100':
                        prev_day_vol = get_prev_value(ticker, prev_day_dt, col)
                        vol_1d_pct = (current_vol - prev_day_vol) / prev_day_vol * 100 if prev_day_vol is not None and prev_day_vol != 0 else 'N/A'
                        rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = vol_1d_pct
                        prev_week_vol = get_prev_value(ticker, prev_week_dt, col)
                        vol_1w_pct = (current_vol - prev_week_vol) / prev_week_vol * 100 if prev_week_vol is not None and prev_week_vol != 0 else 'N/A'
                        rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = vol_1w_pct
                        past_year = df_historic[(df_historic['Ticker'] == ticker) & (df_historic['Date'] >= past_year_start) & (df_historic['Date'] <= current_dt)]
                        vols = past_year[col].dropna()
                        if not vols.empty:
                            min_vol = vols.min()
                            max_vol = vols.max()
                            mean_vol = vols.mean()
                            percentile = calculate_rvol_percentile(ticker, current_vol, df_historic, past_year_start, current_dt)
                            z_score_percentile = calculate_rvol_z_score_percentile(ticker, current_vol, df_historic, past_year_start, current_dt)
                        else:
                            min_vol = max_vol = mean_vol = percentile = z_score_percentile = 'N/A'
                        rank_dict[f'Min Realised Volatility {rvol_type}d (1y)'] = min_vol
                        rank_dict[f'Max Realised Volatility {rvol_type}d (1y)'] = max_vol
                        rank_dict[f'Mean Realised Volatility {rvol_type}d (1y)'] = mean_vol
                        rank_dict['Rvol Close 100d Percentile (%)'] = percentile
                        rank_dict['Rvol Close 100d Z-Score Percentile (%)'] = z_score_percentile
                        rvol100d_minus_weighted_iv = current_vol - (weighted_iv * 100) if current_vol != 'N/A' and not np.isnan(weighted_iv) else 'N/A'
                        rank_dict['Rvol Close 100d - Weighted IV'] = rvol100d_minus_weighted_iv
            ranking.append(rank_dict)
       
        column_order = [
            'Ticker',
            'Latest Close',
            'Latest High',
            'Latest Low',
            'Close 1d (%)',
            'Close 1w (%)',
            'High 1d (%)',
            'High 1w (%)',
            'Low 1d (%)',
            'Low 1w (%)',
            'Realised Volatility Close 30d (%)',
            'Realised Volatility Close 60d (%)',
            'Realised Volatility Close 100d (%)',
            'Realised Volatility Close 100d 1d (%)',
            'Realised Volatility Close 100d 1w (%)',
            'Min Realised Volatility Close 100d (1y)',
            'Max Realised Volatility Close 100d (1y)',
            'Mean Realised Volatility Close 100d (1y)',
            'Rvol Close 100d Percentile (%)',
            'Rvol Close 100d Z-Score Percentile (%)',
            'Realised Volatility Close 180d (%)',
            'Realised Volatility Close 252d (%)',
            'Weighted IV (%)',
            'Weighted IV 1d (%)',
            'Weighted IV 1w (%)',
            'Rvol Close 100d - Weighted IV',
            'Volume',
            'Volume 1d (%)',
            'Volume 1w (%)',
            'Open Interest',
            'OI 1d (%)',
            'OI 1w (%)'
        ]
        df_ranking = pd.DataFrame(ranking)
        df_ranking = df_ranking[column_order]
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
