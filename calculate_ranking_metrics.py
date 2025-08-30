import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from scipy.stats import norm
import glob

def load_rvol_from_historic(ticker, timestamp, days):
    historic_file = f'data/{timestamp}/historic/historic_{ticker}.csv'
    if not os.path.exists(historic_file):
        return None
    df_hist = pd.read_csv(historic_file, parse_dates=['Date'])
    if df_hist.empty or 'Ticker' not in df_hist.columns:
        return None
    col = f'Realised_Vol_Close_{days}'
    if col not in df_hist.columns:
        return None
    latest_vol = df_hist[col].iloc[-1]
    return latest_vol if not pd.isna(latest_vol) else None

def load_historic_data(ts):
    if ts is None:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Close', 'High', 'Low', 'Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100', 'Realised_Vol_Close_180', 'Realised_Vol_Close_252'])
    historic_dir = f'data/{ts}/historic'
    if not os.path.exists(historic_dir):
        return pd.DataFrame(columns=['Date', 'Ticker', 'Close', 'High', 'Low', 'Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100', 'Realised_Vol_Close_180', 'Realised_Vol_Close_252'])
    historic_files = glob.glob(f'{historic_dir}/historic_*.csv')
    dfs = []
    required_columns = ['Date', 'Ticker', 'Close', 'High', 'Low', 'Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100', 'Realised_Vol_Close_180', 'Realised_Vol_Close_252']
    for file in historic_files:
        try:
            df = pd.read_csv(file, parse_dates=['Date'])
            if 'Ticker' not in df.columns or df.empty:
                continue
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                continue
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=required_columns)
    df_concat = pd.concat(dfs, ignore_index=True)
    return df_concat if not df_concat.empty else pd.DataFrame(columns=required_columns)

def calculate_ranking_metrics(timestamp, sources, data_dir='data'):
    dates_file = os.path.join(data_dir, 'dates.json')
    try:
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    except Exception:
        return
    timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
    if timestamp not in timestamps:
        return
    sources = ['yfinance']
    for source in sources:
        prefix = '_yfinance' if source == 'yfinance' else ''
        current_index = timestamps.index(timestamp)
        current_dt = datetime.strptime(timestamp[:8], "%Y%m%d")
        past_year_start = current_dt - timedelta(days=365)
        prev_day_ts = None
        for ts in timestamps[:current_index][::-1]:
            ts_dt = datetime.strptime(ts[:8], "%Y%m%d")
            if ts_dt <= current_dt - timedelta(days=1):
                prev_day_ts = ts
                break
        prev_week_ts = None
        for ts in timestamps[:current_index][::-1]:
            ts_dt = datetime.strptime(ts[:8], "%Y%m%d")
            if ts_dt <= current_dt - timedelta(days=7):
                prev_week_ts = ts
                break
        def get_option_totals(ts, prefix):
            if ts is None:
                return pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
            raw_dir = f'data/{ts}/raw{prefix}'
            if not os.path.exists(raw_dir):
                return pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
            raw_files = glob.glob(f'{raw_dir}/raw{prefix}_*.csv')
            totals = []
            for file in raw_files:
                ticker = os.path.basename(file).split(f'raw{prefix}_')[1].split('.csv')[0]
                try:
                    df = pd.read_csv(file)
                    oi_sum = df['Open Interest'].sum() if 'Open Interest' in df.columns else 0
                    vol_sum = df['Volume'].sum() if 'Volume' in df.columns else 0
                    totals.append({'Ticker': ticker, 'OI': oi_sum, 'Vol': vol_sum})
                except Exception:
                    continue
            return pd.DataFrame(totals)
        current_option = get_option_totals(timestamp, prefix)
        prev_day_option = get_option_totals(prev_day_ts, prefix)
        prev_week_option = get_option_totals(prev_week_ts, prefix)
        df_historic = load_historic_data(timestamp)
        def load_processed_data(ts, prefix):
            processed_dir = f'data/{ts}/processed{prefix}'
            if not os.path.exists(processed_dir):
                return pd.DataFrame()
            processed_files = glob.glob(f'{processed_dir}/processed{prefix}_*.csv')
            dfs = []
            for file in processed_files:
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception:
                    continue
            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs, ignore_index=True)
        df_processed = load_processed_data(timestamp, prefix)
        if df_processed.empty:
            continue
        processed_prev_day = load_processed_data(prev_day_ts, prefix) if prev_day_ts else pd.DataFrame(columns=['Ticker'])
        processed_prev_week = load_processed_data(prev_week_ts, prefix) if prev_week_ts else pd.DataFrame(columns=['Ticker'])
        latest_historic = df_historic.loc[df_historic.groupby('Ticker')['Date'].idxmax()] if not df_historic.empty and 'Ticker' in df_historic.columns else pd.DataFrame()
        prev_day_historic = load_historic_data(prev_day_ts)
        prev_day_historic_latest = prev_day_historic.loc[prev_day_historic.groupby('Ticker')['Date'].idxmax()] if not prev_day_historic.empty and 'Ticker' in prev_day_historic.columns else pd.DataFrame()
        prev_week_historic = load_historic_data(prev_week_ts)
        prev_week_historic_latest = prev_week_historic.loc[prev_week_historic.groupby('Ticker')['Date'].idxmax()] if not prev_week_historic.empty and 'Ticker' in prev_week_historic.columns else pd.DataFrame()
        def get_prev_value(ticker, target_date, col):
            g = df_historic[df_historic['Ticker'] == ticker]
            prev = g[g['Date'] <= target_date]
            if prev.empty or col not in prev.columns:
                return None
            return prev.loc[prev['Date'].idxmax(), col]
        def filter_3m_expiry(ticker_df, current_dt):
            if 'Expiry' not in ticker_df.columns:
                return pd.DataFrame()
            try:
                ticker_df = ticker_df.copy()
                ticker_df.loc[:, 'Expiry'] = pd.to_datetime(ticker_df['Expiry'])
                days_to_expiry = (ticker_df['Expiry'] - current_dt).dt.days
                valid_expiry = ticker_df.loc[(days_to_expiry >= 75) & (days_to_expiry <= 105)]
                if valid_expiry.empty:
                    return pd.DataFrame()
                valid_expiry.loc[:, 'Days_Diff'] = abs(days_to_expiry.loc[valid_expiry.index] - 90)
                closest_expiry = valid_expiry.loc[valid_expiry['Days_Diff'].idxmin(), 'Expiry']
                return valid_expiry[valid_expiry['Expiry'] == closest_expiry]
            except Exception:
                return pd.DataFrame()
        def calculate_weighted_iv(ticker_df, expiry_filter=False, current_dt=None):
            if expiry_filter and current_dt is not None:
                ticker_df = filter_3m_expiry(ticker_df, current_dt)
            if ticker_df.empty or 'IV_mid' not in ticker_df.columns or ticker_df['IV_mid'].isna().all():
                return np.nan
            mid = (ticker_df['Bid'] + ticker_df['Ask']) / 2
            spread = ticker_df['Ask'] - ticker_df['Bid']
            smi = 100 * (spread / mid)
            smi = smi.replace(0, np.nan)
            weights = np.log(1 + ticker_df['Open Interest']) / smi
            valid = (~weights.isna()) & (~ticker_df['IV_mid'].isna())
            if not valid.any():
                return np.nan
            weighted_iv = np.average(ticker_df.loc[valid, 'IV_mid'], weights=weights[valid])
            return weighted_iv
        def calculate_atm_iv(ticker_df, current_price, current_dt):
            ticker_df_3m = filter_3m_expiry(ticker_df, current_dt)
            if ticker_df_3m.empty or 'Strike' not in ticker_df_3m.columns or 'IV_mid' not in ticker_df_3m.columns:
                return np.nan
            if pd.isna(current_price) or current_price == 'N/A':
                return np.nan
            ticker_df_3m.loc[:, 'Strike_Diff'] = abs(ticker_df_3m['Strike'] - current_price)
            atm_option = ticker_df_3m.loc[ticker_df_3m['Strike_Diff'].idxmin()]
            return atm_option['IV_mid'] if not pd.isna(atm_option['IV_mid']) else np.nan
        def calculate_rvol_percentile(ticker, rvol, historic_data, start_date, end_date):
            past_year = historic_data[(historic_data['Ticker'] == ticker) & (historic_data['Date'] >= start_date) & (historic_data['Date'] <= end_date)]
            vols = past_year['Realised_Vol_Close_100'].dropna() if 'Realised_Vol_Close_100' in past_year.columns else pd.Series()
            if vols.empty or rvol is None or pd.isna(rvol):
                return np.nan
            return (vols < rvol).sum() / len(vols) * 100
        def calculate_rvol_z_score_percentile(ticker, rvol, historic_data, start_date, end_date):
            past_year = historic_data[(historic_data['Ticker'] == ticker) & (historic_data['Date'] >= start_date) & (historic_data['Date'] <= end_date)]
            vols = past_year['Realised_Vol_Close_100'].dropna() if 'Realised_Vol_Close_100' in past_year.columns else pd.Series()
            if vols.empty or rvol is None or pd.isna(rvol):
                return np.nan
            if vols.std() == 0:
                return np.nan
            z_score = (rvol - vols.mean()) / vols.std()
            return norm.cdf(z_score) * 100
        ranking = []
        tickers = set(df_processed['Ticker'].unique()) | set(latest_historic['Ticker'].unique()) if not df_processed.empty and not latest_historic.empty else set()
        rvol_types = ['30', '60', '100', '180', '252']
        for ticker in tickers:
            rank_dict = {'Ticker': ticker}
            ticker_processed = df_processed[df_processed['Ticker'] == ticker] if not df_processed.empty else pd.DataFrame()
            weighted_iv = calculate_weighted_iv(ticker_processed) * 100 if not ticker_processed.empty else np.nan
            weighted_iv_3m = calculate_weighted_iv(ticker_processed, expiry_filter=True, current_dt=current_dt) * 100 if not ticker_processed.empty else np.nan
            prev_day_weighted_iv = calculate_weighted_iv(processed_prev_day[processed_prev_day['Ticker'] == ticker]) * 100 if not processed_prev_day.empty else np.nan
            prev_week_weighted_iv = calculate_weighted_iv(processed_prev_week[processed_prev_week['Ticker'] == ticker]) * 100 if not processed_prev_week.empty else np.nan
            rank_dict['Weighted IV (%)'] = weighted_iv if not np.isnan(weighted_iv) else np.nan
            rank_dict['Weighted IV 3m (%)'] = weighted_iv_3m if not np.isnan(weighted_iv_3m) else np.nan
            rank_dict['Weighted IV 1d (%)'] = (weighted_iv - prev_day_weighted_iv) / prev_day_weighted_iv * 100 if not np.isnan(weighted_iv) and not np.isnan(prev_day_weighted_iv) and prev_day_weighted_iv != 0 else np.nan
            rank_dict['Weighted IV 1w (%)'] = (weighted_iv - prev_week_weighted_iv) / prev_week_weighted_iv * 100 if not np.isnan(weighted_iv) and not np.isnan(prev_week_weighted_iv) and prev_week_weighted_iv != 0 else np.nan
            ticker_option = current_option[current_option['Ticker'] == ticker] if not current_option.empty else pd.DataFrame()
            ticker_prev_day_option = prev_day_option[prev_day_option['Ticker'] == ticker] if not prev_day_option.empty else pd.DataFrame()
            ticker_prev_week_option = prev_week_option[prev_week_option['Ticker'] == ticker] if not prev_week_option.empty else pd.DataFrame()
            rank_dict['Volume'] = ticker_option['Vol'].iloc[0] if not ticker_option.empty else 0
            rank_dict['Open Interest'] = ticker_option['OI'].iloc[0] if not ticker_option.empty else 0
            prev_day_vol = ticker_prev_day_option['Vol'].iloc[0] if not ticker_prev_day_option.empty else 0
            prev_week_vol = ticker_prev_week_option['Vol'].iloc[0] if not ticker_prev_week_option.empty else 0
            prev_day_oi = ticker_prev_day_option['OI'].iloc[0] if not ticker_prev_day_option.empty else 0
            prev_week_oi = ticker_prev_week_option['OI'].iloc[0] if not ticker_prev_week_option.empty else 0
            rank_dict['Volume 1d (%)'] = (rank_dict['Volume'] - prev_day_vol) / prev_day_vol * 100 if prev_day_vol != 0 else np.nan
            rank_dict['Volume 1w (%)'] = (rank_dict['Volume'] - prev_week_vol) / prev_week_vol * 100 if prev_week_vol != 0 else np.nan
            rank_dict['OI 1d (%)'] = (rank_dict['Open Interest'] - prev_day_oi) / prev_day_oi * 100 if prev_day_oi != 0 else np.nan
            rank_dict['OI 1w (%)'] = (rank_dict['Open Interest'] - prev_week_oi) / prev_week_oi * 100 if prev_week_oi != 0 else np.nan
            if not latest_historic.empty and ticker in latest_historic['Ticker'].values:
                ticker_historic = latest_historic[latest_historic['Ticker'] == ticker]
                rank_dict['Latest Close'] = ticker_historic['Close'].iloc[0] if not ticker_historic.empty else np.nan
                rank_dict['Latest High'] = ticker_historic['High'].iloc[0] if not ticker_historic.empty else np.nan
                rank_dict['Latest Low'] = ticker_historic['Low'].iloc[0] if not ticker_historic.empty else np.nan
                current_price = ticker_historic['Close'].iloc[0] if not ticker_historic.empty else np.nan
                prev_day_close = prev_day_historic_latest['Close'].iloc[0] if not prev_day_historic_latest.empty and ticker in prev_day_historic_latest['Ticker'].values else None
                prev_week_close = prev_week_historic_latest['Close'].iloc[0] if not prev_week_historic_latest.empty and ticker in prev_week_historic_latest['Ticker'].values else None
                prev_day_high = prev_day_historic_latest['High'].iloc[0] if not prev_day_historic_latest.empty and ticker in prev_day_historic_latest['Ticker'].values else None
                prev_week_high = prev_week_historic_latest['High'].iloc[0] if not prev_week_historic_latest.empty and ticker in prev_week_historic_latest['Ticker'].values else None
                prev_day_low = prev_day_historic_latest['Low'].iloc[0] if not prev_day_historic_latest.empty and ticker in prev_day_historic_latest['Ticker'].values else None
                prev_week_low = prev_week_historic_latest['Low'].iloc[0] if not prev_week_historic_latest.empty and ticker in prev_week_historic_latest['Ticker'].values else None
                rank_dict['Close 1d (%)'] = (rank_dict['Latest Close'] - prev_day_close) / prev_day_close * 100 if prev_day_close is not None and prev_day_close != 0 and not np.isnan(rank_dict['Latest Close']) else np.nan
                rank_dict['Close 1w (%)'] = (rank_dict['Latest Close'] - prev_week_close) / prev_week_close * 100 if prev_week_close is not None and prev_week_close != 0 and not np.isnan(rank_dict['Latest Close']) else np.nan
                rank_dict['High 1d (%)'] = (rank_dict['Latest High'] - prev_day_high) / prev_day_high * 100 if prev_day_high is not None and prev_day_high != 0 and not np.isnan(rank_dict['Latest High']) else np.nan
                rank_dict['High 1w (%)'] = (rank_dict['Latest High'] - prev_week_high) / prev_week_high * 100 if prev_week_high is not None and prev_week_high != 0 and not np.isnan(rank_dict['Latest High']) else np.nan
                rank_dict['Low 1d (%)'] = (rank_dict['Latest Low'] - prev_day_low) / prev_day_low * 100 if prev_day_low is not None and prev_day_low != 0 and not np.isnan(rank_dict['Latest Low']) else np.nan
                rank_dict['Low 1w (%)'] = (rank_dict['Latest Low'] - prev_week_low) / prev_week_low * 100 if prev_week_low is not None and prev_week_low != 0 and not np.isnan(rank_dict['Latest Low']) else np.nan
                atm_iv_3m = calculate_atm_iv(ticker_processed, current_price, current_dt) * 100 if not ticker_processed.empty and not np.isnan(current_price) else np.nan
                rank_dict['ATM IV 3m (%)'] = atm_iv_3m if not np.isnan(atm_iv_3m) else np.nan
                rank_dict['ATM IV 3m 1d (%)'] = np.nan
                rank_dict['ATM IV 3m 1w (%)'] = np.nan
                for rvol_type in rvol_types:
                    current_vol = load_rvol_from_historic(ticker, timestamp, rvol_type)
                    rank_dict[f'Realised Volatility {rvol_type}d (%)'] = current_vol if current_vol is not None else np.nan
                    if rvol_type == '100':
                        prev_day_vol = load_rvol_from_historic(ticker, prev_day_ts, 100) if prev_day_ts else None
                        vol_1d_pct = (current_vol - prev_day_vol) / prev_day_vol * 100 if prev_day_vol is not None and prev_day_vol != 0 and current_vol is not None else np.nan
                        rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = vol_1d_pct
                        prev_week_vol = load_rvol_from_historic(ticker, prev_week_ts, 100) if prev_week_ts else None
                        vol_1w_pct = (current_vol - prev_week_vol) / prev_week_vol * 100 if prev_week_vol is not None and prev_week_vol != 0 and current_vol is not None else np.nan
                        rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = vol_1w_pct
                        past_year = df_historic[(df_historic['Ticker'] == ticker) & (df_historic['Date'] >= past_year_start) & (df_historic['Date'] <= current_dt)]
                        vols = past_year['Realised_Vol_Close_100'].dropna() if 'Realised_Vol_Close_100' in past_year.columns else pd.Series()
                        if not vols.empty:
                            min_vol = vols.min()
                            max_vol = vols.max()
                            mean_vol = vols.mean()
                            percentile = calculate_rvol_percentile(ticker, current_vol, df_historic, past_year_start, current_dt)
                            z_score_percentile = calculate_rvol_z_score_percentile(ticker, current_vol, df_historic, past_year_start, current_dt)
                        else:
                            min_vol = max_vol = mean_vol = percentile = z_score_percentile = np.nan
                        rank_dict[f'Min Realised Volatility {rvol_type}d (1y)'] = min_vol if rvol_type == '100' else np.nan
                        rank_dict[f'Max Realised Volatility {rvol_type}d (1y)'] = max_vol if rvol_type == '100' else np.nan
                        rank_dict[f'Mean Realised Volatility {rvol_type}d (1y)'] = mean_vol if rvol_type == '100' else np.nan
                        rank_dict['Rvol 100d Percentile (%)'] = percentile if rvol_type == '100' else np.nan
                        rank_dict['Rvol 100d Z-Score Percentile (%)'] = z_score_percentile if rvol_type == '100' else np.nan
                        if rvol_type == '100':
                            rvol100d_minus_weighted_iv = current_vol - weighted_iv if current_vol is not None and not np.isnan(weighted_iv) else float('-inf')
                            rank_dict['Rvol100d - Weighted IV'] = rvol100d_minus_weighted_iv
            else:
                current_price = np.nan
                rank_dict['Latest Close'] = np.nan
                rank_dict['Latest High'] = np.nan
                rank_dict['Latest Low'] = np.nan
                rank_dict['Close 1d (%)'] = np.nan
                rank_dict['Close 1w (%)'] = np.nan
                rank_dict['High 1d (%)'] = np.nan
                rank_dict['High 1w (%)'] = np.nan
                rank_dict['Low 1d (%)'] = np.nan
                rank_dict['Low 1w (%)'] = np.nan
                rank_dict['ATM IV 3m (%)'] = np.nan
                rank_dict['ATM IV 3m 1d (%)'] = np.nan
                rank_dict['ATM IV 3m 1w (%)'] = np.nan
                for rvol_type in rvol_types:
                    current_vol = load_rvol_from_historic(ticker, timestamp, rvol_type)
                    rank_dict[f'Realised Volatility {rvol_type}d (%)'] = current_vol if current_vol is not None else np.nan
                    if rvol_type == '100':
                        prev_day_vol = load_rvol_from_historic(ticker, prev_day_ts, 100) if prev_day_ts else None
                        vol_1d_pct = (current_vol - prev_day_vol) / prev_day_vol * 100 if prev_day_vol is not None and prev_day_vol != 0 and current_vol is not None else np.nan
                        rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = vol_1d_pct
                        prev_week_vol = load_rvol_from_historic(ticker, prev_week_ts, 100) if prev_week_ts else None
                        vol_1w_pct = (current_vol - prev_week_vol) / prev_week_vol * 100 if prev_week_vol is not None and prev_week_vol != 0 and current_vol is not None else np.nan
                        rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = vol_1w_pct
                        past_year = df_historic[(df_historic['Ticker'] == ticker) & (df_historic['Date'] >= past_year_start) & (df_historic['Date'] <= current_dt)]
                        vols = past_year['Realised_Vol_Close_100'].dropna() if 'Realised_Vol_Close_100' in past_year.columns else pd.Series()
                        if not vols.empty:
                            min_vol = vols.min()
                            max_vol = vols.max()
                            mean_vol = vols.mean()
                            percentile = calculate_rvol_percentile(ticker, current_vol, df_historic, past_year_start, current_dt)
                            z_score_percentile = calculate_rvol_z_score_percentile(ticker, current_vol, df_historic, past_year_start, current_dt)
                        else:
                            min_vol = max_vol = mean_vol = percentile = z_score_percentile = np.nan
                        rank_dict[f'Min Realised Volatility {rvol_type}d (1y)'] = min_vol if rvol_type == '100' else np.nan
                        rank_dict[f'Max Realised Volatility {rvol_type}d (1y)'] = max_vol if rvol_type == '100' else np.nan
                        rank_dict[f'Mean Realised Volatility {rvol_type}d (1y)'] = mean_vol if rvol_type == '100' else np.nan
                        rank_dict['Rvol 100d Percentile (%)'] = percentile if rvol_type == '100' else np.nan
                        rank_dict['Rvol 100d Z-Score Percentile (%)'] = z_score_percentile if rvol_type == '100' else np.nan
                        if rvol_type == '100':
                            rvol100d_minus_weighted_iv = current_vol - weighted_iv if current_vol is not None and not np.isnan(weighted_iv) else float('-inf')
                            rank_dict['Rvol100d - Weighted IV'] = rvol100d_minus_weighted_iv
            ranking.append(rank_dict)
        def sort_key(item):
            value = item.get('Rvol100d - Weighted IV', float('-inf'))
            return value if isinstance(value, (int, float)) else float('-inf')
        ranking = sorted(ranking, key=sort_key, reverse=True)
        for i, rank_dict in enumerate(ranking):
            rank_dict['Rank'] = i + 1
        column_order = [
            'Rank', 'Ticker', 'Latest Close', 'Latest High', 'Latest Low',
            'Close 1d (%)', 'Close 1w (%)', 'High 1d (%)', 'High 1w (%)', 'Low 1d (%)', 'Low 1w (%)',
            'Realised Volatility 30d (%)', 'Realised Volatility 60d (%)', 'Realised Volatility 100d (%)',
            'Realised Volatility 100d 1d (%)', 'Realised Volatility 100d 1w (%)',
            'Min Realised Volatility 100d (1y)', 'Max Realised Volatility 100d (1y)', 'Mean Realised Volatility 100d (1y)',
            'Rvol 100d Percentile (%)', 'Rvol 100d Z-Score Percentile (%)',
            'Realised Volatility 180d (%)', 'Realised Volatility 252d (%)',
            'Weighted IV (%)', 'Weighted IV 3m (%)', 'Weighted IV 3m 1d (%)', 'Weighted IV 3m 1w (%)',
            'ATM IV 3m (%)', 'ATM IV 3m 1d (%)', 'ATM IV 3m 1w (%)',
            'Weighted IV 1d (%)', 'Weighted IV 1w (%)', 'Rvol100d - Weighted IV',
            'Volume', 'Volume 1d (%)', 'Volume 1w (%)', 'Open Interest', 'OI 1d (%)', 'OI 1w (%)'
        ]
        df_ranking = pd.DataFrame(ranking)
        ranking_dir = f'data/{timestamp}/ranking'
        os.makedirs(ranking_dir, exist_ok=True)
        output_file = f'{ranking_dir}/ranking{prefix}.csv'
        df_ranking.to_csv(output_file, index=False)

def main():
    data_dir = 'data'
    dates_file = os.path.join(data_dir, 'dates.json')
    try:
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    except Exception:
        return
    timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
        sources = ['yfinance']
        calculate_ranking_metrics(timestamp, sources)
    else:
        sources = ['yfinance']
        for timestamp in timestamps:
            calculate_ranking_metrics(timestamp, sources)

main()
