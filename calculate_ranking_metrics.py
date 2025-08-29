import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from scipy.stats import norm
import glob
import yfinance as yf

def calculate_rvol_days(ticker, days):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < days + 1:
            print(f"No sufficient historical data for {ticker} for {days} days")
            return None
        hist_last = hist.iloc[-(days + 1):]
        log_returns = np.log(hist_last["Close"] / hist_last["Close"].shift(1)).dropna()
        if len(log_returns) < 2:
            print(f"Insufficient returns data for {ticker} for {days} days")
            return None
        realised_vol = np.std(log_returns, ddof=1) * np.sqrt(252) * 100  # Convert to percentage
        return realised_vol
    except Exception as e:
        print(f"Error calculating RVOL for {ticker}: {e}")
        return None

def calculate_ranking_metrics(timestamp, sources, data_dir='data'):
    dates_file = os.path.join(data_dir, 'dates.json')
    
    try:
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    except Exception as e:
        print(f"Error loading dates.json: {e}")
        return
    
    timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
    
    if timestamp not in timestamps:
        print(f"Timestamp {timestamp} not in dates.json")
        return
    
    # Process only yfinance since nasdaq directories are missing
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
                print(f"Raw directory {raw_dir} not found")
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
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            return pd.DataFrame(totals)
        
        current_option = get_option_totals(timestamp, prefix)
        prev_day_option = get_option_totals(prev_day_ts, prefix)
        prev_week_option = get_option_totals(prev_week_ts, prefix)
        
        def load_historic_data(ts):
            historic_dir = f'data/{ts}/historic'
            if not os.path.exists(historic_dir):
                print(f"Historic directory {historic_dir} not found")
                return pd.DataFrame()
            historic_files = glob.glob(f'{historic_dir}/historic_*.csv')
            dfs = []
            required_columns = ['Date', 'Ticker', 'Close', 'High', 'Low', 'Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100', 'Realised_Vol_Close_180', 'Realised_Vol_Close_252']
            for file in historic_files:
                try:
                    df = pd.read_csv(file, parse_dates=['Date'])
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    if missing_cols:
                        print(f"Missing columns {missing_cols} in {file}")
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            if not dfs:
                print(f"No historic data files found in {historic_dir}")
                return pd.DataFrame()
            df_concat = pd.concat(dfs, ignore_index=True)
            return df_concat if not df_concat.empty else pd.DataFrame(columns=required_columns)
        
        df_historic = load_historic_data(timestamp)
        if df_historic.empty:
            print(f"No historic data found for {timestamp}")
        
        def load_processed_data(ts, prefix):
            processed_dir = f'data/{ts}/processed{prefix}'
            if not os.path.exists(processed_dir):
                print(f"Processed directory {processed_dir} not found")
                return pd.DataFrame()
            processed_files = glob.glob(f'{processed_dir}/processed{prefix}_*.csv')
            dfs = []
            for file in processed_files:
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            if not dfs:
                print(f"No processed data files found in {processed_dir}")
                return pd.DataFrame()
            return pd.concat(dfs, ignore_index=True)
        
        df_processed = load_processed_data(timestamp, prefix)
        if df_processed.empty:
            print(f"No processed data found for {timestamp}")
            continue
        
        processed_prev_day = load_processed_data(prev_day_ts, prefix) if prev_day_ts else pd.DataFrame(columns=['Ticker'])
        processed_prev_week = load_processed_data(prev_week_ts, prefix) if prev_week_ts else pd.DataFrame(columns=['Ticker'])
        
        prev_day_historic = load_historic_data(prev_day_ts) if prev_day_ts else pd.DataFrame()
        prev_week_historic = load_historic_data(prev_week_ts) if prev_week_ts else pd.DataFrame()
        
        latest_historic = df_historic.loc[df_historic.groupby('Ticker')['Date'].idxmax()] if not df_historic.empty else pd.DataFrame()
        prev_day_historic_latest = prev_day_historic.loc[prev_day_historic.groupby('Ticker')['Date'].idxmax()] if not prev_day_historic.empty else pd.DataFrame()
        prev_week_historic_latest = prev_week_historic.loc[prev_week_historic.groupby('Ticker')['Date'].idxmax()] if not prev_week_historic.empty else pd.DataFrame()
        
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
                ticker_df['Expiry'] = pd.to_datetime(ticker_df['Expiry'])
                days_to_expiry = (ticker_df['Expiry'] - current_dt).dt.days
                # Target 3-month expiry (approx 90 days, allowing range 75-105 days)
                valid_expiry = ticker_df[(days_to_expiry >= 75) & (days_to_expiry <= 105)]
                if valid_expiry.empty:
                    return pd.DataFrame()
                # Select the expiry closest to 90 days
                valid_expiry['Days_Diff'] = abs(days_to_expiry - 90)
                closest_expiry = valid_expiry.loc[valid_expiry['Days_Diff'].idxmin(), 'Expiry']
                return valid_expiry[valid_expiry['Expiry'] == closest_expiry]
            except Exception as e:
                print(f"Error filtering 3-month expiry: {e}")
                return pd.DataFrame()
        
        def calculate_weighted_iv(ticker_df, expiry_filter=False, current_dt=None):
            if expiry_filter and current_dt is not None:
                ticker_df = filter_3m_expiry(ticker_df, current_dt)
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
        
        def calculate_atm_iv(ticker_df, current_price, current_dt):
            ticker_df_3m = filter_3m_expiry(ticker_df, current_dt)
            if ticker_df_3m.empty or 'Strike' not in ticker_df_3m.columns or 'IV_mid' not in ticker_df_3m.columns:
                return np.nan
            if pd.isna(current_price) or current_price == 'N/A':
                return np.nan
            # Find the strike closest to the current price
            ticker_df_3m['Strike_Diff'] = abs(ticker_df_3m['Strike'] - current_price)
            atm_option = ticker_df_3m.loc[ticker_df_3m['Strike_Diff'].idxmin()]
            return atm_option['IV_mid'] if not pd.isna(atm_option['IV_mid']) else np.nan
        
        def calculate_rvol_percentile(ticker, current_vol, df_historic, past_year_start, current_dt):
            past_year = df_historic[(df_historic['Ticker'] == ticker) & (df_historic['Date'] >= past_year_start) & (df_historic['Date'] <= current_dt)]
            cols = ['Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100', 'Realised_Vol_Close_180', 'Realised_Vol_Close_252']
            for c in cols:
                if c in past_year.columns:
                    vols = past_year[c].dropna()
                    if not vols.empty and len(vols) >= 2 and current_vol != 'N/A':
                        percentile = (vols < current_vol).sum() / len(vols) * 100
                        return percentile
            # Fallback to yfinance
            if current_vol != 'N/A':
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                if not hist.empty:
                    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                    if len(log_returns) >= 2:
                        vols = log_returns.rolling(window=100).std() * np.sqrt(252) * 100
                        vols = vols.dropna()
                        if not vols.empty and len(vols) >= 2:
                            percentile = (vols < current_vol).sum() / len(vols) * 100
                            return percentile
            return 'N/A'
        
        def calculate_rvol_z_score_percentile(ticker, current_vol, df_historic, past_year_start, current_dt):
            past_year = df_historic[(df_historic['Ticker'] == ticker) & (df_historic['Date'] >= past_year_start) & (df_historic['Date'] <= current_dt)]
            cols = ['Realised_Vol_Close_30', 'Realised_Vol_Close_60', 'Realised_Vol_Close_100', 'Realised_Vol_Close_180', 'Realised_Vol_Close_252']
            for c in cols:
                if c in past_year.columns:
                    vols = past_year[c].dropna()
                    if not vols.empty and len(vols) >= 2 and current_vol != 'N/A':
                        mean_vol = vols.mean()
                        std_vol = vols.std()
                        if std_vol == 0:
                            return 'N/A'
                        z_score = (current_vol - mean_vol) / std_vol
                        percentile = norm.cdf(z_score) * 100
                        return percentile
            # Fallback to yfinance
            if current_vol != 'N/A':
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                if not hist.empty:
                    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                    if len(log_returns) >= 2:
                        vols = log_returns.rolling(window=100).std() * np.sqrt(252) * 100
                        vols = vols.dropna()
                        if not vols.empty and len(vols) >= 2:
                            mean_vol = vols.mean()
                            std_vol = vols.std()
                            if std_vol == 0:
                                return 'N/A'
                            z_score = (current_vol - mean_vol) / std_vol
                            percentile = norm.cdf(z_score) * 100
                            return percentile
            return 'N/A'
        
        ranking = []
        rvol_types = ['30', '60', '100', '180', '252']
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
            weighted_iv_3m = calculate_weighted_iv(ticker_processed, expiry_filter=True, current_dt=current_dt) if not ticker_processed.empty else np.nan
            ticker_processed_prev_day = processed_prev_day[processed_prev_day['Ticker'] == ticker] if 'Ticker' in processed_prev_day.columns else pd.DataFrame()
            ticker_processed_prev_week = processed_prev_week[processed_prev_week['Ticker'] == ticker] if 'Ticker' in processed_prev_week.columns else pd.DataFrame()
            weighted_iv_prev_day = calculate_weighted_iv(ticker_processed_prev_day) if not ticker_processed_prev_day.empty else np.nan
            weighted_iv_3m_prev_day = calculate_weighted_iv(ticker_processed_prev_day, expiry_filter=True, current_dt=current_dt - timedelta(days=1)) if not ticker_processed_prev_day.empty else np.nan
            weighted_iv_prev_week = calculate_weighted_iv(ticker_processed_prev_week) if not ticker_processed_prev_week.empty else np.nan
            weighted_iv_3m_prev_week = calculate_weighted_iv(ticker_processed_prev_week, expiry_filter=True, current_dt=current_dt - timedelta(days=7)) if not ticker_processed_prev_week.empty else np.nan
            weighted_iv_1d_pct = (weighted_iv - weighted_iv_prev_day) / weighted_iv_prev_day * 100 if not np.isnan(weighted_iv_prev_day) and weighted_iv_prev_day != 0 else 'N/A'
            weighted_iv_1w_pct = (weighted_iv - weighted_iv_prev_week) / weighted_iv_prev_week * 100 if not np.isnan(weighted_iv_prev_week) and weighted_iv_prev_week != 0 else 'N/A'
            weighted_iv_3m_1d_pct = (weighted_iv_3m - weighted_iv_3m_prev_day) / weighted_iv_3m_prev_day * 100 if not np.isnan(weighted_iv_3m_prev_day) and weighted_iv_3m_prev_day != 0 else 'N/A'
            weighted_iv_3m_1w_pct = (weighted_iv_3m - weighted_iv_3m_prev_week) / weighted_iv_3m_prev_week * 100 if not np.isnan(weighted_iv_3m_prev_week) and weighted_iv_3m_prev_week != 0 else 'N/A'
            latest = latest_historic[latest_historic['Ticker'] == ticker] if not latest_historic.empty else pd.DataFrame()
            current_price = latest['Close'].values[0] if not latest.empty and 'Close' in latest.columns else 'N/A'
            prev_day_price = prev_day_historic_latest['Close'].values[0] if not prev_day_historic_latest.empty and prev_day_historic_latest['Ticker'].eq(ticker).any() and 'Close' in prev_day_historic_latest.columns else 'N/A'
            prev_week_price = prev_week_historic_latest['Close'].values[0] if not prev_week_historic_latest.empty and prev_week_historic_latest['Ticker'].eq(ticker).any() and 'Close' in prev_week_historic_latest.columns else 'N/A'
            atm_iv_3m = calculate_atm_iv(ticker_processed, current_price, current_dt) if not ticker_processed.empty else np.nan
            atm_iv_3m_prev_day = calculate_atm_iv(ticker_processed_prev_day, prev_day_price, current_dt - timedelta(days=1)) if not ticker_processed_prev_day.empty else np.nan
            atm_iv_3m_prev_week = calculate_atm_iv(ticker_processed_prev_week, prev_week_price, current_dt - timedelta(days=7)) if not ticker_processed_prev_week.empty else np.nan
            atm_iv_3m_1d_pct = (atm_iv_3m - atm_iv_3m_prev_day) / atm_iv_3m_prev_day * 100 if not np.isnan(atm_iv_3m_prev_day) and atm_iv_3m_prev_day != 0 else 'N/A'
            atm_iv_3m_1w_pct = (atm_iv_3m - atm_iv_3m_prev_week) / atm_iv_3m_prev_week * 100 if not np.isnan(atm_iv_3m_prev_week) and atm_iv_3m_prev_week != 0 else 'N/A'
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
                'Realised Volatility 30d (%)': 'N/A',
                'Realised Volatility 60d (%)': 'N/A',
                'Realised Volatility 100d (%)': 'N/A',
                'Realised Volatility 100d 1d (%)': 'N/A',
                'Realised Volatility 100d 1w (%)': 'N/A',
                'Min Realised Volatility 100d (1y)': 'N/A',
                'Max Realised Volatility 100d (1y)': 'N/A',
                'Mean Realised Volatility 100d (1y)': 'N/A',
                'Rvol 100d Percentile (%)': 'N/A',
                'Rvol 100d Z-Score Percentile (%)': 'N/A',
                'Realised Volatility 180d (%)': 'N/A',
                'Realised Volatility 252d (%)': 'N/A',
                'Weighted IV (%)': weighted_iv * 100 if not np.isnan(weighted_iv) else 'N/A',
                'Weighted IV 3m (%)': weighted_iv_3m * 100 if not np.isnan(weighted_iv_3m) else 'N/A',
                'Weighted IV 3m 1d (%)': weighted_iv_3m_1d_pct,
                'Weighted IV 3m 1w (%)': weighted_iv_3m_1w_pct,
                'ATM IV 3m (%)': atm_iv_3m * 100 if not np.isnan(atm_iv_3m) else 'N/A',
                'ATM IV 3m 1d (%)': atm_iv_3m_1d_pct,
                'ATM IV 3m 1w (%)': atm_iv_3m_1w_pct,
                'Weighted IV 1d (%)': weighted_iv_1d_pct,
                'Weighted IV 1w (%)': weighted_iv_1w_pct,
                'Rvol100d - Weighted IV': 'N/A',
                'Volume': vol,
                'Volume 1d (%)': vol_1d_pct,
                'Volume 1w (%)': vol_1w_pct,
                'Open Interest': oi,
                'OI 1d (%)': oi_1d_pct,
                'OI 1w (%)': oi_1w_pct
            }
            if not latest.empty:
                latest_close = latest['Close'].values[0] if 'Close' in latest.columns else 'N/A'
                rank_dict['Latest Close'] = latest_close
                prev_close = get_prev_value(ticker, current_dt - timedelta(days=1), 'Close')
                close_1d_pct = (latest_close - prev_close) / prev_close * 100 if prev_close is not None and prev_close != 0 and latest_close != 'N/A' else 'N/A'
                rank_dict['Close 1d (%)'] = close_1d_pct
                prev_week_close = get_prev_value(ticker, current_dt - timedelta(days=7), 'Close')
                close_1w_pct = (latest_close - prev_week_close) / prev_week_close * 100 if prev_week_close is not None and prev_week_close != 0 and latest_close != 'N/A' else 'N/A'
                rank_dict['Close 1w (%)'] = close_1w_pct
                latest_high = latest['High'].values[0] if 'High' in latest.columns else 'N/A'
                rank_dict['Latest High'] = latest_high
                prev_high = get_prev_value(ticker, current_dt - timedelta(days=1), 'High')
                high_1d_pct = (latest_high - prev_high) / prev_high * 100 if prev_high is not None and prev_high != 0 and latest_high != 'N/A' else 'N/A'
                rank_dict['High 1d (%)'] = high_1d_pct
                prev_week_high = get_prev_value(ticker, current_dt - timedelta(days=7), 'High')
                high_1w_pct = (latest_high - prev_week_high) / prev_week_high * 100 if prev_week_high is not None and prev_week_high != 0 and latest_high != 'N/A' else 'N/A'
                rank_dict['High 1w (%)'] = high_1w_pct
                latest_low = latest['Low'].values[0] if 'Low' in latest.columns else 'N/A'
                rank_dict['Latest Low'] = latest_low
                prev_low = get_prev_value(ticker, current_dt - timedelta(days=1), 'Low')
                low_1d_pct = (latest_low - prev_low) / prev_low * 100 if prev_low is not None and prev_low != 0 and latest_low != 'N/A' else 'N/A'
                rank_dict['Low 1d (%)'] = low_1d_pct
                prev_week_low = get_prev_value(ticker, current_dt - timedelta(days=7), 'Low')
                low_1w_pct = (latest_low - prev_week_low) / prev_week_low * 100 if prev_week_low is not None and prev_week_low != 0 and latest_low != 'N/A' else 'N/A'
                rank_dict['Low 1w (%)'] = low_1w_pct
                for rvol_type in rvol_types:
                    col = f'Realised_Vol_Close_{rvol_type}'
                    current_vol = latest[col].values[0] if col in latest.columns and not latest.empty else 'N/A'
                    if current_vol == 'N/A' or pd.isna(current_vol):
                        current_vol = calculate_rvol_days(ticker, int(rvol_type))
                    rank_dict[f'Realised Volatility {rvol_type}d (%)'] = current_vol
                    if rvol_type == '100':
                        prev_day_vol = get_prev_value(ticker, current_dt - timedelta(days=1), col) if col in df_historic.columns else None
                        if prev_day_vol is None or pd.isna(prev_day_vol):
                            prev_day_vol = calculate_rvol_days(ticker, 100)
                        vol_1d_pct = (current_vol - prev_day_vol) / prev_day_vol * 100 if prev_day_vol is not None and prev_day_vol != 0 and current_vol != 'N/A' and not pd.isna(current_vol) else 'N/A'
                        rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = vol_1d_pct
                        prev_week_vol = get_prev_value(ticker, current_dt - timedelta(days=7), col) if col in df_historic.columns else None
                        if prev_week_vol is None or pd.isna(prev_week_vol):
                            prev_week_vol = calculate_rvol_days(ticker, 100)
                        vol_1w_pct = (current_vol - prev_week_vol) / prev_week_vol * 100 if prev_week_vol is not None and prev_week_vol != 0 and current_vol != 'N/A' and not pd.isna(current_vol) else 'N/A'
                        rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = vol_1w_pct
                        past_year = df_historic[(df_historic['Ticker'] == ticker) & (df_historic['Date'] >= past_year_start) & (df_historic['Date'] <= current_dt)]
                        vols = past_year[col].dropna() if col in past_year.columns and not past_year.empty else pd.Series()
                        if vols.empty and current_vol != 'N/A' and not pd.isna(current_vol):
                            stock = yf.Ticker(ticker)
                            hist = stock.history(period="1y")
                            if not hist.empty:
                                log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                                if len(log_returns) >= 2:
                                    vols = log_returns.rolling(window=int(rvol_type)).std() * np.sqrt(252) * 100
                                    vols = vols.dropna()
                        if not vols.empty:
                            min_vol = vols.min()
                            max_vol = vols.max()
                            mean_vol = vols.mean()
                            percentile = calculate_rvol_percentile(ticker, current_vol, df_historic, past_year_start, current_dt)
                            z_score_percentile = calculate_rvol_z_score_percentile(ticker, current_vol, df_historic, past_year_start, current_dt)
                        else:
                            min_vol = max_vol = mean_vol = percentile = z_score_percentile = 'N/A'
                        rank_dict[f'Min Realised Volatility {rvol_type}d (1y)'] = min_vol if rvol_type == '100' else 'N/A'
                        rank_dict[f'Max Realised Volatility {rvol_type}d (1y)'] = max_vol if rvol_type == '100' else 'N/A'
                        rank_dict[f'Mean Realised Volatility {rvol_type}d (1y)'] = mean_vol if rvol_type == '100' else 'N/A'
                        rank_dict['Rvol 100d Percentile (%)'] = percentile if rvol_type == '100' else rank_dict.get('Rvol 100d Percentile (%)', 'N/A')
                        rank_dict['Rvol 100d Z-Score Percentile (%)'] = z_score_percentile if rvol_type == '100' else rank_dict.get('Rvol 100d Z-Score Percentile (%)', 'N/A')
                        rvol100d_minus_weighted_iv = current_vol - (weighted_iv * 100) if current_vol != 'N/A' and not pd.isna(current_vol) and not np.isnan(weighted_iv) else 'N/A'
                        rank_dict['Rvol100d - Weighted IV'] = rvol100d_minus_weighted_iv if rvol_type == '100' else rank_dict.get('Rvol100d - Weighted IV', 'N/A')
            else:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    rank_dict['Latest Close'] = hist['Close'].iloc[-1]
                    rank_dict['Latest High'] = hist['High'].iloc[-1]
                    rank_dict['Latest Low'] = hist['Low'].iloc[-1]
                    current_price = hist['Close'].iloc[-1]
                    atm_iv_3m = calculate_atm_iv(ticker_processed, current_price, current_dt) if not ticker_processed.empty else np.nan
                    rank_dict['ATM IV 3m (%)'] = atm_iv_3m * 100 if not np.isnan(atm_iv_3m) else 'N/A'
                    # Note: 1d and 1w changes for ATM IV 3m are not calculated here due to missing historical prices
                    rank_dict['ATM IV 3m 1d (%)'] = 'N/A'
                    rank_dict['ATM IV 3m 1w (%)'] = 'N/A'
                for rvol_type in rvol_types:
                    current_vol = calculate_rvol_days(ticker, int(rvol_type))
                    rank_dict[f'Realised Volatility {rvol_type}d (%)'] = current_vol
                    if rvol_type == '100':
                        prev_day_vol = calculate_rvol_days(ticker, 100)
                        vol_1d_pct = (current_vol - prev_day_vol) / prev_day_vol * 100 if prev_day_vol is not None and prev_day_vol != 0 and current_vol != 'N/A' and not pd.isna(current_vol) else 'N/A'
                        rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = vol_1d_pct
                        prev_week_vol = calculate_rvol_days(ticker, 100)
                        vol_1w_pct = (current_vol - prev_week_vol) / prev_week_vol * 100 if prev_week_vol is not None and prev_week_vol != 0 and current_vol != 'N/A' and not pd.isna(current_vol) else 'N/A'
                        rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = vol_1w_pct
                        hist = stock.history(period="1y")
                        if not hist.empty:
                            log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                            if len(log_returns) >= 2:
                                vols = log_returns.rolling(window=100).std() * np.sqrt(252) * 100
                                vols = vols.dropna()
                                if not vols.empty:
                                    min_vol = vols.min()
                                    max_vol = vols.max()
                                    mean_vol = vols.mean()
                                    percentile = (vols < current_vol).sum() / len(vols) * 100 if current_vol != 'N/A' and not pd.isna(current_vol) else 'N/A'
                                    z_score_percentile = norm.cdf((current_vol - vols.mean()) / vols.std()) * 100 if vols.std() != 0 and current_vol != 'N/A' and not pd.isna(current_vol) else 'N/A'
                                else:
                                    min_vol = max_vol = mean_vol = percentile = z_score_percentile = 'N/A'
                            else:
                                min_vol = max_vol = mean_vol = percentile = z_score_percentile = 'N/A'
                        else:
                            min_vol = max_vol = mean_vol = percentile = z_score_percentile = 'N/A'
                        rank_dict[f'Min Realised Volatility {rvol_type}d (1y)'] = min_vol if rvol_type == '100' else 'N/A'
                        rank_dict[f'Max Realised Volatility {rvol_type}d (1y)'] = max_vol if rvol_type == '100' else 'N/A'
                        rank_dict[f'Mean Realised Volatility {rvol_type}d (1y)'] = mean_vol if rvol_type == '100' else 'N/A'
                        rank_dict['Rvol 100d Percentile (%)'] = percentile if rvol_type == '100' else rank_dict.get('Rvol 100d Percentile (%)', 'N/A')
                        rank_dict['Rvol 100d Z-Score Percentile (%)'] = z_score_percentile if rvol_type == '100' else rank_dict.get('Rvol 100d Z-Score Percentile (%)', 'N/A')
                        rvol100d_minus_weighted_iv = current_vol - (weighted_iv * 100) if current_vol != 'N/A' and not pd.isna(current_vol) and not np.isnan(weighted_iv) else 'N/A'
                        rank_dict['Rvol100d - Weighted IV'] = rvol100d_minus_weighted_iv if rvol_type == '100' else rank_dict.get('Rvol100d - Weighted IV', 'N/A')
            ranking.append(rank_dict)
        
        column_order = [
            'Rank',
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
            'Realised Volatility 30d (%)',
            'Realised Volatility 60d (%)',
            'Realised Volatility 100d (%)',
            'Realised Volatility 100d 1d (%)',
            'Realised Volatility 100d 1w (%)',
            'Min Realised Volatility 100d (1y)',
            'Max Realised Volatility 100d (1y)',
            'Mean Realised Volatility 100d (1y)',
            'Rvol 100d Percentile (%)',
            'Rvol 100d Z-Score Percentile (%)',
            'Realised Volatility 180d (%)',
            'Realised Volatility 252d (%)',
            'Weighted IV (%)',
            'Weighted IV 3m (%)',
            'Weighted IV 3m 1d (%)',
            'Weighted IV 3m 1w (%)',
            'ATM IV 3m (%)',
            'ATM IV 3m 1d (%)',
            'ATM IV 3m 1w (%)',
            'Weighted IV 1d (%)',
            'Weighted IV 1w (%)',
            'Rvol100d - Weighted IV',
            'Volume',
            'Volume 1d (%)',
            'Volume 1w (%)',
            'Open Interest',
            'OI 1d (%)',
            'OI 1w (%)'
        ]
        df_ranking = pd.DataFrame(ranking)
        ranking_dir = f'data/{timestamp}/ranking'
        os.makedirs(ranking_dir, exist_ok=True)
        output_file = f'{ranking_dir}/ranking{prefix}.csv'
        df_ranking.to_csv(output_file, index=False)
        print(f"Ranking metrics saved to {output_file}")

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
        sources = ['yfinance']
        calculate_ranking_metrics(timestamp, sources)
    else:
        sources = ['yfinance']
        for timestamp in timestamps:
            calculate_ranking_metrics(timestamp, sources)

main()
