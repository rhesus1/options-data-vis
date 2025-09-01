import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
import glob
from scipy.stats import norm

def calculate_atm_iv(ticker_processed, current_price, current_dt):
    if ticker_processed.empty or 'Ticker' not in ticker_processed.columns:
        return np.nan
    ticker_processed['Expiry_dt'] = pd.to_datetime(ticker_processed['Expiry'])
    ticker_processed['Days_to_Expiry'] = (ticker_processed['Expiry_dt'] - current_dt).dt.days
    three_month_data = ticker_processed[(ticker_processed['Days_to_Expiry'] >= 80) & (ticker_processed['Days_to_Expiry'] <= 100)]
    if three_month_data.empty:
        return np.nan
    atm_options = three_month_data.iloc[(three_month_data['Moneyness'] - 1.0).abs().argsort()]
    atm_iv = atm_options['IV_mid'].iloc[0] if 'IV_mid' in atm_options.columns and not atm_options['IV_mid'].isna().all() else np.nan
    return atm_iv

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
            if ts is None:
                return pd.DataFrame(columns=['Ticker'])
            processed_dir = f'data/{ts}/processed{prefix}'
            if not os.path.exists(processed_dir):
                print(f"Processed directory {processed_dir} not found")
                return pd.DataFrame(columns=['Ticker'])
            processed_files = glob.glob(f'{processed_dir}/processed{prefix}_*.csv')
            dfs = []
            required_columns = ['Ticker', 'IV_mid', 'Years_to_Expiry', 'Moneyness', 'Expiry']
            for file in processed_files:
                try:
                    df = pd.read_csv(file)
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    if missing_cols:
                        print(f"Missing columns {missing_cols} in {file}")
                        continue
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            if not dfs:
                print(f"No processed data files found in {processed_dir}")
                return pd.DataFrame(columns=['Ticker'])
            df_concat = pd.concat(dfs, ignore_index=True)
            return df_concat if not df_concat.empty else pd.DataFrame(columns=['Ticker'])
        
        df_processed = load_processed_data(timestamp, prefix)
        if df_processed.empty:
            print(f"No processed data found for {timestamp}")
            continue
        
        processed_prev_day = load_processed_data(prev_day_ts, prefix)
        processed_prev_week = load_processed_data(prev_week_ts, prefix)
        
        prev_day_historic = load_historic_data(prev_day_ts)
        prev_week_historic = load_historic_data(prev_week_ts)
        
        def get_prev_value(ticker, target_date, col, ts):
            if ts is None:
                return 'N/A'
            historic_file = f'data/{ts}/historic/historic_{ticker}.csv'
            if not os.path.exists(historic_file):
                print(f"No historic file found for {ticker} in data/{ts}/historic")
                return 'N/A'
            try:
                historic_data = pd.read_csv(historic_file, parse_dates=['Date'])
                if historic_data.empty:
                    return 'N/A'
                date_data = historic_data[historic_data['Date'] == target_date]
                return date_data[col].iloc[0] if not date_data.empty and col in date_data.columns else 'N/A'
            except Exception as e:
                print(f"Error reading historic file for {ticker} in data/{ts}/historic: {e}")
                return 'N/A'
        
        tickers = list(set(df_processed['Ticker'].unique()) | set(df_historic['Ticker'].unique()) | set(current_option['Ticker'].unique()))
        ranking = []
        rvol_types = ['30', '60', '100', '180', '252']
        
        for idx, ticker in enumerate(tickers, 1):
            rank_dict = {'Rank': idx, 'Ticker': ticker}
            
            # Load historic data for the current timestamp
            historic_file = f'data/{timestamp}/historic/historic_{ticker}.csv'
            ticker_historic = pd.DataFrame()
            if os.path.exists(historic_file):
                try:
                    ticker_historic = pd.read_csv(historic_file, parse_dates=['Date'])
                except Exception as e:
                    print(f"Error reading historic file for {ticker}: {e}")
            
            ticker_processed = df_processed[df_processed['Ticker'] == ticker] if 'Ticker' in df_processed.columns else pd.DataFrame()
            ticker_option = current_option[current_option['Ticker'] == ticker] if 'Ticker' in current_option.columns else pd.DataFrame()
            
            if not ticker_historic.empty:
                rank_dict['Latest Close'] = ticker_historic['Close'].iloc[-1] if 'Close' in ticker_historic.columns else 'N/A'
                rank_dict['Latest High'] = ticker_historic['High'].iloc[-1] if 'High' in ticker_historic.columns else 'N/A'
                rank_dict['Latest Low'] = ticker_historic['Low'].iloc[-1] if 'Low' in ticker_historic.columns else 'N/A'
                current_price = ticker_historic['Close'].iloc[-1] if 'Close' in ticker_historic.columns else 'N/A'
                
                prev_day_close = get_prev_value(ticker, (current_dt - timedelta(days=1)).strftime('%Y-%m-%d'), 'Close', prev_day_ts)
                prev_week_close = get_prev_value(ticker, (current_dt - timedelta(days=7)).strftime('%Y-%m-%d'), 'Close', prev_week_ts)
                prev_day_high = get_prev_value(ticker, (current_dt - timedelta(days=1)).strftime('%Y-%m-%d'), 'High', prev_day_ts)
                prev_week_high = get_prev_value(ticker, (current_dt - timedelta(days=7)).strftime('%Y-%m-%d'), 'High', prev_week_ts)
                prev_day_low = get_prev_value(ticker, (current_dt - timedelta(days=1)).strftime('%Y-%m-%d'), 'Low', prev_day_ts)
                prev_week_low = get_prev_value(ticker, (current_dt - timedelta(days=7)).strftime('%Y-%m-%d'), 'Low', prev_week_ts)
                
                rank_dict['Close 1d (%)'] = ((current_price - prev_day_close) / prev_day_close * 100) if prev_day_close != 'N/A' and prev_day_close != 0 else 'N/A'
                rank_dict['Close 1w (%)'] = ((current_price - prev_week_close) / prev_week_close * 100) if prev_week_close != 'N/A' and prev_week_close != 0 else 'N/A'
                rank_dict['High 1d (%)'] = ((ticker_historic['High'].iloc[-1] - prev_day_high) / prev_day_high * 100) if prev_day_high != 'N/A' and prev_day_high != 0 and 'High' in ticker_historic.columns else 'N/A'
                rank_dict['High 1w (%)'] = ((ticker_historic['High'].iloc[-1] - prev_week_high) / prev_week_high * 100) if prev_week_high != 'N/A' and prev_week_high != 0 and 'High' in ticker_historic.columns else 'N/A'
                rank_dict['Low 1d (%)'] = ((ticker_historic['Low'].iloc[-1] - prev_day_low) / prev_day_low * 100) if prev_day_low != 'N/A' and prev_day_low != 0 and 'Low' in ticker_historic.columns else 'N/A'
                rank_dict['Low 1w (%)'] = ((ticker_historic['Low'].iloc[-1] - prev_week_low) / prev_week_low * 100) if prev_week_low != 'N/A' and prev_week_low != 0 and 'Low' in ticker_historic.columns else 'N/A'
                
                for rvol_type in rvol_types:
                    col_name = f'Realised_Vol_Close_{rvol_type}'
                    if col_name in ticker_historic.columns:
                        current_vol = ticker_historic[col_name].iloc[-1]
                        rank_dict[f'Realised Volatility {rvol_type}d (%)'] = current_vol
                        if rvol_type == '100':
                            prev_day_vol = get_prev_value(ticker, (current_dt - timedelta(days=1)).strftime('%Y-%m-%d'), col_name, prev_day_ts)
                            prev_week_vol = get_prev_value(ticker, (current_dt - timedelta(days=7)).strftime('%Y-%m-%d'), col_name, prev_week_ts)
                            rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = ((current_vol - prev_day_vol) / prev_day_vol * 100) if prev_day_vol != 'N/A' and prev_day_vol != 0 else 'N/A'
                            rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = ((current_vol - prev_week_vol) / prev_week_vol * 100) if prev_week_vol != 'N/A' and prev_week_vol != 0 else 'N/A'
                            
                            # Load all historic data for the past year from all timestamps
                            year_historic = pd.DataFrame()
                            for ts in timestamps:
                                ts_dt = datetime.strptime(ts[:8], "%Y%m%d")
                                if past_year_start <= ts_dt <= current_dt:
                                    hist_file = f'data/{ts}/historic/historic_{ticker}.csv'
                                    if os.path.exists(hist_file):
                                        try:
                                            hist_df = pd.read_csv(hist_file, parse_dates=['Date'])
                                            year_historic = pd.concat([year_historic, hist_df], ignore_index=True)
                                        except Exception as e:
                                            print(f"Error reading historic file for {ticker} at {ts}: {e}")
                            
                            if not year_historic.empty and col_name in year_historic.columns:
                                vols = year_historic[col_name].dropna()
                                if not vols.empty:
                                    min_vol = vols.min()
                                    max_vol = vols.max()
                                    mean_vol = vols.mean()
                                    percentile = (vols < current_vol).sum() / len(vols) * 100 if not pd.isna(current_vol) else 'N/A'
                                    z_score_percentile = norm.cdf((current_vol - vols.mean()) / vols.std()) * 100 if vols.std() != 0 and not pd.isna(current_vol) else 'N/A'
                                else:
                                    min_vol = max_vol = mean_vol = percentile = z_score_percentile = 'N/A'
                            else:
                                min_vol = max_vol = mean_vol = percentile = z_score_percentile = 'N/A'
                            rank_dict[f'Min Realised Volatility {rvol_type}d (1y)'] = min_vol if rvol_type == '100' else 'N/A'
                            rank_dict[f'Max Realised Volatility {rvol_type}d (1y)'] = max_vol if rvol_type == '100' else 'N/A'
                            rank_dict[f'Mean Realised Volatility {rvol_type}d (1y)'] = mean_vol if rvol_type == '100' else 'N/A'
                            rank_dict['Rvol 100d Percentile (%)'] = percentile if rvol_type == '100' else rank_dict.get('Rvol 100d Percentile (%)', 'N/A')
                            rank_dict['Rvol 100d Z-Score Percentile (%)'] = z_score_percentile if rvol_type == '100' else rank_dict.get('Rvol 100d Z-Score Percentile (%)', 'N/A')
                    else:
                        rank_dict[f'Realised Volatility {rvol_type}d (%)'] = 'N/A'
                        if rvol_type == '100':
                            rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = 'N/A'
                            rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = 'N/A'
                            rank_dict[f'Min Realised Volatility {rvol_type}d (1y)'] = 'N/A'
                            rank_dict[f'Max Realised Volatility {rvol_type}d (1y)'] = 'N/A'
                            rank_dict[f'Mean Realised Volatility {rvol_type}d (1y)'] = 'N/A'
                            rank_dict['Rvol 100d Percentile (%)'] = 'N/A'
                            rank_dict['Rvol 100d Z-Score Percentile (%)'] = 'N/A'
            
            if not ticker_processed.empty and 'Ticker' in ticker_processed.columns:
                weighted_iv = ticker_processed['IV_mid'].mean() if 'IV_mid' in ticker_processed.columns and not ticker_processed['IV_mid'].isna().all() else np.nan
                rank_dict['Weighted IV (%)'] = weighted_iv * 100 if not np.isnan(weighted_iv) else 'N/A'
                
                three_month_data = ticker_processed[(ticker_processed['Years_to_Expiry'] >= 80/365.25) & (ticker_processed['Years_to_Expiry'] <= 100/365.25)]
                weighted_iv_3m = three_month_data['IV_mid'].mean() if not three_month_data.empty and 'IV_mid' in three_month_data.columns and not three_month_data['IV_mid'].isna().all() else np.nan
                rank_dict['Weighted IV 3m (%)'] = weighted_iv_3m * 100 if not np.isnan(weighted_iv_3m) else 'N/A'
                
                prev_day_ticker_processed = processed_prev_day[processed_prev_day['Ticker'] == ticker] if not processed_prev_day.empty and 'Ticker' in processed_prev_day.columns else pd.DataFrame()
                prev_week_ticker_processed = processed_prev_week[processed_prev_week['Ticker'] == ticker] if not processed_prev_week.empty and 'Ticker' in processed_prev_week.columns else pd.DataFrame()
                
                prev_day_weighted_iv = prev_day_ticker_processed['IV_mid'].mean() if not prev_day_ticker_processed.empty and 'IV_mid' in prev_day_ticker_processed.columns and not prev_day_ticker_processed['IV_mid'].isna().all() else np.nan
                prev_week_weighted_iv = prev_week_ticker_processed['IV_mid'].mean() if not prev_week_ticker_processed.empty and 'IV_mid' in prev_week_ticker_processed.columns and not prev_week_ticker_processed['IV_mid'].isna().all() else np.nan
                
                rank_dict['Weighted IV 1d (%)'] = ((weighted_iv - prev_day_weighted_iv) / prev_day_weighted_iv * 100) if not np.isnan(weighted_iv) and not np.isnan(prev_day_weighted_iv) and prev_day_weighted_iv != 0 else 'N/A'
                rank_dict['Weighted IV 1w (%)'] = ((weighted_iv - prev_week_weighted_iv) / prev_week_weighted_iv * 100) if not np.isnan(weighted_iv) and not np.isnan(prev_week_weighted_iv) and prev_week_weighted_iv != 0 else 'N/A'
                
                prev_day_weighted_iv_3m = prev_day_ticker_processed[(prev_day_ticker_processed['Years_to_Expiry'] >= 80/365.25) & (prev_day_ticker_processed['Years_to_Expiry'] <= 100/365.25)]['IV_mid'].mean() if not prev_day_ticker_processed.empty and 'Years_to_Expiry' in prev_day_ticker_processed.columns else np.nan
                prev_week_weighted_iv_3m = prev_week_ticker_processed[(prev_week_ticker_processed['Years_to_Expiry'] >= 80/365.25) & (prev_week_ticker_processed['Years_to_Expiry'] <= 100/365.25)]['IV_mid'].mean() if not prev_week_ticker_processed.empty and 'Years_to_Expiry' in prev_week_ticker_processed.columns else np.nan
                
                rank_dict['Weighted IV 3m 1d (%)'] = ((weighted_iv_3m - prev_day_weighted_iv_3m) / prev_day_weighted_iv_3m * 100) if not np.isnan(weighted_iv_3m) and not np.isnan(prev_day_weighted_iv_3m) and prev_day_weighted_iv_3m != 0 else 'N/A'
                rank_dict['Weighted IV 3m 1w (%)'] = ((weighted_iv_3m - prev_week_weighted_iv_3m) / prev_week_weighted_iv_3m * 100) if not np.isnan(weighted_iv_3m) and not np.isnan(prev_week_weighted_iv_3m) and prev_week_weighted_iv_3m != 0 else 'N/A'
                
                atm_iv_3m = calculate_atm_iv(ticker_processed, current_price, current_dt) if not ticker_processed.empty and current_price != 'N/A' else np.nan
                rank_dict['ATM IV 3m (%)'] = atm_iv_3m * 100 if not np.isnan(atm_iv_3m) else 'N/A'
                
                prev_day_atm_iv_3m = calculate_atm_iv(prev_day_ticker_processed, prev_day_close, current_dt - timedelta(days=1)) if not prev_day_ticker_processed.empty and prev_day_close != 'N/A' else np.nan
                prev_week_atm_iv_3m = calculate_atm_iv(prev_week_ticker_processed, prev_week_close, current_dt - timedelta(days=7)) if not prev_week_ticker_processed.empty and prev_week_close != 'N/A' else np.nan
                
                rank_dict['ATM IV 3m 1d (%)'] = ((atm_iv_3m - prev_day_atm_iv_3m) / prev_day_atm_iv_3m * 100) if not np.isnan(atm_iv_3m) and not np.isnan(prev_day_atm_iv_3m) and prev_day_atm_iv_3m != 0 else 'N/A'
                rank_dict['ATM IV 3m 1w (%)'] = ((atm_iv_3m - prev_week_atm_iv_3m) / prev_week_atm_iv_3m * 100) if not np.isnan(atm_iv_3m) and not np.isnan(prev_week_atm_iv_3m) and prev_week_atm_iv_3m != 0 else 'N/A'
                
                if not ticker_option.empty and 'Ticker' in ticker_option.columns:
                    rank_dict['Volume'] = ticker_option['Vol'].iloc[0]
                    rank_dict['Open Interest'] = ticker_option['OI'].iloc[0]
                    prev_day_option_ticker = prev_day_option[prev_day_option['Ticker'] == ticker] if not prev_day_option.empty and 'Ticker' in prev_day_option.columns else pd.DataFrame()
                    prev_week_option_ticker = prev_week_option[prev_week_option['Ticker'] == ticker] if not prev_week_option.empty and 'Ticker' in prev_week_option.columns else pd.DataFrame()
                    prev_day_volume = prev_day_option_ticker['Vol'].iloc[0] if not prev_day_option_ticker.empty and 'Vol' in prev_day_option_ticker.columns else 'N/A'
                    prev_week_volume = prev_week_option_ticker['Vol'].iloc[0] if not prev_week_option_ticker.empty and 'Vol' in prev_week_option_ticker.columns else 'N/A'
                    prev_day_oi = prev_day_option_ticker['OI'].iloc[0] if not prev_day_option_ticker.empty and 'OI' in prev_day_option_ticker.columns else 'N/A'
                    prev_week_oi = prev_week_option_ticker['OI'].iloc[0] if not prev_week_option_ticker.empty and 'OI' in prev_week_option_ticker.columns else 'N/A'
                    rank_dict['Volume 1d (%)'] = ((rank_dict['Volume'] - prev_day_volume) / prev_day_volume * 100) if prev_day_volume != 'N/A' and prev_day_volume != 0 else 'N/A'
                    rank_dict['Volume 1w (%)'] = ((rank_dict['Volume'] - prev_week_volume) / prev_week_volume * 100) if prev_week_volume != 'N/A' and prev_week_volume != 0 else 'N/A'
                    rank_dict['OI 1d (%)'] = ((rank_dict['Open Interest'] - prev_day_oi) / prev_day_oi * 100) if prev_day_oi != 'N/A' and prev_day_oi != 0 else 'N/A'
                    rank_dict['OI 1w (%)'] = ((rank_dict['Open Interest'] - prev_week_oi) / prev_week_oi * 100) if prev_week_oi != 'N/A' and prev_week_oi != 0 else 'N/A'
                else:
                    rank_dict['Volume'] = 'N/A'
                    rank_dict['Open Interest'] = 'N/A'
                    rank_dict['Volume 1d (%)'] = 'N/A'
                    rank_dict['Volume 1w (%)'] = 'N/A'
                    rank_dict['OI 1d (%)'] = 'N/A'
                    rank_dict['OI 1w (%)'] = 'N/A'
                
                for rvol_type in rvol_types:
                    if rvol_type == '100':
                        current_vol = ticker_historic[f'Realised_Vol_Close_{rvol_type}'].iloc[-1] if not ticker_historic.empty and f'Realised_Vol_Close_{rvol_type}' in ticker_historic.columns else 'N/A'
                        rank_dict['Rvol100d - Weighted IV'] = (current_vol - (weighted_iv * 100)) if current_vol != 'N/A' and not pd.isna(current_vol) and not np.isnan(weighted_iv) else 'N/A'
                    else:
                        rank_dict['Rvol100d - Weighted IV'] = rank_dict.get('Rvol100d - Weighted IV', 'N/A')
            
            else:
                rank_dict['Latest Close'] = 'N/A'
                rank_dict['Latest High'] = 'N/A'
                rank_dict['Latest Low'] = 'N/A'
                rank_dict['Close 1d (%)'] = 'N/A'
                rank_dict['Close 1w (%)'] = 'N/A'
                rank_dict['High 1d (%)'] = 'N/A'
                rank_dict['High 1w (%)'] = 'N/A'
                rank_dict['Low 1d (%)'] = 'N/A'
                rank_dict['Low 1w (%)'] = 'N/A'
                for rvol_type in rvol_types:
                    rank_dict[f'Realised Volatility {rvol_type}d (%)'] = 'N/A'
                    if rvol_type == '100':
                        rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = 'N/A'
                        rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = 'N/A'
                        rank_dict[f'Min Realised Volatility {rvol_type}d (1y)'] = 'N/A'
                        rank_dict[f'Max Realised Volatility {rvol_type}d (1y)'] = 'N/A'
                        rank_dict[f'Mean Realised Volatility {rvol_type}d (1y)'] = 'N/A'
                        rank_dict['Rvol 100d Percentile (%)'] = 'N/A'
                        rank_dict['Rvol 100d Z-Score Percentile (%)'] = 'N/A'
                rank_dict['Weighted IV (%)'] = 'N/A'
                rank_dict['Weighted IV 3m (%)'] = 'N/A'
                rank_dict['Weighted IV 3m 1d (%)'] = 'N/A'
                rank_dict['Weighted IV 3m 1w (%)'] = 'N/A'
                rank_dict['ATM IV 3m (%)'] = 'N/A'
                rank_dict['ATM IV 3m 1d (%)'] = 'N/A'
                rank_dict['ATM IV 3m 1w (%)'] = 'N/A'
                rank_dict['Weighted IV 1d (%)'] = 'N/A'
                rank_dict['Weighted IV 1w (%)'] = 'N/A'
                rank_dict['Rvol100d - Weighted IV'] = 'N/A'
                rank_dict['Volume'] = 'N/A'
                rank_dict['Open Interest'] = 'N/A'
                rank_dict['Volume 1d (%)'] = 'N/A'
                rank_dict['Volume 1w (%)'] = 'N/A'
                rank_dict['OI 1d (%)'] = 'N/A'
                rank_dict['OI 1w (%)'] = 'N/A'
            
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
        df_ranking = df_ranking[column_order]
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
