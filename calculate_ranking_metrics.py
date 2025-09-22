import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime, timedelta
import os
import glob
from scipy.stats import norm

pd.options.mode.chained_assignment = None
pd.set_option('future.no_silent_downcasting', True)

def global_vol_model_hyp(x, a0, a1, b0, b1, m0, m1, rho0, rho1, sigma0, sigma1, c):
    moneyness, tau = x
    k = np.log(moneyness)
    try:
        a_T = a0 + a1 / (1 + c * tau)
        b_T = b0 + b1 / (1 + c * tau)
        m_T = m0 + m1 / (1 + c * tau)
        rho_T = rho0 + rho1 / (1 + c * tau)
        sigma_T = sigma0 + sigma1 / (1 + c * tau)
        return a_T + b_T * (rho_T * (k - m_T) + np.sqrt((k - m_T)**2 + sigma_T**2))
    except Exception:
        return np.nan

def calculate_atm_iv(ticker_processed, current_price, current_dt, timestamp, option_type='Call'):
    if ticker_processed.empty:
        print(f"calculate_atm_iv: Empty ticker_processed DataFrame for {ticker_processed.get('Ticker', 'unknown')}")
        return np.nan

    ticker = ticker_processed.get('Ticker', 'unknown') if not ticker_processed.empty else 'unknown'

    vol_surf_file = f'data/{timestamp}/vol_surf/vol_surf.csv'
    if os.path.exists(vol_surf_file):
        try:
            vol_surf_df = pd.read_csv(vol_surf_file)
            vol_surf_df = vol_surf_df[vol_surf_df['Ticker'] == ticker] if 'Ticker' in vol_surf_df.columns else pd.DataFrame()
            if not vol_surf_df.empty:
                params = vol_surf_df[vol_surf_df['Option_Type'] == option_type]
                if params.empty and option_type == 'Call':
                    params = vol_surf_df[vol_surf_df['Option_Type'] == 'Put']
                if not params.empty:
                    params = params.iloc[0][['a0', 'a1', 'b0', 'b1', 'm0', 'm1', 'rho0', 'rho1', 'sigma0', 'sigma1', 'c']].values
                    T = 0.25  # 3 months
                    moneyness = 1.0  # ATM
                    iv = global_vol_model_hyp((moneyness, T), *params)
                    if not np.isnan(iv) and iv > 0:
                        print(f"calculate_atm_iv: ATM IV = {iv} for {ticker} using fitted vol surface (T=0.25, moneyness=1.0)")
                        return iv
                    else:
                        print(f"calculate_atm_iv: Invalid IV {iv} from fitted vol surface for {ticker}, falling back to IV_mid")
        except Exception as e:
            print(f"calculate_atm_iv: Error reading {vol_surf_file} for {ticker}: {e}")

    required_cols = ['Expiry', 'IV_mid', 'Moneyness']
    missing_cols = [col for col in required_cols if col not in ticker_processed.columns]
    if missing_cols:
        print(f"calculate_atm_iv: Missing columns {missing_cols} for {ticker}")
        return np.nan
    try:
        ticker_processed['Expiry_dt'] = pd.to_datetime(ticker_processed['Expiry'], errors='coerce')
        if ticker_processed['Expiry_dt'].isna().all():
            print(f"calculate_atm_iv: Invalid Expiry dates for {ticker}")
            return np.nan

        ticker_processed['Days_to_Expiry'] = (ticker_processed['Expiry_dt'] - current_dt).dt.days
        three_month_data = ticker_processed[(ticker_processed['Days_to_Expiry'] >= 70) &
                                           (ticker_processed['Days_to_Expiry'] <= 110)]

        if three_month_data.empty:
            print(f"calculate_atm_iv: No options with 70-110 days to expiry for {ticker}")
            three_month_data = ticker_processed[(ticker_processed['Days_to_Expiry'] >= 60) &
                                               (ticker_processed['Days_to_Expiry'] <= 120)]
            if three_month_data.empty:
                print(f"calculate_atm_iv: Fallback range 60-120 days also empty for {ticker}")
                return np.nan

        atm_options = three_month_data[(three_month_data['Moneyness'] >= 0.9) & (three_month_data['Moneyness'] <= 1.1)]
        if atm_options.empty:
            print(f"calculate_atm_iv: No options with moneyness 0.9-1.1 for {ticker}")
            return np.nan
        atm_options = atm_options.iloc[(atm_options['Moneyness'] - 1.0).abs().argsort()]
        if atm_options['IV_mid'].isna().all():
            print(f"calculate_atm_iv: All IV_mid values are NaN for {ticker}")
            return np.nan

        atm_iv = atm_options['IV_mid'].iloc[0]
        print(f"calculate_atm_iv: ATM IV = {atm_iv} for {ticker} using IV_mid fallback")
        return atm_iv
    except Exception as e:
        print(f"calculate_atm_iv: Error processing data for {ticker}: {e}")
        return np.nan

def get_option_totals(ts, prefix):
    if ts is None:
        print(f"get_option_totals: No timestamp provided")
        return pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
    raw_dir = f'data/{ts}/raw{prefix}'
    print(f"get_option_totals: Reading from {raw_dir} for timestamp {ts}")
    if not os.path.exists(raw_dir):
        print(f"get_option_totals: Raw directory {raw_dir} not found")
        return pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
    raw_files = glob.glob(f'{raw_dir}/raw{prefix}_*.csv')
    if not raw_files:
        print(f"get_option_totals: No raw files found in {raw_dir}")
        return pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
    totals = []
    for file in raw_files:
        ticker = os.path.basename(file).split(f'raw{prefix}_')[1].split('.csv')[0]
        try:
            df = pd.read_csv(file)
            required_cols = ['Open Interest', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"get_option_totals: Missing columns {missing_cols} in {file}")
                totals.append({'Ticker': ticker, 'OI': 0, 'Vol': 0})
                continue
            df['Open Interest'] = pd.to_numeric(df['Open Interest'], errors='coerce').fillna(0)
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            oi_sum = df['Open Interest'].sum()
            vol_sum = df['Volume'].sum()
            totals.append({'Ticker': ticker, 'OI': oi_sum, 'Vol': vol_sum})
        except Exception as e:
            print(f"get_option_totals: Error reading {file}: {e}")
            totals.append({'Ticker': ticker, 'OI': 0, 'Vol': 0})
    return pd.DataFrame(totals, columns=['Ticker', 'OI', 'Vol'])

def load_historic_data(ts):
    if ts is None:
        print(f"load_historic_data: No timestamp provided")
        return pd.DataFrame()
    historic_dir = f'data/{ts}/historic'
    print(f"load_historic_data: Reading from {historic_dir} for timestamp {ts}")
    if not os.path.exists(historic_dir):
        print(f"load_historic_data: Historic directory {historic_dir} not found")
        return pd.DataFrame()
    historic_files = glob.glob(f'{historic_dir}/historic_*.csv')
    if not historic_files:
        print(f"load_historic_data: No historic data files found in {historic_dir}")
        return pd.DataFrame()
    dfs = []
    required_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close',
                       'Realised_Vol_Close_30', 'Realised_Vol_Close_60',
                       'Realised_Vol_Close_100', 'Realised_Vol_Close_180',
                       'Realised_Vol_Close_252']
    for file in historic_files:
        try:
            df = pd.read_csv(file, parse_dates=['Date'])
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"load_historic_data: Missing columns {missing_cols} in {file}")
                for col in missing_cols:
                    if col != 'Date' and col != 'Ticker':
                        df[col] = np.nan
                    elif col == 'Ticker':
                        ticker = os.path.basename(file).split('historic_')[1].split('.csv')[0]
                        df['Ticker'] = ticker
            numeric_cols = [col for col in required_columns if col not in ['Date', 'Ticker']]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            dfs.append(df)
        except Exception as e:
            print(f"load_historic_data: Error reading {file}: {e}")
    if dfs:
        historic_df = pd.concat(dfs, ignore_index=True)
        return historic_df
    return pd.DataFrame(columns=required_columns)

def get_prev_data(ts, days_back, source):
    prefix = "_yfinance" if source == "yfinance" else ""
    dates_file = 'data/dates.json'
    if not os.path.exists(dates_file):
        print(f"get_prev_data: dates.json not found")
        return pd.DataFrame(), pd.DataFrame(columns=['Ticker', 'OI', 'Vol']), pd.DataFrame()
    try:
        with open(dates_file, 'r') as f:
            dates = json.load(f)
    except Exception as e:
        print(f"get_prev_data: Error reading dates.json: {e}")
        return pd.DataFrame(), pd.DataFrame(columns=['Ticker', 'OI', 'Vol']), pd.DataFrame()
    timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
    ts_dt = datetime.strptime(ts, "%Y%m%d_%H%M")
    prev_ts = None
    for candidate_ts in reversed(timestamps):
        candidate_dt = datetime.strptime(candidate_ts, "%Y%m%d_%H%M")
        if candidate_dt < ts_dt and (ts_dt - candidate_dt).days >= days_back:
            prev_ts = candidate_ts
            break
    if not prev_ts:
        print(f"get_prev_data: No previous timestamp found {days_back} days back from {ts}")
        return pd.DataFrame(), pd.DataFrame(columns=['Ticker', 'OI', 'Vol']), pd.DataFrame()
    prev_ranking_path = f'data/{prev_ts}/ranking/ranking{prefix}.csv'
    prev_option_path = f'data/{prev_ts}/option_totals.csv'
    prev_atm_path = f'data/{prev_ts}/atm_iv.csv'
    prev_ranking = pd.read_csv(prev_ranking_path) if os.path.exists(prev_ranking_path) else pd.DataFrame()
    try:
        prev_option = pd.read_csv(prev_option_path) if os.path.exists(prev_option_path) else pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
        if not prev_option.empty and 'Ticker' not in prev_option.columns:
            print(f"get_prev_data: 'Ticker' column missing in {prev_option_path}, returning empty DataFrame with correct columns")
            prev_option = pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
    except Exception as e:
        print(f"get_prev_data: Error reading {prev_option_path}: {e}")
        prev_option = pd.DataFrame(columns=['Ticker', 'OI', 'Vol'])
    prev_atm = pd.read_csv(prev_atm_path) if os.path.exists(prev_atm_path) else pd.DataFrame()
    return prev_ranking, prev_option, prev_atm

def get_prev_value(ticker, target_date, col, historic_data):
    ticker_data = historic_data[historic_data['Ticker'] == ticker]
    if ticker_data.empty:
        print(f"get_prev_value: No data for ticker {ticker}")
        return 'N/A'
    target_date_only = target_date.date() if isinstance(target_date, datetime) else target_date
    ticker_data = ticker_data[ticker_data['Date'].dt.date <= target_date_only].sort_values('Date')
    if ticker_data.empty:
        print(f"get_prev_value: No data for ticker {ticker} on or before {target_date_only}")
        return 'N/A'
    date_data = ticker_data.iloc[-1:]
    print(f"get_prev_value: Using date {date_data['Date'].iloc[0]} for target {target_date_only}, ticker {ticker}")
    return date_data[col].iloc[0] if col in date_data.columns and not date_data[col].isna().all() else 'N/A'

def calculate_ranking_metrics(timestamp, sources):
    try:
        rvol_types = ['30', '60', '100', '180', '252']
        for source in sources:
            prefix = "_yfinance" if source == "yfinance" else ""
            historic = load_historic_data(timestamp)
            option_totals = get_option_totals(timestamp, prefix)
            current_dt = datetime.strptime(timestamp[:8], "%Y%m%d")
            prev_day_dt = current_dt - timedelta(days=1) if current_dt.weekday() != 0 else current_dt - timedelta(days=3)
            prev_week_dt = current_dt - timedelta(days=7)
            dates_file = 'data/dates.json'
            if not os.path.exists(dates_file):
                print(f"calculate_ranking_metrics: dates.json not found")
                return
            try:
                with open(dates_file, 'r') as f:
                    dates = json.load(f)
            except Exception as e:
                print(f"calculate_ranking_metrics: Error reading dates.json: {e}")
                return
            timestamps = sorted(dates, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
            prev_day_ts = None
            prev_week_ts = None
            current_index = timestamps.index(timestamp) if timestamp in timestamps else -1
            for ts in timestamps[:current_index][::-1]:
                ts_dt = datetime.strptime(ts[:8], "%Y%m%d")
                if prev_day_ts is None and ts_dt <= prev_day_dt:
                    prev_day_ts = ts
                if prev_week_ts is None and ts_dt <= prev_week_dt:
                    prev_week_ts = ts
                if prev_day_ts and prev_week_ts:
                    break

            ranking = []
            import yfinance as yf
            ticker_info_dict = {}
            tickers_file = 'tickers.txt'
            if not os.path.exists(tickers_file):
                print(f"tickers.txt not found, skipping ranking metrics calculation")
                return
            try:
                with open(tickers_file, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error reading tickers.txt: {e}")
                return

            for ticker in tickers:
                try:
                    yf_ticker = yf.Ticker(ticker)
                    ticker_info_dict[ticker] = yf_ticker.info
                except Exception as e:
                    print(f"Error fetching ticker info for {ticker}: {e}")
                    ticker_info_dict[ticker] = {'marketCap': 'N/A'}

            vol_surf_file = f'data/{timestamp}/vol_surf/vol_surf.csv'
            vol_surf_df = pd.DataFrame()
            if os.path.exists(vol_surf_file):
                try:
                    vol_surf_df = pd.read_csv(vol_surf_file)
                except Exception as e:
                    print(f"Error reading {vol_surf_file}: {e}")

            for ticker in tickers:
                ticker_historic = historic[historic['Ticker'] == ticker] if not historic.empty else pd.DataFrame()
                ticker_historic_full = ticker_historic.sort_values('Date') if not ticker_historic.empty else pd.DataFrame()
                if ticker_historic.empty:
                    print(f"No historic data for ticker {ticker}")
                    continue
                ticker_info = ticker_info_dict.get(ticker, {'marketCap': 'N/A'})
                company_name = ticker_historic['Company Name'].iloc[0] if 'Company Name' in ticker_historic.columns and not ticker_historic.empty else 'N/A'
                rank_dict = {'Ticker': ticker, 'Company Name': company_name}

                for period in ['3m', '6m', '1y']:
                    rank_dict[f'Normalized {period}'] = ticker_historic[f'Normalized {period}'].iloc[0] if f'Normalized {period}' in ticker_historic.columns and not ticker_historic.empty else 'N/A'

                summary_path = f"data/{timestamp}/tables/summary/summary_table{prefix}.csv"
                summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
                weighted_iv = 'N/A'
                weighted_iv_3m = 'N/A'
                atm_iv_3m = 'N/A'
                if not summary.empty and 'Ticker' in summary.columns:
                    ticker_summary = summary[summary['Ticker'] == ticker]
                    weighted_iv = ticker_summary['Weighted IV (%)'].iloc[0] if not ticker_summary.empty and 'Weighted IV (%)' in ticker_summary.columns else 'N/A'
                    weighted_iv_3m = ticker_summary['Weighted IV 3m (%)'].iloc[0] if not ticker_summary.empty and 'Weighted IV 3m (%)' in ticker_summary.columns else 'N/A'
                    atm_iv_3m = ticker_summary['ATM IV 3m (%)'].iloc[0] if not ticker_summary.empty and 'ATM IV 3m (%)' in ticker_summary.columns else 'N/A'
                rank_dict['Weighted IV (%)'] = weighted_iv
                rank_dict['Weighted IV 3m (%)'] = weighted_iv_3m
                rank_dict['ATM IV 3m (%)'] = atm_iv_3m
                prev_day_ranking, _, _ = get_prev_data(timestamp, 1, source)
                prev_week_ranking, _, _ = get_prev_data(timestamp, 7, source)
                prev_day_weighted_iv = prev_day_ranking[prev_day_ranking['Ticker'] == ticker]['Weighted IV (%)'].iloc[0] if not prev_day_ranking.empty and 'Weighted IV (%)' in prev_day_ranking.columns and ticker in prev_day_ranking['Ticker'].values else 'N/A'
                prev_week_weighted_iv = prev_week_ranking[prev_week_ranking['Ticker'] == ticker]['Weighted IV (%)'].iloc[0] if not prev_week_ranking.empty and 'Weighted IV (%)' in prev_week_ranking.columns and ticker in prev_week_ranking['Ticker'].values else 'N/A'
                rank_dict['Weighted IV 1d (%)'] = ((weighted_iv - prev_day_weighted_iv) / prev_day_weighted_iv * 100) if prev_day_weighted_iv != 'N/A' and prev_day_weighted_iv != 0 and weighted_iv != 'N/A' else 'N/A'
                rank_dict['Weighted IV 1w (%)'] = ((weighted_iv - prev_week_weighted_iv) / prev_week_weighted_iv * 100) if prev_week_weighted_iv != 'N/A' and prev_week_weighted_iv != 0 and weighted_iv != 'N/A' else 'N/A'
                prev_day_weighted_iv_3m = prev_day_ranking[prev_day_ranking['Ticker'] == ticker]['Weighted IV 3m (%)'].iloc[0] if not prev_day_ranking.empty and 'Weighted IV 3m (%)' in prev_day_ranking.columns and ticker in prev_day_ranking['Ticker'].values else 'N/A'
                prev_week_weighted_iv_3m = prev_week_ranking[prev_week_ranking['Ticker'] == ticker]['Weighted IV 3m (%)'].iloc[0] if not prev_week_ranking.empty and 'Weighted IV 3m (%)' in prev_week_ranking.columns and ticker in prev_week_ranking['Ticker'].values else 'N/A'
                rank_dict['Weighted IV 3m 1d (%)'] = ((weighted_iv_3m - prev_day_weighted_iv_3m) / prev_day_weighted_iv_3m * 100) if prev_day_weighted_iv_3m != 'N/A' and prev_day_weighted_iv_3m != 0 and weighted_iv_3m != 'N/A' else 'N/A'
                rank_dict['Weighted IV 3m 1w (%)'] = ((weighted_iv_3m - prev_week_weighted_iv_3m) / prev_week_weighted_iv_3m * 100) if prev_week_weighted_iv_3m != 'N/A' and prev_week_weighted_iv_3m != 0 and weighted_iv_3m != 'N/A' else 'N/A'
                prev_day_atm_iv_3m = prev_day_ranking[prev_day_ranking['Ticker'] == ticker]['ATM IV 3m (%)'].iloc[0] if not prev_day_ranking.empty and 'ATM IV 3m (%)' in prev_day_ranking.columns and ticker in prev_day_ranking['Ticker'].values else 'N/A'
                prev_week_atm_iv_3m = prev_week_ranking[prev_week_ranking['Ticker'] == ticker]['ATM IV 3m (%)'].iloc[0] if not prev_week_ranking.empty and 'ATM IV 3m (%)' in prev_week_ranking.columns and ticker in prev_week_ranking['Ticker'].values else 'N/A'
                rank_dict['ATM IV 3m 1d (%)'] = ((atm_iv_3m - prev_day_atm_iv_3m) / prev_day_atm_iv_3m * 100) if prev_day_atm_iv_3m != 'N/A' and prev_day_atm_iv_3m != 0 and atm_iv_3m != 'N/A' else 'N/A'
                rank_dict['ATM IV 3m 1w (%)'] = ((atm_iv_3m - prev_week_atm_iv_3m) / prev_week_atm_iv_3m * 100) if prev_week_atm_iv_3m != 'N/A' and prev_week_atm_iv_3m != 0 and atm_iv_3m != 'N/A' else 'N/A'

                if not ticker_historic.empty and 'Close' in ticker_historic.columns:
                    latest_historic = ticker_historic.sort_values('Date').iloc[-1:]
                    current_close = latest_historic['Close'].iloc[0] if 'Close' in latest_historic.columns else 'N/A'
                    current_open = latest_historic['Open'].iloc[0] if 'Open' in latest_historic.columns else 'N/A'
                    current_high = latest_historic['High'].iloc[0] if 'High' in latest_historic.columns else 'N/A'
                    current_low = latest_historic['Low'].iloc[0] if 'Low' in latest_historic.columns else 'N/A'
                    rank_dict['Latest Close'] = current_close
                    rank_dict['Latest Open'] = current_open
                    rank_dict['Latest High'] = current_high
                    rank_dict['Latest Low'] = current_low

                    prev_day_close = get_prev_value(ticker, prev_day_dt, 'Close', ticker_historic_full)
                    prev_week_close = get_prev_value(ticker, prev_week_dt, 'Close', ticker_historic_full)
                    prev_day_open = get_prev_value(ticker, prev_day_dt, 'Open', ticker_historic_full)
                    prev_week_open = get_prev_value(ticker, prev_week_dt, 'Open', ticker_historic_full)
                    prev_day_high = get_prev_value(ticker, prev_day_dt, 'High', ticker_historic_full)
                    prev_week_high = get_prev_value(ticker, prev_week_dt, 'High', ticker_historic_full)
                    prev_day_low = get_prev_value(ticker, prev_day_dt, 'Low', ticker_historic_full)
                    prev_week_low = get_prev_value(ticker, prev_week_dt, 'Low', ticker_historic_full)

                    rank_dict['Close 1d (%)'] = ((current_close - prev_day_close) / prev_day_close * 100) if prev_day_close != 'N/A' and prev_day_close != 0 and current_close != 'N/A' else 'N/A'
                    rank_dict['Close 1w (%)'] = ((current_close - prev_week_close) / prev_week_close * 100) if prev_week_close != 'N/A' and prev_week_close != 0 and current_close != 'N/A' else 'N/A'
                    rank_dict['Open 1d (%)'] = ((current_open - prev_day_open) / prev_day_open * 100) if prev_day_open != 'N/A' and prev_day_open != 0 and current_open != 'N/A' else 'N/A'
                    rank_dict['Open 1w (%)'] = ((current_open - prev_week_open) / prev_week_open * 100) if prev_week_open != 'N/A' and prev_week_open != 0 and current_open != 'N/A' else 'N/A'
                    rank_dict['High 1d (%)'] = ((current_high - prev_day_high) / prev_day_high * 100) if prev_day_high != 'N/A' and prev_day_high != 0 and current_high != 'N/A' else 'N/A'
                    rank_dict['High 1w (%)'] = ((current_high - prev_week_high) / prev_week_high * 100) if prev_week_high != 'N/A' and prev_week_high != 0 and current_high != 'N/A' else 'N/A'
                    rank_dict['Low 1d (%)'] = ((current_low - prev_day_low) / prev_day_low * 100) if prev_day_low != 'N/A' and prev_day_low != 0 and current_low != 'N/A' else 'N/A'
                    rank_dict['Low 1w (%)'] = ((current_low - prev_week_low) / prev_week_low * 100) if prev_week_low != 'N/A' and prev_week_low != 0 and current_low != 'N/A' else 'N/A'

                    for rvol_type in rvol_types:
                        col_name = f'Realised_Vol_Close_{rvol_type}'
                        rank_dict[f'Realised Volatility {rvol_type}d (%)'] = latest_historic[col_name].iloc[0] if col_name in latest_historic.columns and not latest_historic[col_name].isna().all() else 'N/A'
                        if rvol_type == '100':
                            past_year_start = current_dt - timedelta(days=730)
                            year_historic = ticker_historic_full[ticker_historic_full['Date'].dt.date >= past_year_start.date()]
                            if not year_historic.empty:
                                current_vol = latest_historic[col_name].iloc[0] if col_name in latest_historic.columns else 'N/A'
                                prev_day_vol = get_prev_value(ticker, prev_day_dt, col_name, ticker_historic_full)
                                prev_week_vol = get_prev_value(ticker, prev_week_dt, col_name, ticker_historic_full)
                                rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = ((current_vol - prev_day_vol) / prev_day_vol * 100) if prev_day_vol != 'N/A' and prev_day_vol != 0 and current_vol != 'N/A' else 'N/A'
                                rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = ((current_vol - prev_week_vol) / prev_week_vol * 100) if prev_week_vol != 'N/A' and prev_week_vol != 0 and current_vol != 'N/A' else 'N/A'
                                year_historic = year_historic.sort_values('Date')
                                year_historic['Close'] = pd.to_numeric(year_historic['Close'], errors='coerce')
                                if year_historic['Close'].isna().any():
                                    year_historic = year_historic.dropna(subset=['Close'])
                                log_returns = np.log(year_historic['Close'] / year_historic['Close'].shift(1)).dropna()
                                window = int(rvol_type)
                                if len(log_returns) >= window:
                                    vols = log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252) * 100
                                    vols = vols.dropna()
                                    if not vols.empty and current_vol != 'N/A' and not pd.isna(current_vol):
                                        rank_dict[f'Min Realised Volatility {rvol_type}d (2y)'] = vols.min()
                                        rank_dict[f'Max Realised Volatility {rvol_type}d (2y)'] = vols.max()
                                        rank_dict[f'Mean Realised Volatility {rvol_type}d (2y)'] = vols.mean()
                                        rank_dict[f'Rvol {rvol_type}d Percentile 2y (%)'] = (vols < current_vol).sum() / len(vols) * 100
                                        z_score = (current_vol - vols.mean()) / vols.std() if vols.std() != 0 else 0
                                        rank_dict[f'Rvol {rvol_type}d Z-Score Percentile 2y (%)'] = norm.cdf(z_score) * 100
                                    else:
                                        rank_dict[f'Min Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                                        rank_dict[f'Max Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                                        rank_dict[f'Mean Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                                        rank_dict[f'Rvol {rvol_type}d Percentile 2y (%)'] = 'N/A'
                                        rank_dict[f'Rvol {rvol_type}d Z-Score Percentile 2y (%)'] = 'N/A'
                                else:
                                    rank_dict[f'Min Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                                    rank_dict[f'Max Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                                    rank_dict[f'Mean Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                                    rank_dict[f'Rvol {rvol_type}d Percentile 2y (%)'] = 'N/A'
                                    rank_dict[f'Rvol {rvol_type}d Z-Score Percentile 2y (%)'] = 'N/A'
                            else:
                                rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = 'N/A'
                                rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = 'N/A'
                                rank_dict[f'Min Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                                rank_dict[f'Max Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                                rank_dict[f'Mean Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                                rank_dict[f'Rvol {rvol_type}d Percentile 2y (%)'] = 'N/A'
                                rank_dict[f'Rvol {rvol_type}d Z-Score Percentile 2y (%)'] = 'N/A'
                else:
                    print(f"No historic data for ticker {ticker}, setting all price-related metrics to 'N/A'")
                    rank_dict['Latest Close'] = 'N/A'
                    rank_dict['Latest Open'] = 'N/A'
                    rank_dict['Latest High'] = 'N/A'
                    rank_dict['Latest Low'] = 'N/A'
                    rank_dict['Close 1d (%)'] = 'N/A'
                    rank_dict['Close 1w (%)'] = 'N/A'
                    rank_dict['Open 1d (%)'] = 'N/A'
                    rank_dict['Open 1w (%)'] = 'N/A'
                    rank_dict['High 1d (%)'] = 'N/A'
                    rank_dict['High 1w (%)'] = 'N/A'
                    rank_dict['Low 1d (%)'] = 'N/A'
                    rank_dict['Low 1w (%)'] = 'N/A'
                    for rvol_type in rvol_types:
                        rank_dict[f'Realised Volatility {rvol_type}d (%)'] = 'N/A'
                        rank_dict[f'Realised Volatility {rvol_type}d 1d (%)'] = 'N/A'
                        rank_dict[f'Realised Volatility {rvol_type}d 1w (%)'] = 'N/A'
                        rank_dict[f'Min Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                        rank_dict[f'Max Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                        rank_dict[f'Mean Realised Volatility {rvol_type}d (2y)'] = 'N/A'
                        rank_dict[f'Rvol {rvol_type}d Percentile 2y (%)'] = 'N/A'
                        rank_dict[f'Rvol {rvol_type}d Z-Score Percentile 2y (%)'] = 'N/A'

                if 'Realised_Vol_Close_100' in ticker_historic.columns and not ticker_historic.empty:
                    current_vol = ticker_historic['Realised_Vol_Close_100'].iloc[-1]
                    rank_dict['Rvol100d - Weighted IV'] = (current_vol - (weighted_iv * 100)) if current_vol != 'N/A' and not pd.isna(current_vol) and weighted_iv != 'N/A' and not pd.isna(weighted_iv) else 'N/A'
                else:
                    rank_dict['Rvol100d - Weighted IV'] = 'N/A'

                cleaned_path = f"data/{timestamp}/cleaned_yfinance/cleaned_yfinance_{ticker}.csv"
                total_notional = np.nan
                if os.path.exists(cleaned_path):
                    try:
                        cleaned = pd.read_csv(cleaned_path)
                        if 'Bid' in cleaned.columns and 'Ask' in cleaned.columns and 'Volume' in cleaned.columns:
                            cleaned['Mid'] = (cleaned['Bid'] + cleaned['Ask']) / 2
                            total_notional = (cleaned['Mid'] * cleaned['Volume'] * 100).sum()
                        else:
                            print(f"Error calculating options notional for {ticker}: Missing Bid, Ask, or Volume columns")
                    except Exception as e:
                        print(f"Error calculating options notional for {ticker}: {e}")
                market_cap = ticker_info.get('marketCap', 'N/A')
                notional_pct = ((total_notional / market_cap) * 100) if market_cap != 'N/A' and market_cap != 0 and not np.isnan(total_notional) else 'N/A'
                rank_dict['Options Notional (% Market Cap)'] = notional_pct

                ticker_option = option_totals[option_totals['Ticker'] == ticker] if not option_totals.empty and 'Ticker' in option_totals.columns else pd.DataFrame()
                if not ticker_option.empty:
                    rank_dict['Volume'] = ticker_option['Vol'].iloc[0] if 'Vol' in ticker_option.columns else 'N/A'
                    rank_dict['Open Interest'] = ticker_option['OI'].iloc[0] if 'OI' in ticker_option.columns else 'N/A'
                    prev_day_ranking, prev_day_option, _ = get_prev_data(timestamp, 1, source)
                    prev_week_ranking, prev_week_option, _ = get_prev_data(timestamp, 7, source)
                    prev_day_option_ticker = prev_day_option[prev_day_option['Ticker'] == ticker] if not prev_day_option.empty and 'Ticker' in prev_day_option.columns else pd.DataFrame()
                    prev_week_option_ticker = prev_week_option[prev_week_option['Ticker'] == ticker] if not prev_week_option.empty and 'Ticker' in prev_week_option.columns else pd.DataFrame()
                    prev_day_volume = prev_day_option_ticker['Vol'].iloc[0] if not prev_day_option_ticker.empty and 'Vol' in prev_day_option_ticker.columns else 'N/A'
                    prev_week_volume = prev_week_option_ticker['Vol'].iloc[0] if not prev_week_option_ticker.empty and 'Vol' in prev_week_option_ticker.columns else 'N/A'
                    prev_day_oi = prev_day_option_ticker['OI'].iloc[0] if not prev_day_option_ticker.empty and 'OI' in prev_day_option_ticker.columns else 'N/A'
                    prev_week_oi = prev_week_option_ticker['OI'].iloc[0] if not prev_week_option_ticker.empty and 'OI' in prev_week_option_ticker.columns else 'N/A'
                    rank_dict['Volume 1d (%)'] = ((rank_dict['Volume'] - prev_day_volume) / prev_day_volume * 100) if prev_day_volume != 'N/A' and prev_day_volume != 0 and rank_dict['Volume'] != 'N/A' else 'N/A'
                    rank_dict['Volume 1w (%)'] = ((rank_dict['Volume'] - prev_week_volume) / prev_week_volume * 100) if prev_week_volume != 'N/A' and prev_week_volume != 0 and rank_dict['Volume'] != 'N/A' else 'N/A'
                    rank_dict['OI 1d (%)'] = ((rank_dict['Open Interest'] - prev_day_oi) / prev_day_oi * 100) if prev_day_oi != 'N/A' and prev_day_oi != 0 and rank_dict['Open Interest'] != 'N/A' else 'N/A'
                    rank_dict['OI 1w (%)'] = ((rank_dict['Open Interest'] - prev_week_oi) / prev_week_oi * 100) if prev_week_oi != 'N/A' and prev_week_oi != 0 and rank_dict['Open Interest'] != 'N/A' else 'N/A'
                else:
                    rank_dict['Volume'] = 'N/A'
                    rank_dict['Open Interest'] = 'N/A'
                    rank_dict['Volume 1d (%)'] = 'N/A'
                    rank_dict['Volume 1w (%)'] = 'N/A'
                    rank_dict['OI 1d (%)'] = 'N/A'
                    rank_dict['OI 1w (%)'] = 'N/A'

                rank_dict['Num Contracts'] = len(pd.read_csv(cleaned_path)) if os.path.exists(cleaned_path) else 'N/A'

                ticker_vol_surf = vol_surf_df[vol_surf_df['Ticker'] == ticker] if not vol_surf_df.empty and 'Ticker' in vol_surf_df.columns else pd.DataFrame()
                rank_dict['One_Yr_ATM_Rel_Error_Call (%)'] = ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Call']['One_Yr_ATM_Rel_Error_%'].iloc[0] if not ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Call'].empty else 'N/A'
                rank_dict['P90_Rel_Error_Call (%)'] = ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Call']['P90_Rel_Error_%'].iloc[0] if not ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Call'].empty else 'N/A'
                rank_dict['Restricted_P90_Rel_Error_Call (%)'] = ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Call']['Restricted_P90_Rel_Error_%'].iloc[0] if not ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Call'].empty else 'N/A'
                rank_dict['One_Yr_ATM_Rel_Error_Put (%)'] = ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Put']['One_Yr_ATM_Rel_Error_%'].iloc[0] if not ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Put'].empty else 'N/A'
                rank_dict['P90_Rel_Error_Put (%)'] = ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Put']['P90_Rel_Error_%'].iloc[0] if not ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Put'].empty else 'N/A'
                rank_dict['Restricted_P90_Rel_Error_Put (%)'] = ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Put']['Restricted_P90_Rel_Error_%'].iloc[0] if not ticker_vol_surf[ticker_vol_surf['Option_Type'] == 'Put'].empty else 'N/A'

                ranking.append(rank_dict)

            column_order = [
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
            df_ranking = pd.DataFrame(ranking)
            if not df_ranking.empty:
                # Assign Rank before selecting columns
                df_ranking['Rank'] = range(1, len(df_ranking) + 1)
                # Ensure all columns in column_order are present
                for col in column_order:
                    if col not in df_ranking.columns:
                        df_ranking[col] = 'N/A'
                df_ranking = df_ranking[column_order]
                df_ranking.replace('N/A', np.nan, inplace=True)
                df_ranking.sort_values(by='Rvol 100d Percentile 2y (%)', ascending=False, na_position='last', inplace=True)
                df_ranking['Rank'] = range(1, len(df_ranking) + 1)
                df_ranking = df_ranking.fillna('N/A')
                ranking_dir = f'data/{timestamp}/tables/ranking'
                os.makedirs(ranking_dir, exist_ok=True)
                output_file = f'{ranking_dir}/ranking{prefix}.csv'
                df_ranking.to_csv(output_file, index=False)
                print(f"Ranking metrics saved to {output_file}")
            else:
                print(f"No ranking data generated for timestamp {timestamp}")
    except Exception as e:
        print(f"Error calculating ranking metrics: {e}")

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
        sources = ['yfinance']
        calculate_ranking_metrics(timestamp, sources)
    else:
        sources = ['yfinance']
        timestamp = timestamps[-1]
        print(f"No timestamp provided, using latest: {timestamp}")
        calculate_ranking_metrics(timestamp, sources)

if __name__ == '__main__':
    main()
