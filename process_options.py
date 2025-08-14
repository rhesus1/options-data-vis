import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import json
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RBFInterpolator
from scipy.optimize import minimize
from joblib import Parallel, delayed
import multiprocessing
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

def black_scholes_call(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def implied_vol(price, S, K, T, r, q, option_type, contract_name=""):
    if price <= 0 or T <= 0:
        return 0.0
    intrinsic = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
    if price < intrinsic * np.exp(-r * T):
        return 0.0001
    def objective(sigma):
        if option_type.lower() == 'call':
            return black_scholes_call(S, K, T, r, q, sigma) - price
        else:
            return black_scholes_put(S, K, T, r, q, sigma) - price
    try:
        iv = brentq(objective, 0.0001, 50.0)
        return iv
    except ValueError as e:
        return np.nan

def calculate_rvol_days(ticker, days):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < days + 1:
            return None
        hist_last = hist.iloc[-(days + 1):]
        log_returns = np.log(hist_last["Close"] / hist_last["Close"].shift(1)).dropna()
        if len(log_returns) < 2:
            return None
        realised_vol = np.std(log_returns, ddof=1) * np.sqrt(252)
        return realised_vol
    except Exception as e:
        return None

def calc_Ivol_Rvol(df, rvol100d):
    if df.empty:
        return df
    df["Ivol/Rvol100d Ratio"] = df["IV_mid"] / rvol100d
    return df

def compute_ivs(row, S, r, q):
    if pd.isna(row['Years_to_Expiry']):
        return np.nan, np.nan, np.nan, np.nan
    T = max(row['Years_to_Expiry'], 0.0001)
    option_type = row['Type'].lower()
    contract_name = row['Contract Name']
    iv_bid = implied_vol(row['Bid'], S, row['Strike'], T, r, q, option_type, contract_name)
    iv_ask = implied_vol(row['Ask'], S, row['Strike'], T, r, q, option_type, contract_name)
    iv_mid = implied_vol(0.5*(row['Bid']+row['Ask']), S, row['Strike'], T, r, q, option_type, contract_name)
    iv_spread = iv_ask - iv_bid if not np.isnan(iv_bid) else np.nan
    return iv_bid, iv_ask, iv_mid, iv_spread

def calculate_metrics(df, ticker, r):
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()
    skew_data = []
    for exp in df["Expiry"].unique():
        for strike in df["Strike"].unique():
            call_iv = df[(df["Type"] == "Call") & (df["Strike"] == strike) & (df["Expiry"] == exp)]["IV_mid"]
            put_iv = df[(df["Type"] == "Put") & (df["Strike"] == strike) & (df["Expiry"] == exp)]["IV_mid"]
            if not call_iv.empty and not put_iv.empty and call_iv.iloc[0] > 0:
                skew = put_iv.iloc[0] / call_iv.iloc[0]
                skew_data.append({"Expiry": exp, "Strike": strike, "Vol Skew": f"{skew*100:.2f}%"})
    skew_df = pd.DataFrame(skew_data)
    slope_data = []
    for strike in df["Strike"].unique():
        for opt_type in ["Call", "Put"]:
            subset = df[(df["Strike"] == strike) & (df["Type"] == opt_type)].sort_values("Expiry")
            if len(subset) > 1:
                iv_diff = subset["IV_mid"].diff()
                subset["Expiry_dt"] = pd.to_datetime(subset["Expiry"])
                time_diff = (subset["Expiry_dt"] - subset["Expiry_dt"].shift(1)).map(lambda x: x.days / 365.0)
                slope = iv_diff / time_diff
                for i in range(1, len(subset)):
                    slope_data.append({
                        "Strike": strike,
                        "Type": opt_type,
                        "Expiry": subset["Expiry"].iloc[i],
                        "IV Slope": slope.iloc[i]
                    })
    slope_df = pd.DataFrame(slope_data)
    return df, skew_df, slope_df

def calculate_iv_mid(df, ticker, r):
    if df.empty:
        return df, None, None, None
    stock = yf.Ticker(ticker)
    S = stock.history(period='1d')['Close'].iloc[-1]
    q = float(stock.info.get('trailingAnnualDividendYield', 0.0))
    today = datetime.today()
    df["Expiry_dt"] = df["Expiry"]
    df['Years_to_Expiry'] = (df['Expiry_dt'] - today).dt.days / 365.25
    df['Forward'] = S * np.exp((r - q) * df['Years_to_Expiry'])
    df['LogMoneyness'] = np.log(df['Strike'] / df['Forward'])
    df['IV_bid'] = np.nan
    df['IV_ask'] = np.nan
    df['IV_mid'] = np.nan
    df['IV_spread'] = np.nan
    results = Parallel(n_jobs=-1, backend='threading')(delayed(compute_ivs)(row, S, r, q) for _, row in df.iterrows())
    df[['IV_bid', 'IV_ask', 'IV_mid', 'IV_spread']] = pd.DataFrame(results, index=df.index)
    return df, S, r, q

def compute_local_vol_from_iv_row(row, r, q, interp):
    y = row['LogMoneyness']
    T = row['Years_to_Expiry']
    if T <= 0:
        return None
    
    w = interp(np.array([[y, T]]))[0]
    if np.isnan(w):
        return None
    
    h_t = max(0.01 * T, 1e-4)  # Adaptive, with min to avoid zero
    h_y = max(0.01 * abs(y) if y != 0 else 0.01, 1e-4)  # Adaptive to moneyness scale
    
    # ∂w/∂T
    w_T_plus = interp(np.array([[y, T + h_t]]))[0]
    w_T_minus = interp(np.array([[y, max(T - h_t, 1e-6)]]))[0]
    if np.isnan(w_T_plus) or np.isnan(w_T_minus):
        return None
    dw_dT = (w_T_plus - w_T_minus) / (2 * h_t)
    
    # ∂w/∂y
    w_y_plus = interp(np.array([[y + h_y, T]]))[0]
    w_y_minus = interp(np.array([[y - h_y, T]]))[0]
    if np.isnan(w_y_plus) or np.isnan(w_y_minus):
        return None
    dw_dy = (w_y_plus - w_y_minus) / (2 * h_y)
    
    # ∂²w/∂y²
    d2w_dy2 = (w_y_plus - 2 * w + w_y_minus) / (h_y ** 2)
    
    if np.isnan(dw_dT) or np.isnan(dw_dy) or np.isnan(d2w_dy2):
        return None
    
    # Dupire formula for local vol from IV
    denom = 1 - (y / w) * dw_dy + 0.25 * (-0.25 - 1/w + (y**2 / w**2)) * (dw_dy ** 2) + 0.5 * d2w_dy2
    if denom <= 1
