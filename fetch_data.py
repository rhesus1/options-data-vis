import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import random

def setup_driver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_argument("--start-maximized")
    chrome_options.binary_location = "/usr/bin/chromium-browser"  # Explicitly set Chromium binary
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(40)
    return driver

def fetch_option_data(ticker, driver, max_retries=3):
    option_data = []
    url = f"https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}/option-chain"
    last_expiry_group = pd.to_datetime(datetime.now().year, format='%Y') # Default to current year as fallback
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to fetch option data for {ticker} from Nasdaq")
            driver.get(url)
            time.sleep(random.uniform(4, 6))
            
            # Handle cookie consent with retry and fallback
            for _ in range(5):
                try:
                    consent_banner = WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.ID, "onetrust-banner-sdk"))
                    )
                    accept_button = WebDriverWait(driver, 15).until(
                        EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                    )
                    driver.execute_script("arguments[0].click();", accept_button)
                    time.sleep(random.uniform(2, 4))
                    print(f"Cookie consent accepted for {ticker}")
                    break
                except (TimeoutException, NoSuchElementException):
                    print(f"Attempt {_ + 1} to find cookie consent banner failed for {ticker}")
                    time.sleep(random.uniform(2, 4))
            else:
                print(f"Cookie consent banner not found for {ticker}, proceeding without consent")
                # Removed 'continue' to proceed to the next steps
            
            # Scroll to ensure filters are in view
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(random.uniform(2, 3))
            
            # Wait for the table to be present
            WebDriverWait(driver, 40).until(
                EC.presence_of_element_located((By.CLASS_NAME, "jupiter22-options-chain__table"))
            )
            
            # Open and select "All" for Expiration Dates
            try:
                expiry_toggle = WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'jupiter22-option-chain-filter-toggle-month')]"))
                )
                driver.execute_script("arguments[0].click();", expiry_toggle)
                time.sleep(random.uniform(2, 3))
                
                expiry_all = WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'jupiter22-option-chain-filter-option-month') and @data-value='all']"))
                )
                driver.execute_script("arguments[0].click();", expiry_all)
                time.sleep(random.uniform(3, 5))
            except (TimeoutException, NoSuchElementException) as e:
                print(f"Failed to set expiration filter for {ticker}: {e}")
                continue
            
            # Open and select "All (Moneyness)" for Moneyness
            try:
                moneyness_toggle = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'jupiter22-option-chain-filter-toggle-moneyness')]"))
                )
                driver.execute_script("arguments[0].click();", moneyness_toggle)
                time.sleep(random.uniform(2, 3))
                
                WebDriverWait(driver, 20).until(
                    EC.visibility_of_element_located((By.CLASS_NAME, "jupiter22-option-chain-filter-options-moneyness"))
                )
                
                moneyness_all = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@class='jupiter22-option-chain-filter-option-month jupiter22-options-chain__dropdown-option' and @data-value='all' and contains(@aria-label, 'All (Moneyness)')]"))
                )
                driver.execute_script("arguments[0].click();", moneyness_all)
                time.sleep(random.uniform(3, 5))
            except (TimeoutException, NoSuchElementException) as e:
                print(f"Failed to set moneyness filter for {ticker}: {e}")
                continue
            
            while True:
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                table = soup.find('table', class_='jupiter22-options-chain__table')
                if not table:
                    print(f"No option chain table found for {ticker}")
                    break
                
                rows = table.find('tbody').find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) < 17:
                        continue
                    
                    # Check if this is a new expiry group heading
                    if cells[0].get('class', []) and 'jupiter22-options-chain__cell--expirygroup' in cells[0].get('class', []):
                        expiry_group_text = cells[0].text.strip()
                        if expiry_group_text:
                            last_expiry_group = pd.to_datetime(expiry_group_text, format='%B %d, %Y')
                    
                    expiry_date_str = cells[1].text.strip() # e.g., "Aug 15"
                    if last_expiry_group and expiry_date_str:
                        # Combine month and day with year from the last expiry group
                        expiry_date = pd.to_datetime(f"{expiry_date_str} {last_expiry_group.year}", format='%b %d %Y', errors='coerce')
                    else:
                        # Fallback to current year if no expiry group is set
                        expiry_date = pd.to_datetime(f"{expiry_date_str} {datetime.now().year}", format='%b %d %Y', errors='coerce')
                    
                    call_last = cells[2].text.strip() or np.nan
                    call_change = cells[3].text.strip() or np.nan
                    call_bid = cells[4].text.strip() or np.nan
                    call_ask = cells[5].text.strip() or np.nan
                    call_volume = cells[6].text.strip() or np.nan
                    call_open_int = cells[7].text.strip() or np.nan
                    strike = cells[9].text.strip()
                    put_last = cells[10].text.strip() or np.nan
                    put_change = cells[11].text.strip() or np.nan
                    put_bid = cells[12].text.strip() or np.nan
                    put_ask = cells[13].text.strip() or np.nan
                    put_volume = cells[14].text.strip() or np.nan
                    put_open_int = cells[15].text.strip() or np.nan
                    
                    if call_last != '--' or call_bid != '--' or call_ask != '--':
                        option_data.append({
                            "Type": "Call",
                            "Strike": float(strike) if strike.replace('.', '', 1).isdigit() else np.nan,
                            "Expiry": expiry_date,
                            "Last Price": float(call_last) if call_last.replace('.', '', 1).isdigit() else np.nan,
                            "Bid": float(call_bid) if call_bid.replace('.', '', 1).isdigit() else np.nan,
                            "Ask": float(call_ask) if call_ask.replace('.', '', 1).isdigit() else np.nan,
                            "Change": float(call_change) if call_change.replace('.', '', 1).isdigit() else np.nan,
                            "% Change": np.nan,
                            "Volume": int(call_volume) if call_volume.isdigit() else 0,
                            "Open Interest": int(call_open_int) if call_open_int.isdigit() else 0,
                            "Implied Volatility": np.nan,
                            "Contract Name": f"{ticker}_CALL_{expiry_date_str}_{strike}",
                            "Ticker": ticker,
                            "Last Trade Date": np.nan
                        })
                    
                    if put_last != '--' or put_bid != '--' or put_ask != '--':
                        option_data.append({
                            "Type": "Put",
                            "Strike": float(strike) if strike.replace('.', '', 1).isdigit() else np.nan,
                            "Expiry": expiry_date,
                            "Last Price": float(put_last) if put_last.replace('.', '', 1).isdigit() else np.nan,
                            "Bid": float(put_bid) if put_bid.replace('.', '', 1).isdigit() else np.nan,
                            "Ask": float(put_ask) if put_ask.replace('.', '', 1).isdigit() else np.nan,
                            "Change": float(put_change) if put_change.replace('.', '', 1).isdigit() else np.nan,
                            "% Change": np.nan,
                            "Volume": int(put_volume) if put_volume.isdigit() else 0,
                            "Open Interest": int(put_open_int) if put_open_int.isdigit() else 0,
                            "Implied Volatility": np.nan,
                            "Contract Name": f"{ticker}_PUT_{expiry_date_str}_{strike}",
                            "Ticker": ticker,
                            "Last Trade Date": np.nan
                        })
                
                try:
                    next_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[@class='pagination__next' and @aria-label='click to go to the next page']"))
                    )
                    if next_button.get_attribute('disabled'):
                        break
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(random.uniform(2, 4))
                except (TimeoutException, NoSuchElementException):
                    break
            
            return pd.DataFrame(option_data)
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(5, 7))
            continue
    
    return pd.DataFrame()
    
def process_ticker_fetch(ticker, driver):
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    try:
        S = stock.history(period='1d')['Close'].iloc[-1]
        info = stock.info
        bid = info.get('bid', None)
        ask = info.get('ask', None)
        mid = (bid + ask)/2 if bid is not None and ask is not None else 0
    except:
        print(f"Failed to fetch stock price for {ticker}")
        return pd.DataFrame()
    
    full_df = fetch_option_data(ticker, driver)
    if full_df.empty:
        return pd.DataFrame()
    
    full_df['Last Stock Price'] = S
    full_df['Bid Stock'] = bid
    full_df['Ask Stock'] = ask
    full_df['Mid Option'] = (full_df['Bid'] + full_df['Ask'])/2
    full_df['Mid Stock'] = (full_df['Bid Stock'] + full_df['Ask Stock'])/2 if bid is not None and ask is not None else 0
    if mid > 0:
        full_df['Moneyness'] = np.round(mid / full_df['Strike'] / 0.01) * 0.01
    else:
        full_df['Moneyness'] = np.round(S / full_df['Strike'] / 0.01) * 0.01
    
    columns = ['Ticker', 'Contract Name', 'Type', 'Expiry', 'Strike', 'Moneyness', 'Bid', 'Ask', 'Volume', 'Open Interest', 'Bid Stock', 'Ask Stock', 'Last Stock Price', 'Implied Volatility']
    return full_df[columns]

def fetch_historic_data(ticker):
    print(f"Fetching historic data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period='max')
    if hist.empty:
        return pd.DataFrame()
    hist = hist[['Close']]
    hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist['Realised_Vol_30'] = hist['Log_Return'].rolling(window=30).std() * np.sqrt(252) * 100
    hist['Realised_Vol_60'] = hist['Log_Return'].rolling(window=60).std() * np.sqrt(252) * 100
    hist['Realised_Vol_100'] = hist['Log_Return'].rolling(window=100).std() * np.sqrt(252) * 100
    hist['Realised_Vol_180'] = hist['Log_Return'].rolling(window=180).std() * np.sqrt(252) * 100
    hist['Realised_Vol_252'] = hist['Log_Return'].rolling(window=252).std() * np.sqrt(252) * 100
    hist = hist.dropna()
    hist['Date'] = hist.index.strftime('%Y-%m-%d')
    hist['Ticker'] = ticker
    return hist[['Ticker', 'Date', 'Close', 'Realised_Vol_30', 'Realised_Vol_60', 'Realised_Vol_100', 'Realised_Vol_180', 'Realised_Vol_252']]
    
def main():
    with open('tickers.txt', 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]
    
    driver = setup_driver(headless=True) # Set to False for debugging
    all_data = []
    all_hist = []
    
    try:
        for ticker in tickers:
            df = process_ticker_fetch(ticker, driver)
            if not df.empty:
                all_data.append(df)
            
            df_hist = fetch_historic_data(ticker)
            if not df_hist.empty:
                all_hist.append(df_hist)
            
            time.sleep(1) # Avoid rate limiting
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs('data', exist_ok=True)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            filename = f'data/raw_{timestamp}.csv'
            combined_df.to_csv(filename, index=False)
            print(f"Raw data saved to {filename}")
        else:
            print("No data to save")
        
        if all_hist:
            combined_hist_df = pd.concat(all_hist, ignore_index=True)
            hist_filename = f'data/historic_{timestamp}.csv'
            combined_hist_df.to_csv(hist_filename, index=False)
            print(f"Historic data saved to {hist_filename}")
        else:
            print("No historic data to save")
    
    finally:
        driver.quit()

main()
