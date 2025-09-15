import csv
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager  # Automatic driver download

debug = True  # Enable verbose output for debugging in GitHub Actions

def handle_cookie_consent(driver):
    consent_selectors = [
        "//button[contains(@class, 'fc-cta-consent')]",
        "//button[@aria-label='Consent']",
        "//button[contains(text(), 'Consent')]",
        "//button[contains(@class, 'fc-primary-button')]",
        "//button[contains(text(), 'Accept') or contains(text(), 'Agree')]"
    ]
    consent_clicked = False
    for selector in consent_selectors:
        try:
            consent_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, selector))
            )
            consent_button.click()
            if debug:
                print(f"Accepted cookies using selector: {selector}")
            consent_clicked = True
            time.sleep(2)
            break
        except Exception as e:
            if debug:
                print(f"Failed to click consent button with selector {selector}: {e}")
    if not consent_clicked:
        if debug:
            print("No cookie consent popup found with any selector")
    return consent_clicked

def check_for_bot_detection(soup):
    """Check for common bot detection or CAPTCHA indicators in the page."""
    bot_indicators = ['cloudflare', 'captcha', 'access denied', 'please verify you are not a robot']
    page_text = soup.get_text().lower()
    for indicator in bot_indicators:
        if indicator in page_text:
            if debug:
                print(f"Bot detection indicator found: {indicator}")
            return True
    return False

def scrape_page(url, output_prefixes, headers, identifier_col=0, skip_cols=0, expected_cols=8, is_shares=False, retries=3):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Use new headless mode for better compatibility
    chrome_options.add_argument("--no-sandbox")  # Required for GitHub Actions
    chrome_options.add_argument("--disable-dev-shm-usage")  # Prevent resource issues
    chrome_options.add_argument("--disable-gpu")  # Disable GPU in headless mode
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")  # Mimic real browser
    chrome_options.add_argument("--window-size=1920,1080")  # Set standard resolution
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Hide Selenium automation
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])  # Avoid detection

    for attempt in range(1, retries + 1):
        driver = None
        try:
            if debug:
                print(f"Attempt {attempt} of {retries} for {url}")
            # Use webdriver-manager to automatically download matching ChromeDriver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.delete_all_cookies()
            if debug:
                print(f"Cleared cookies for {url}")

            driver.get(url)
            if debug:
                print(f"Navigated to {url}")

            handle_cookie_consent(driver)

            # Wait for tables or a reasonable amount of time
            try:
                WebDriverWait(driver, 30).until(  # Increased timeout to 30 seconds
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                if debug:
                    print("At least one table found on the page")
            except Exception as e:
                print(f"No tables found within wait time on attempt {attempt}: {e}")
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                if check_for_bot_detection(soup):
                    print(f"Bot detection or CAPTCHA likely encountered on {url}")
                with open(f"{output_prefixes[0]}_page_source_attempt_{attempt}.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                print(f"Saved page source to {output_prefixes[0]}_page_source_attempt_{attempt}.html")
                if attempt == retries:
                    print(f"Failed to find tables after {retries} attempts for {url}")
                    return
                time.sleep(5)  # Wait before retrying
                continue

            # Parse page content
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            if check_for_bot_detection(soup):
                print(f"Bot detection or CAPTCHA likely encountered on {url}")
                with open(f"{output_prefixes[0]}_page_source_attempt_{attempt}.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                print(f"Saved page source to {output_prefixes[0]}_page_source_attempt_{attempt}.html")
                if attempt == retries:
                    print(f"Failed to bypass bot detection after {retries} attempts for {url}")
                    return
                time.sleep(5)
                continue

            tables = soup.find_all('table')
            if debug:
                print(f"Found {len(tables)} tables on {url}")

            if is_shares:
                if len(tables) >= 2:
                    for t_idx, table in enumerate(tables[:2]):  # First two tables
                        all_data = []
                        header_row = table.find('tr')
                        if not header_row:
                            if debug:
                                print(f"Table {t_idx+1} has no rows, skipping")
                            continue
                        table_headers = [th.get_text().strip() if th else td.get_text().strip() for th, td in zip(header_row.find_all('th'), header_row.find_all('td')) if th or td]
                        if debug:
                            print(f"Table {t_idx+1} headers: {table_headers}")
                        rows = table.find_all('tr')[1:]
                        for j, row in enumerate(rows):
                            cells = row.find_all(['td', 'th'])
                            if not cells:
                                continue
                            row_data = [cell.get_text().strip() for cell in cells]
                            if debug:
                                print(f"Raw row data (table {t_idx+1}, row {j+1}): {row_data}")
                            if len(row_data) > skip_cols:
                                identifier = row_data[skip_cols].replace('\n', ' ').strip() if len(row_data) > skip_cols else ''
                                metrics = row_data[skip_cols + 1:] if skip_cols + 1 < len(row_data) else []
                                full_row = [identifier] + metrics[:expected_cols - 1]
                                while len(full_row) < expected_cols:
                                    full_row.append('')
                                full_row = full_row[:expected_cols]
                                full_row = [re.sub(r'\s+', ' ', cell) for cell in full_row]
                                if full_row[1]:
                                    all_data.append(full_row)
                                    if debug:
                                        print(f"Added row: {full_row[0]} - {full_row[1]}")
                                else:
                                    if debug:
                                        print(f"Skipped row (no primary value): {full_row}")
                        # Deduplicate
                        unique_data = []
                        seen_identifiers = set()
                        for row in all_data:
                            identifier = row[identifier_col].lower().strip()
                            if identifier and identifier not in seen_identifiers:
                                unique_data.append(row)
                                seen_identifiers.add(identifier)
                                if debug:
                                    print(f"Unique row added: {row[0]}")
                        # Write to CSV
                        output_file = f"{output_prefixes[t_idx]}.csv"
                        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(headers)
                            for row in unique_data:
                                writer.writerow(row)
                        print(f"Data scraped for table {t_idx+1}: {len(unique_data)} unique rows saved to {output_file}")
                        if len(unique_data) > 0 and debug:
                            print("First few rows:")
                            for row in unique_data[:5]:
                                print(row)
                else:
                    print(f"Expected at least 2 tables for shares, but found {len(tables)}.")
            else:
                all_data = []
                for i, table in enumerate(tables):
                    if debug:
                        print(f"Processing table {i+1}")
                    header_row = table.find('tr')
                    if not header_row:
                        if debug:
                            print(f"Table {i+1} has no rows, skipping")
                        continue
                    table_headers = [th.get_text().strip() if th else td.get_text().strip() for th, td in zip(header_row.find_all('th'), header_row.find_all('td')) if th or td]
                    if debug:
                        print(f"Table {i+1} headers: {table_headers}")
                    rows = table.find_all('tr')[1:]
                    for j, row in enumerate(rows):
                        cells = row.find_all(['td', 'th'])
                        if not cells:
                            continue
                        row_data = [cell.get_text().strip() for cell in cells]
                        if debug:
                            print(f"Raw row data (table {i+1}, row {j+1}): {row_data}")
                        if len(row_data) > skip_cols:
                            identifier = row_data[skip_cols].replace('\n', ' ').strip() if len(row_data) > skip_cols else ''
                            metrics = row_data[skip_cols + 1:] if skip_cols + 1 < len(row_data) else []
                            full_row = [identifier] + metrics[:expected_cols - 1]
                            while len(full_row) < expected_cols:
                                full_row.append('')
                            full_row = full_row[:expected_cols]
                            full_row = [re.sub(r'\s+', ' ', cell) for cell in full_row]
                            if full_row[1]:
                                all_data.append(full_row)
                                if debug:
                                    print(f"Added row: {full_row[0]} - {full_row[1]}")
                            else:
                                if debug:
                                    print(f"Skipped row (no primary value): {full_row}")
                # Deduplicate
                unique_data = []
                seen_identifiers = set()
                for row in all_data:
                    identifier = row[identifier_col].lower().strip()
                    if identifier and identifier not in seen_identifiers:
                        unique_data.append(row)
                        seen_identifiers.add(identifier)
                        if debug:
                            print(f"Unique row added: {row[0]}")
                # Write to CSV
                output_file = f"{output_prefixes[0]}.csv"
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)
                    for row in unique_data:
                        writer.writerow(row)
                print(f"Data scraped: {len(unique_data)} unique rows saved to {output_file}")
                if len(unique_data) > 0 and debug:
                    print("First few rows:")
                    for row in unique_data[:5]:
                        print(row)
            # Save page source for debugging
            with open(f"{output_prefixes[0]}_page_source_success.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print(f"Saved page source to {output_prefixes[0]}_page_source_success.html")
            break  # Exit retry loop on success
        except Exception as e:
            print(f"An error occurred during scraping {url} on attempt {attempt}: {e}")
            if driver:
                with open(f"{output_prefixes[0]}_page_source_attempt_{attempt}.html", "w", encoding='utf-8') as f:
                    f.write(driver.page_source)
                print(f"Saved page source to {output_prefixes[0]}_page_source_attempt_{attempt}.html")
            if attempt == retries:
                print(f"Failed to scrape {url} after {retries} attempts")
        finally:
            if driver:
                driver.quit()
            time.sleep(2)  # Brief pause between attempts

def main():
    pages = [
        {
            "url": "https://tradingeconomics.com/bonds",
            "output_prefixes": ["bonds"],
            "headers": ["Country", "Yield", "Day", "Weekly", "Monthly", "YTD", "YoY", "Date"],
            "identifier_col": 0,
            "skip_cols": 1,
            "expected_cols": 8
        },
        {
            "url": "https://tradingeconomics.com/commodities",
            "output_prefixes": ["commodities"],
            "headers": ["Commodity", "Price", "Day", "Weekly", "Monthly", "YTD", "YoY", "Date"],
            "identifier_col": 0,
            "skip_cols": 0,
            "expected_cols": 8
        },
        {
            "url": "https://tradingeconomics.com/stocks",
            "output_prefixes": ["stocks"],
            "headers": ["Index", "Price", "Day", "%", "Weekly", "Monthly", "YTD", "YoY", "Date"],
            "identifier_col": 0,
            "skip_cols": 1,
            "expected_cols": 9
        },
        {
            "url": "https://tradingeconomics.com/shares",
            "output_prefixes": ["shares_gainers", "shares_losers"],
            "headers": ["Company", "Price", "Chg", "%Chg", "YoY"],
            "identifier_col": 0,
            "skip_cols": 0,
            "expected_cols": 5,
            "is_shares": True
        },
        {
            "url": "https://tradingeconomics.com/currencies",
            "output_prefixes": ["currencies"],
            "headers": ["Pair", "Price", "Day", "Weekly", "Monthly", "YTD", "YoY", "Date"],
            "identifier_col": 0,
            "skip_cols": 1,
            "expected_cols": 8
        },
        {
            "url": "https://tradingeconomics.com/crypto",
            "output_prefixes": ["crypto"],
            "headers": ["Crypto", "Price", "Day", "Weekly", "Monthly", "YTD", "YoY", "MarketCap", "Date"],
            "identifier_col": 0,
            "skip_cols": 0,
            "expected_cols": 9
        }
    ]
    for page in pages:
        print(f"\nScraping {page['url']}...")
        scrape_page(
            url=page["url"],
            output_prefixes=page["output_prefixes"],
            headers=page["headers"],
            identifier_col=page["identifier_col"],
            skip_cols=page["skip_cols"],
            expected_cols=page["expected_cols"],
            is_shares=page.get("is_shares", False),
            retries=3
        )

if __name__ == "__main__":
    main()
