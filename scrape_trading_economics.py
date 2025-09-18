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

def scrape_page(url, page_type, headers=None, expected_cols=None, headers_9=None, expected_cols_9=None, headers_10=None, expected_cols_10=None, has_extra_col=False, retries=3):
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
                if attempt == retries:
                    print(f"Failed to find tables after {retries} attempts for {url}")
                    return
                time.sleep(5)  # Wait before retrying
                continue
            # Parse page content
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            if check_for_bot_detection(soup):
                print(f"Bot detection or CAPTCHA likely encountered on {url}")
                if attempt == retries:
                    print(f"Failed to bypass bot detection after {retries} attempts for {url}")
                    return
                time.sleep(5)
                continue
            tables = soup.find_all('table')
            if debug:
                print(f"Found {len(tables)} tables on {url}")
            processed_count = 0
            for t_idx, table in enumerate(tables):
                header_row = table.find('tr')
                if not header_row:
                    if debug:
                        print(f"Table {t_idx+1} has no header row, skipping")
                    continue
                header_cells = header_row.find_all(['th', 'td'])
                if not header_cells:
                    if debug:
                        print(f"Table {t_idx+1} has no header cells, skipping")
                    continue
                # Determine category
                if has_extra_col:
                    category = header_cells[1].get_text().strip() if len(header_cells) > 1 else f"table_{t_idx+1}"
                else:
                    category = header_cells[0].get_text().strip() if header_cells else f"table_{t_idx+1}"
                if not category:
                    category = f"table_{t_idx+1}"
                clean_category = re.sub(r'[^a-zA-Z0-9]', '_', category).lower()
                output_file = f"{clean_category}_{page_type}.csv"
                if debug:
                    print(f"Processing table {t_idx+1}: category '{category}', file: {output_file}")
                # Determine headers and expected columns for this table
                table_headers_list = [h.get_text().strip() for h in header_cells]
                if page_type == "crypto":
                    if any('market' in h.lower() and 'cap' in h.lower() for h in table_headers_list):
                        this_headers = headers_10
                        this_expected = expected_cols_10
                    else:
                        this_headers = headers_9
                        this_expected = expected_cols_9
                else:
                    this_headers = headers
                    this_expected = expected_cols
                # Process rows
                rows = table.find_all('tr')[1:]
                all_data = []
                start_idx = 1 if has_extra_col else 0
                for j, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < start_idx + 2:
                        continue
                    row_data = [re.sub(r'\s+', ' ', cell.get_text().strip()) for cell in cells]
                    if len(row_data) < start_idx + this_expected or not row_data[start_idx + 1].strip():
                        if debug and j < 3:
                            print(f"Skipped row in table {t_idx+1}: too short or no value {row_data[:5]}")
                        continue
                    full_row = row_data[start_idx : start_idx + this_expected]
                    while len(full_row) < this_expected:
                        full_row.append('')
                    full_row = full_row[:this_expected]
                    if full_row[1].strip():
                        all_data.append(full_row)
                        if debug and len(all_data) <= 3:
                            print(f"Sample row in {output_file}: {full_row}")
                # Deduplicate
                unique_data = []
                seen_identifiers = set()
                for row in all_data:
                    identifier = row[0].lower().strip()
                    if identifier and identifier not in seen_identifiers:
                        unique_data.append(row)
                        seen_identifiers.add(identifier)
                        if debug:
                            print(f"Unique row added: {row[0]}")
                # Write to CSV
                if unique_data:
                    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(this_headers)
                        for row in unique_data:
                            writer.writerow(row)
                    print(f"Data scraped for table {t_idx+1} ({category}): {len(unique_data)} unique rows saved to {output_file}")
                    processed_count += 1
                    if len(unique_data) > 0 and debug:
                        print("First few rows:")
                        for row in unique_data[:3]:
                            print(row)
                else:
                    print(f"No valid data found for table {t_idx+1} ({category})")
            if processed_count == 0:
                print(f"No tables processed successfully for {url}")
            break  # Exit retry loop on success
        except Exception as e:
            print(f"An error occurred during scraping {url} on attempt {attempt}: {e}")
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
            "page_type": "bonds",
            "headers": ["Country", "Yield", "Day", "Weekly", "Monthly", "YTD", "YoY", "Date"],
            "expected_cols": 8,
            "has_extra_col": True
        },
        {
            "url": "https://tradingeconomics.com/commodities",
            "page_type": "commodities",
            "headers": ["Commodity", "Price", "Day", "%", "Weekly", "Monthly", "YTD", "YoY", "Date"],
            "expected_cols": 9,
            "has_extra_col": False
        },
        {
            "url": "https://tradingeconomics.com/stocks",
            "page_type": "stocks",
            "headers": ["Index", "Price", "Day", "%", "Weekly", "Monthly", "YTD", "YoY", "Date"],
            "expected_cols": 9,
            "has_extra_col": True
        },
        {
            "url": "https://tradingeconomics.com/shares",
            "page_type": "shares",
            "headers": ["Company", "Price", "Chg", "%Chg", "YoY"],
            "expected_cols": 5,
            "has_extra_col": False
        },
        {
            "url": "https://tradingeconomics.com/currencies",
            "page_type": "currencies",
            "headers": ["Pair", "Price", "Day", "%", "Weekly", "Monthly", "YTD", "YoY", "Date"],
            "expected_cols": 9,
            "has_extra_col": True
        },
        {
            "url": "https://tradingeconomics.com/crypto",
            "page_type": "crypto",
            "headers_9": ["Name", "Price", "Day", "%", "Weekly", "Monthly", "YTD", "YoY", "Date"],
            "expected_cols_9": 9,
            "headers_10": ["Crypto", "Price", "Day", "%", "Weekly", "Monthly", "YTD", "YoY", "MarketCap", "Date"],
            "expected_cols_10": 10,
            "has_extra_col": False
        }
    ]
    for page_config in pages:
        print(f"\nScraping {page_config['url']}...")
        scrape_page(
            url=page_config["url"],
            page_type=page_config["page_type"],
            headers=page_config.get("headers"),
            expected_cols=page_config.get("expected_cols"),
            headers_9=page_config.get("headers_9"),
            expected_cols_9=page_config.get("expected_cols_9"),
            headers_10=page_config.get("headers_10"),
            expected_cols_10=page_config.get("expected_cols_10"),
            has_extra_col=page_config.get("has_extra_col", False),
            retries=3
        )

if __name__ == "__main__":
    main()
