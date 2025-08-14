import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os

# =========================
# CONFIGURATION
# =========================
API_URL = "https://dps.psx.com.pk/market-watch"  # Replace with your actual API URL
OUTPUT_FILE = "stocks_data.xlsx"

# =========================
# FETCH DATA FROM API
# =========================
response = requests.get(API_URL)
response.raise_for_status()

# Parse HTML table
soup = BeautifulSoup(response.text, "html.parser")
table_body = soup.find("tbody", class_="tbl__body")

rows = []
for tr in table_body.find_all("tr"):
    tds = tr.find_all("td")
    if len(tds) < 11:
        continue  # Skip if not enough columns

    symbol = tds[0].get("data-order", "").strip()
    sector = tds[1].text.strip()
    listed_in = tds[2].text.strip()
    ldcp = tds[3].text.strip()
    open_price = tds[4].text.strip()
    high = tds[5].text.strip()
    low = tds[6].text.strip()
    current = tds[7].text.strip()
    change = tds[8].text.strip()
    change_percent = tds[9].text.strip()
    volume = tds[10].text.strip()

    rows.append({
        "SYMBOL": symbol,
        "SECTOR": sector,
        "LISTED IN": listed_in,
        "LDCP": ldcp,
        "OPEN": open_price,
        "HIGH": high,
        "LOW": low,
        "CURRENT": current,
        "CHANGE": change,
        "CHANGE (%)": change_percent,
        "VOLUME": volume,
        "FETCHED AT": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# =========================
# SAVE TO EXCEL
# =========================
df_new = pd.DataFrame(rows)

if os.path.exists(OUTPUT_FILE):
    # Append to existing file
    df_old = pd.read_excel(OUTPUT_FILE)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.to_excel(OUTPUT_FILE, index=False)
else:
    # Create new file
    df_new.to_excel(OUTPUT_FILE, index=False)

print(f"✅ Data saved to {OUTPUT_FILE} with {len(rows)} new records.")
