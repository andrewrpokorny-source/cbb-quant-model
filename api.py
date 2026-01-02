import cloudscraper
import pandas as pd
import io
import time
from config import CURRENT_SEASON

BASE_URL = "https://barttorvik.com"

# Create a scraper instance that mimics a real desktop browser
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

def fetch_season_games(year):
    """
    Fetches detailed game logs for a specific season.
    Uses cloudscraper to bypass 'Verifying Browser' checks.
    """
    url = f"{BASE_URL}/get-gamestats.php?year={year}&csv=1"
    
    print(f"   -> 📡 Fetching data for {year}...")
    try:
        # Use scraper.get() instead of requests.get()
        response = scraper.get(url)
        response.raise_for_status()
        
        # Debug check for HTML response
        if "<!DOCTYPE html>" in response.text[:100]:
             print("      ⚠️ STILL BLOCKED. The site requires a stronger bypass.")
             return pd.DataFrame()

        # Parse CSV
        df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip')
        return df
        
    except Exception as e:
        print(f"   ❌ Error fetching {year}: {e}")
        return pd.DataFrame()

def fetch_live_lines():
    """
    Fetches today's team ratings (T-Rank).
    """
    url = f"{BASE_URL}/{CURRENT_SEASON}_team_results.json"
    try:
        res = scraper.get(url)
        return pd.json_normalize(res.json())
    except:
        return pd.DataFrame()