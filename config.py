import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Spoofing a standard Chrome browser on Windows to avoid bot detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://barttorvik.com/"
}

# The Betting Market
VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet']

# File Paths
RAW_DATA_FILE = "cbb_data_raw.csv"
TRAINING_DATA_FILE = "cbb_data_processed.csv"
HISTORY_FILE = "cbb_predictions_log.csv"

# Model Settings
START_YEAR = 2024 
CURRENT_SEASON = 2026