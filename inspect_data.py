import requests
import json

# We check 2025 and 2026 to be safe (Season is 2025-26)
YEARS = [2026]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://barttorvik.com/"
}

def inspect():
    print("--- 🕵️ AUDITING RAW DATA ---")
    
    for year in YEARS:
        url = f"https://barttorvik.com/{year}_super_sked.json"
        print(f"\nDownloading {year} data from: {url} ...")
        
        try:
            res = requests.get(url, headers=HEADERS)
            data = res.json()
            
            if not data:
                print("❌ No data found.")
                continue
                
            print(f"✅ Found {len(data)} games.")
            print("\n--- SAMPLE RAW LINES (Index 4) ---")
            print(f"{'Game Matchup':<40} | {'Raw Line String (Index 4)':<30}")
            print("-" * 80)
            
            # Print first 20 valid rows
            count = 0
            for row in data:
                if count >= 20: break
                
                # Extract names for context
                # Box score is usually at index 50
                if len(row) > 50:
                    box = row[50]
                    if isinstance(box, list) and len(box) > 5:
                        matchup = f"{box[2]} vs {box[3]}"
                    else:
                        matchup = "Unknown"
                else:
                    matchup = "Unknown"
                
                # THE KEY: Index 4 holds the line
                raw_line = row[4]
                
                # Skip empty lines to find the interesting ones
                if raw_line == "": continue
                
                print(f"{matchup:<40} | {str(raw_line):<30}")
                count += 1
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    inspect()