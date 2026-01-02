import pandas as pd
import requests
import time
import numpy as np
import re

def calculate_four_factors(fgm, fga, three_pm, fta, orb, opp_drb, to, poss):
    efg = (fgm + 0.5 * three_pm) / fga if fga > 0 else 0
    to_pct = to / poss if poss > 0 else 0
    orb_pct = orb / (orb + opp_drb) if (orb + opp_drb) > 0 else 0
    ftr = fta / fga if fga > 0 else 0
    return efg * 100, to_pct * 100, orb_pct * 100, ftr * 100

def parse_lines(text_str):
    """
    Robust parser for Totals.
    Finds numbers like 145, 145.5, 98.5.
    Range 90-250 to be safe.
    """
    # Regex: Look for 2 or 3 digits, optional decimal.
    # Exclude small numbers (spreads) by enforcing range check.
    candidates = re.findall(r'\b(\d{2,3}\.?\d?)\b', str(text_str))
    
    for c in candidates:
        try:
            val = float(c)
            # Valid NCAA Totals range
            if 90 <= val <= 250:
                return val
        except: continue
    return np.nan

def parse_spread_val(text_str, home_team):
    try:
        part1 = text_str.split(',')[0].strip()
        match = re.search(r'([+-]?\d+\.?\d*)$', part1)
        if match:
            raw = float(match.group(1))
            if part1.startswith(home_team) or home_team in part1: return raw
            else: return raw * -1
        return np.nan
    except: return np.nan

def fetch_cbb_data(years):
    all_games = []
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://barttorvik.com/"
    }

    print(f"--- 🏀 FETCHING DATA (ROBUST TOTALS) ---")

    for year in years:
        print(f"   -> Downloading {year} season data...")
        url = f"https://barttorvik.com/{year}_super_sked.json"
        
        try:
            response = requests.get(url, headers=HEADERS)
            data = response.json()
            if not data: continue
            
            processed_rows = []
            
            for row in data:
                if len(row) < 50: continue
                box = row[50]
                if not isinstance(box, list) or len(box) < 34: continue

                try:
                    s1, s2 = float(box[18]), float(box[33])
                    if s1 == 0 and s2 == 0: continue
                    away_score, home_score = int(s1), int(s2)
                    possessions = float(box[34]) if box[34] else 70.0
                except: continue

                date = row[1]
                away_name, home_name = box[2], box[3]
                
                # LINES
                raw_str = str(row[4])
                spread = parse_spread_val(raw_str, home_name)
                total_line = parse_lines(raw_str)
                
                # We need valid spreads, but we can tolerate missing totals (fill with NaN)
                if pd.isna(spread): continue

                # Stats
                a_fgm, a_fga, a_3pm, a_3pa = box[4], box[5], box[6], box[7]
                a_fta, a_orb, a_to = box[9], box[10], box[16]
                h_drb = box[26]
                
                h_fgm, h_fga, h_3pm, h_3pa = box[19], box[20], box[21], box[22]
                h_fta, h_orb, h_to = box[24], box[25], box[31]
                a_drb = box[11]
                
                a_factors = calculate_four_factors(a_fgm, a_fga, a_3pm, a_fta, a_orb, h_drb, a_to, possessions)
                h_factors = calculate_four_factors(h_fgm, h_fga, h_3pm, h_fta, h_orb, a_drb, h_to, possessions)
                
                a_3pr = (a_3pa / a_fga * 100) if a_fga > 0 else 0
                h_3pr = (h_3pa / h_fga * 100) if h_fga > 0 else 0
                
                # Common data
                base = {
                    "date": date, "season": year,
                    "possessions": possessions, "total_line": total_line
                }
                
                # AWAY ROW
                r1 = base.copy()
                r1.update({
                    "team": away_name, "opponent": home_name, "is_home": 0,
                    "team_score": away_score, "opp_score": home_score,
                    "spread": spread * -1,
                    "team_eFG": a_factors[0], "team_TO": a_factors[1], "team_ORB": a_factors[2], "team_FTR": a_factors[3], "team_3PR": a_3pr,
                    "opp_eFG": h_factors[0], "opp_TO": h_factors[1], "opp_ORB": h_factors[2], "opp_FTR": h_factors[3], "opp_3PR": h_3pr
                })
                processed_rows.append(r1)

                # HOME ROW
                r2 = base.copy()
                r2.update({
                    "team": home_name, "opponent": away_name, "is_home": 1,
                    "team_score": home_score, "opp_score": away_score,
                    "spread": spread,
                    "team_eFG": h_factors[0], "team_TO": h_factors[1], "team_ORB": h_factors[2], "team_FTR": h_factors[3], "team_3PR": h_3pr,
                    "opp_eFG": a_factors[0], "opp_TO": a_factors[1], "opp_ORB": a_factors[2], "opp_FTR": a_factors[3], "opp_3PR": a_3pr
                })
                processed_rows.append(r2)

            df = pd.DataFrame(processed_rows)
            df = df.drop_duplicates(subset=['date', 'team'])
            all_games.append(df)
            time.sleep(1)

        except Exception as e:
            print(f"      ❌ Error processing {year}: {e}")

    if not all_games: return None
    master_df = pd.concat(all_games, ignore_index=True)
    master_df['date'] = pd.to_datetime(master_df['date'])
    return master_df

if __name__ == "__main__":
    YEARS = [2024, 2025, 2026]
    df = fetch_cbb_data(YEARS)
    if df is not None:
        df.to_csv("cbb_training_data_raw.csv", index=False)
        print(f"\n✅ SUCCESS: Saved {len(df)} games with ROBUST Totals.")