import argparse
import os
import json
from difflib import get_close_matches
from datetime import datetime, timedelta
import re

import pandas as pd
import pytz
import requests

from betting import VALUE_RATINGS
from league_config import (
    get_league_artifact_paths,
    get_scoreboard_base_url,
    normalize_league,
)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_team_name(name):
    """Normalize team names for matching."""
    # Remove common suffixes
    name = name.replace(" Tigers", "").replace(" Bulldogs", "").replace(" Eagles", "")
    name = name.replace(" Wildcats", "").replace(" Cardinals", "")
    # Remove state suffixes
    name = name.replace(" State", "").replace(" St.", "").replace(" St", "")
    return name.strip()

def fetch_completed_games(date_obj, league="mens"):
    """
    Fetch completed games for a specific date from ESPN.
    Returns dict: {(home_team, away_team): {home_score, away_score, spread}}
    """
    league = normalize_league(league)
    base_url = get_scoreboard_base_url(league)

    print(f"   -> Fetching completed games for {date_obj.strftime('%Y-%m-%d')}...")
    
    date_str = date_obj.strftime("%Y%m%d")
    url = f"{base_url}&dates={date_str}"
    
    games = {}
    
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        for event in data.get('events', []):
            # Only process completed games
            status = event['status']['type']['state']
            if status != 'post':
                continue
            
            comp = event['competitions'][0]
            if not comp.get('competitors'):
                continue
            
            home_tm = comp['competitors'][0]
            away_tm = comp['competitors'][1]
            
            home_name = home_tm['team']['displayName']
            away_name = away_tm['team']['displayName']
            
            home_score = int(home_tm['score'])
            away_score = int(away_tm['score'])
            
            # Get the spread (if available)
            spread = 0.0
            if comp.get('odds'):
                odds = comp['odds'][0]
                details = odds.get('details', '0')
                try:
                    if details and details != '0' and details != 'EVEN':
                        parts = details.split()
                        val = abs(float(parts[-1]))
                        fav = " ".join(parts[:-1])
                        
                        home_abbr = home_tm['team'].get('abbreviation', '')
                        is_home_fav = (fav == home_abbr) or (fav == home_name) or (fav in home_name)
                        
                        spread = -val if is_home_fav else val
                except (ValueError, IndexError):
                    print(
                        f"      WARNING: Could not parse closing spread '{details}' "
                        f"for {away_name} @ {home_name}"
                    )
            
            # Extract game start time in Eastern for doubleheader disambiguation.
            # ESPN returns UTC; predictions display Eastern -- must match.
            game_start = event.get("date", "")
            game_time_str = ""
            if game_start:
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    import pytz as _pytz
                    gdt = _dt.fromisoformat(game_start.replace("Z", "+00:00"))
                    eastern = _pytz.timezone("US/Eastern")
                    gdt_eastern = gdt.astimezone(eastern)
                    game_time_str = gdt_eastern.strftime("%H:%M")
                except (ValueError, TypeError, ImportError):
                    pass

            game_key = (home_name, away_name)
            game_data = {
                'home_score': home_score,
                'away_score': away_score,
                'spread': spread,
                'home_name': home_name,
                'away_name': away_name,
                'game_time': game_time_str,
            }
            # Handle doubleheaders: if key exists, use (home, away, N)
            if game_key in games:
                games[(home_name, away_name, 2)] = game_data
            else:
                games[game_key] = game_data
            
        print(f"      Found {len(games)} completed games")
        return games
        
    except requests.HTTPError as e:
        print(f"      ESPN HTTP error fetching games: {e}")
        return {}
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        print(f"      ESPN fetch/parse error: {e}")
        return {}

def match_prediction_to_game(pred_matchup, games, pred_time=""):
    """
    Try to match a prediction matchup to an actual game.
    pred_matchup format: "Away @ Home"
    pred_time: optional "HH:MM" to disambiguate doubleheaders
    """
    # Parse prediction matchup
    parts = pred_matchup.split(' @ ')
    if len(parts) != 2:
        return None

    pred_away, pred_home = parts

    def _is_match(game_home, game_away):
        # Exact match
        if game_home == pred_home and game_away == pred_away:
            return True
        # Fuzzy match
        if (pred_home in game_home or game_home in pred_home) and \
           (pred_away in game_away or game_away in pred_away):
            return True
        return False

    # Collect all matching games (may be >1 for doubleheaders)
    matches = []
    for key, result in games.items():
        # Keys are (home, away) or (home, away, game_number)
        game_home = key[0]
        game_away = key[1]
        if _is_match(game_home, game_away):
            matches.append(result)

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    # Doubleheader: pick the game closest to pred_time
    if pred_time:
        try:
            pred_parts = pred_time.replace(":", "")[:4]
            pred_min = int(pred_parts[:2]) * 60 + int(pred_parts[2:4])
            def _time_dist(g):
                gt = g.get("game_time", "")
                if not gt:
                    return 9999
                gp = gt.replace(":", "")[:4]
                return abs(int(gp[:2]) * 60 + int(gp[2:4]) - pred_min)
            return min(matches, key=_time_dist)
        except (ValueError, IndexError):
            pass

    return matches[0]

def grade_pick(pick_str, spread, home_score, away_score, matchup):
    """
    Determine if a pick was correct.
    pick_str format: "Team Name +/-X.X"
    """
    parts = matchup.split(' @ ')
    if len(parts) != 2:
        return None
    
    away_team, home_team = parts
    
    # Determine which team was picked
    if home_team in pick_str:
        # Picked home team
        picked_home = True
        ats_margin = home_score + spread - away_score
    else:
        # Picked away team
        picked_home = False
        ats_margin = away_score - spread - home_score
    
    # Did the pick win?
    pick_won = ats_margin > 0
    
    return pick_won


def parse_game_pick(pick_str):
    """
    Parse GAME pick strings like:
      - "UConn ML YES"
      - "South Carolina ML NO"
      - "Duke ML"
    """
    match = re.match(r"^(.+?)\s+ML(?:\s+(YES|NO))?$", str(pick_str).strip(), re.IGNORECASE)
    if not match:
        return None
    team = match.group(1).strip()
    side = (match.group(2) or "YES").upper()
    return {"team": team, "side": side}


def grade_game_pick(game_pick, game_result):
    """Grade a GAME market winner pick (YES/NO on team ML)."""
    team = game_pick["team"]
    side = game_pick["side"]

    home_name = game_result["home_name"]
    away_name = game_result["away_name"]
    home_score = game_result["home_score"]
    away_score = game_result["away_score"]

    if home_score == away_score:
        return None

    team_lower = team.lower()
    home_lower = home_name.lower()
    away_lower = away_name.lower()

    if team_lower == home_lower:
        team_won = home_score > away_score
    elif team_lower == away_lower:
        team_won = away_score > home_score
    else:
        # fallback for minor naming differences
        if team_lower in home_lower or home_lower in team_lower:
            print(f"      INFO: Fuzzy matched GAME pick '{team}' to home '{home_name}'")
            team_won = home_score > away_score
        elif team_lower in away_lower or away_lower in team_lower:
            print(f"      INFO: Fuzzy matched GAME pick '{team}' to away '{away_name}'")
            team_won = away_score > home_score
        else:
            return None

    return team_won if side == "YES" else (not team_won)

def grade_predictions(league="mens"):
    """
    Main function to grade yesterday's predictions.
    """
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    pred_file = paths["predictions_file"]
    perf_file = paths["performance_file"]
    archive_prefix = paths["predictions_archive_prefix"]

    print("="*60)
    print(f"GRADING YESTERDAY'S PREDICTIONS ({league})")
    print("="*60)
    
    # 1. Determine yesterday's date (Eastern time)
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern)
    yesterday = today - timedelta(days=1)
    yesterday_date = yesterday.date()
    
    print(f"\nToday: {today.strftime('%Y-%m-%d %I:%M %p %Z')}")
    print(f"Grading date: {yesterday_date}\n")
    
    # 2. Check if we have predictions file
    # Try dated file first (e.g., predictions_20260109.csv / predictions_wbb_20260109.csv)
    dated_pred_file = os.path.join(
        BASE_DIR,
        f"{archive_prefix}_{yesterday_date.strftime('%Y%m%d')}.csv",
    )
    
    if os.path.exists(dated_pred_file):
        pred_source = dated_pred_file
        print(f"Found dated prediction file: {os.path.basename(dated_pred_file)}")
    elif os.path.exists(pred_file):
        pred_source = pred_file
        print(f"Using {os.path.basename(pred_file)} (no dated file found)")
        print(f"   Note: This may contain today's games instead of yesterday's")
    else:
        print("No predictions file found")
        print(f"   Looked for: {dated_pred_file}")
        print(f"   Looked for: {pred_file}")
        print("   Run predict.py to generate predictions first.")
        return
    
    # 3. Load predictions
    try:
        preds = pd.read_csv(pred_source)
        total_preds = len(preds)
        print(f"Loaded {total_preds} predictions from file")
        
        # Filter for actionable bets only (value-rated from either source)
        has_std = 'Std_Rating' in preds.columns
        has_kalshi = 'Rating' in preds.columns
        has_poly = 'Poly_Rating' in preds.columns

        if not has_std and not has_kalshi and not has_poly:
            print("ERROR: Prediction file has no rating columns.")
            print("   Cannot determine which bets are actionable.")
            print(f"   Columns found: {list(preds.columns)}")
            print("   Re-run predict.py to generate a file with rating columns.")
            return

        std_rating = preds['Std_Rating'] if has_std else pd.Series('PASS', index=preds.index)
        kalshi_rating = preds['Rating'].fillna('PASS') if has_kalshi else pd.Series('PASS', index=preds.index)
        poly_rating = preds['Poly_Rating'].fillna('PASS') if has_poly else pd.Series('PASS', index=preds.index)
        preds = preds[
            (std_rating.isin(VALUE_RATINGS)) |
            (kalshi_rating.isin(VALUE_RATINGS)) |
            (poly_rating.isin(VALUE_RATINGS))
        ].copy()

        print(f"   Filtered to {len(preds)} actionable bets (value-rated)")
        if len(preds) < total_preds:
            print(f"   Skipped {total_preds - len(preds)} non-value predictions")
            
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return
    
    # 4. Parse prediction dates
    # The Date/Time column might be "01/09 07:00 PM" format
    # We need to figure out which predictions are for yesterday
    
    # For now, let's assume ALL predictions in the file are recent
    # (since predict.py overwrites daily_predictions.csv each time)
    # We'll match them to yesterday's completed games
    
    print(f"\nPredictions to grade: {len(preds)} (actionable bets only)")
    
    # 5. Fetch yesterday's completed games
    completed_games = fetch_completed_games(yesterday, league=league)
    
    if not completed_games:
        print("\nNo completed games found for yesterday.")
        print("   Either no games were played, or ESPN API is not responding.")
        return
    
    # 6. Grade each prediction
    print(f"\nGrading predictions...")
    
    graded_bets = []
    unmatched = []
    
    for idx, pred in preds.iterrows():
        matchup = pred['Matchup']
        pick = pred['Pick']
        conf = pred['Conf']
        pred_spread = pd.to_numeric(pred.get('Spread', 0.0), errors='coerce')
        if pd.isna(pred_spread):
            pred_spread = 0.0
        bet_type = str(pred.get("Bet_Type", "spread") or "spread").strip().lower()
        
        # Extract game time from prediction for doubleheader disambiguation.
        # Both game_time (from ESPN) and pred_time are stored as Eastern.
        # CBB format "04/09 07:00 PM" is already Eastern (from tz_convert).
        # MLB format "2026-04-09 16:10" is UTC -- convert to Eastern.
        pred_time_str = ""
        raw_dt = pred.get("Date/Time", "")
        if raw_dt and isinstance(raw_dt, str) and ":" in raw_dt:
            try:
                if "-" in raw_dt:
                    # ISO-ish: "2026-04-09 16:10" (UTC from mlb/predict.py)
                    from datetime import datetime as _dt, timezone as _tz
                    import pytz as _pytz
                    utc_dt = _dt.strptime(raw_dt.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=_tz.utc)
                    eastern = _pytz.timezone("US/Eastern")
                    pred_time_str = utc_dt.astimezone(eastern).strftime("%H:%M")
                else:
                    # "04/09 07:00 PM" (already Eastern from CBB predict.py)
                    from datetime import datetime as _dt
                    dt = _dt.strptime(raw_dt.strip(), "%m/%d %I:%M %p")
                    pred_time_str = dt.strftime("%H:%M")
            except (ValueError, IndexError):
                pass

        # Find the corresponding game
        game_result = match_prediction_to_game(matchup, completed_games, pred_time=pred_time_str)
        
        if game_result is None:
            # Couldn't find this game - might be for today/tomorrow
            unmatched.append(matchup)
            continue
        
        if bet_type == "game":
            parsed_game_pick = parse_game_pick(pick)
            if parsed_game_pick is None:
                unmatched.append(matchup)
                continue
            pick_correct = grade_game_pick(parsed_game_pick, game_result)
            picked_team = (
                parsed_game_pick["team"]
                if parsed_game_pick["side"] == "YES"
                else f"NO {parsed_game_pick['team']}"
            )
            picked_spread = 0.0
        else:
            # Use ESPN closing spread when available; fall back to prediction-time spread
            # for manual/Kalshi-sourced lines where closing spread is missing.
            closing_spread = pd.to_numeric(game_result.get("spread", 0.0), errors="coerce")
            if pd.isna(closing_spread):
                closing_spread = 0.0
            spread = closing_spread if closing_spread != 0.0 else pred_spread
            pick_correct = grade_pick(
                pick,
                spread,
                game_result['home_score'],
                game_result['away_score'],
                matchup
            )
            # Extract picked team and spread
            pick_parts = pick.split()
            picked_team = " ".join(pick_parts[:-1])
            picked_spread = float(pick_parts[-1])

        if pick_correct is None:
            unmatched.append(matchup)
            continue

        graded_bets.append({
            'date': yesterday_date,
            'bet_type': bet_type,
            'picked_team': picked_team,
            'picked_spread': picked_spread,
            'conf': conf,
            'pick_correct': pick_correct,
            'matchup': matchup,
            'final_score': f"{game_result['away_score']}-{game_result['home_score']}"
        })
        
        result_icon = "W" if pick_correct else "L"
        print(f"   {result_icon} {matchup}: {pick} ({'WIN' if pick_correct else 'LOSS'})")
    
    print(f"\nGrading Summary:")
    print(f"   Graded: {len(graded_bets)}")
    print(f"   Unmatched: {len(unmatched)}")
    
    if unmatched:
        print(f"\n   Unmatched predictions (likely for today/tomorrow):")
        for m in unmatched[:5]:
            print(f"      - {m}")
    
    # 7. Save to performance log
    if graded_bets:
        graded_df = pd.DataFrame(graded_bets)
        
        # Append to existing performance log or create new one
        if os.path.exists(perf_file):
            existing = pd.read_csv(perf_file)
            existing['date'] = pd.to_datetime(existing['date']).dt.date
            
            # Remove any existing entries for yesterday (in case re-grading)
            existing = existing[existing['date'] != yesterday_date]
            
            # Combine
            combined = pd.concat([existing, graded_df], ignore_index=True)
            combined.to_csv(perf_file, index=False)
            print(f"\nAdded {len(graded_bets)} graded bets to {os.path.basename(perf_file)}")
        else:
            graded_df.to_csv(perf_file, index=False)
            print(f"\nCreated {os.path.basename(perf_file)} with {len(graded_bets)} bets")
        
        # Show win rate
        wins = sum(graded_df['pick_correct'])
        win_rate = wins / len(graded_df)
        profit = sum(1.0 if x else -1.1 for x in graded_df['pick_correct'])
        
        print(f"\nYesterday's Performance:")
        print(f"   Record: {wins}-{len(graded_df)-wins}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Profit: {profit:+.2f} units")
        print(f"   (Only includes value-rated bets)")
    else:
        print("\nNo predictions were graded.")
        print("   This likely means the predictions file contains games for today/tomorrow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grade yesterday's value-rated spread and GAME predictions."
    )
    parser.add_argument(
        "--league",
        default="mens",
        help="League to grade: mens or womens (aliases supported).",
    )
    args = parser.parse_args()
    grade_predictions(args.league)
