import argparse
import pandas as pd
import numpy as np
import requests
import joblib
import json
import os
from datetime import datetime, timedelta
from difflib import get_close_matches
import pytz

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from kalshi import KalshiClient, MarketMapper
from betting import calculate_edge, get_rating, recommended_units, EdgeRating, STANDARD_IMPLIED_PROB
from betting import calculate_line_shopping

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "cbb_model_v1.pkl")
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "daily_predictions.csv")
BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=1000"

# Module-level storage for predictions with line shopping data (for app.py access)
_latest_predictions = None

# Games that need manual spread entry (no ESPN spread available)
_games_needing_spreads = None

# --- TEAM MAP (loaded from config file) ---
TEAM_MAP_FILE = os.path.join(BASE_DIR, "team_map.json")
with open(TEAM_MAP_FILE, 'r') as f:
    TEAM_MAP = json.load(f)

def find_best_match(name, known_teams):
    """Match ESPN team name to historical data team name."""
    if name in TEAM_MAP: 
        return TEAM_MAP[name]
    
    parts = name.split()
    if len(parts) > 1:
        no_mascot = " ".join(parts[:-1])
        if no_mascot in TEAM_MAP: 
            return TEAM_MAP[no_mascot]
    
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    
    # Log warning for unmatched teams
    print(f"      WARNING: Could not match '{name}' to historical data")
    return None

def get_latest_stats(df):
    """Get the most recent stats for each team."""
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    latest_stats = {}
    teams = df['team'].unique()

    for team in teams:
        last_game = df[df['team'] == team].iloc[-1]
        stats = {}
        for col in df.columns:
            if any(x in col for x in ['season_', 'roll', 'prev_', 'opp_win_pct']):
                stats[col] = last_game[col]
        stats['last_game_date'] = last_game['date']
        stats['last_opponent'] = last_game.get('opponent', 'Unknown')
        # Features that may not exist in older rows -- use sensible defaults
        stats['prev_games_played'] = last_game.get('prev_games_played', 10)
        stats['prev_volatility'] = last_game.get('prev_volatility', 10)
        stats['prev_roll5_margin'] = last_game.get('prev_roll5_margin', 0)
        stats['prev_blowout_rate'] = last_game.get('prev_blowout_rate', 0)
        stats['prev_win_pct'] = last_game.get('prev_win_pct', 0.5)
        latest_stats[team] = stats

    return latest_stats

def fetch_schedule():
    """
    Fetch today's and tomorrow's games with TIMEZONE AWARENESS.
    Uses Eastern Time to ensure we're querying the correct date.
    """
    print("   -> Fetching schedule (TIMEZONE AWARE)...")
    
    # Use Eastern Time for proper date handling
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    
    print(f"      Current Eastern Time: {now_eastern.strftime('%Y-%m-%d %I:%M %p %Z')}")
    
    games = []
    
    # Fetch today through 5 days ahead (Eastern time)
    for days_ahead in range(6):
        target_date = now_eastern + timedelta(days=days_ahead)
        date_str = target_date.strftime("%Y%m%d")
        url = f"{BASE_URL}&dates={date_str}"
        
        print(f"      Querying ESPN for: {date_str} ({target_date.strftime('%A, %B %d')})")
        
        try:
            res = requests.get(url, timeout=10)
            data = res.json()
            
            events_count = len(data.get('events', []))
            print(f"         Found {events_count} events")
            
            for event in data['events']:
                game_date = pd.to_datetime(event['date'])
                
                if not event.get('competitions'): 
                    continue
                comp = event['competitions'][0]
                
                if not comp.get('competitors'): 
                    continue
                    
                home_tm = comp['competitors'][0]['team']
                away_tm = comp['competitors'][1]['team']
                home_raw = home_tm['displayName']
                away_raw = away_tm['displayName']
                
                # Get odds
                odds = comp.get('odds', [{}])[0] if comp.get('odds') else {}
                details = odds.get('details', '0')
                raw_odds = details 
                
                spread_val = 0.0
                try:
                    if details and details != '0' and details != 'EVEN':
                        parts = details.split()
                        val = abs(float(parts[-1]))
                        fav = " ".join(parts[:-1])
                        
                        home_abbr = home_tm.get('abbreviation', '')
                        is_home_fav = (fav == home_abbr) or (fav == home_raw) or (fav in home_raw)
                        
                        if is_home_fav:
                            spread_val = -val
                        else:
                            spread_val = val
                except (ValueError, IndexError):
                    spread_val = 0.0

                game_id = event['id']
                if not any(g['id'] == game_id for g in games):
                    games.append({
                        'id': game_id,
                        'home_raw': home_raw,  # Keep original ESPN name
                        'away_raw': away_raw,  # Keep original ESPN name
                        'spread': spread_val,  # May be 0 if ESPN doesn't have it
                        'date': game_date,
                        'raw_odds': raw_odds,
                        'has_espn_spread': spread_val != 0.0
                    })
                    
        except Exception as e:
            print(f"         Error fetching {date_str}: {e}")
            
    return sorted(games, key=lambda x: x['date'])

def fetch_kalshi_markets():
    """Fetch Kalshi NCAAB markets and build mapper."""
    print("   -> Fetching Kalshi markets...")

    api_key = os.getenv("KALSHI_API_KEY")
    if not api_key:
        print("      KALSHI_API_KEY not set. Skipping Kalshi integration.")
        return None, None

    try:
        client = KalshiClient(api_key)
        markets = client.get_ncaab_markets()

        if markets:
            print(f"      Found {len(markets)} NCAAB markets")
            mapper = MarketMapper(markets)
            return client, mapper
        else:
            print("      No NCAAB markets found")
            return client, None
    except Exception as e:
        print(f"      Kalshi API error: {e}")
        return None, None


def get_kalshi_spread(mapper, home_team, away_team, game_date):
    """
    Get spread from Kalshi markets when ESPN doesn't have one.

    Args:
        mapper: MarketMapper instance for finding Kalshi markets
        home_team: Home team name from ESPN
        away_team: Away team name from ESPN
        game_date: Game datetime for market matching

    Returns:
        Tuple of (spread_value, favorite_is_home) or (None, None) if not found
    """
    if not mapper:
        return None, None

    try:
        from kalshi.market_mapper import extract_school_keyword

        all_markets = mapper.find_all_markets_for_game(home_team, away_team, game_date)
        spread_markets = [m for m in all_markets if "SPREAD" in m.get("ticker", "")]

        if not spread_markets:
            print(f"      No Kalshi SPREAD markets found for {away_team} @ {home_team}")
            return None, None

        # Get the first spread market and extract info
        market = spread_markets[0]
        floor_strike = market.get("floor_strike", 0)
        title = market.get("title", "").lower()

        # Determine which team is favored based on the market title
        home_keyword = extract_school_keyword(home_team).lower()
        away_keyword = extract_school_keyword(away_team).lower()

        # If home team is in title as "wins by", they're favored (negative spread)
        if home_keyword in title:
            return -floor_strike, True
        elif away_keyword in title:
            return floor_strike, False

        print(f"      Could not determine favorite from Kalshi market title: {title}")
        return None, None
    except ImportError as e:
        print(f"      Kalshi module not available: {e}")
        return None, None
    except (KeyError, AttributeError) as e:
        print(f"      Kalshi market data format issue for {away_team} @ {home_team}: {e}")
        return None, None
    except Exception as e:
        print(f"      Unexpected error getting Kalshi spread for {away_team} @ {home_team}: {type(e).__name__}: {e}")
        return None, None


def get_kalshi_edge(client, mapper, home_team, away_team, game_date, spread, model_prob, picked_team, picked_spread):
    """
    Get Kalshi market data and calculate edge.

    Args:
        picked_spread: The spread value for our pick (positive = underdog, negative = favorite)

    Returns dict with Kalshi_Yes, Kalshi_No, Edge, Rating, Units, Kalshi_Side
    """
    from kalshi.market_mapper import extract_school_keyword

    result = {
        "Kalshi_Yes": None,
        "Kalshi_No": None,
        "Edge": None,
        "Rating": None,
        "Units": None,
        "Kalshi_Ticker": None,
        "Kalshi_Side": None,
        "Kalshi_Title": None,
    }

    if not mapper:
        return result

    try:
        # Find all spread markets for this game
        all_markets = mapper.find_all_markets_for_game(home_team, away_team, game_date)
        spread_markets = [m for m in all_markets if "SPREAD" in m.get("ticker", "")]

        if not spread_markets:
            return result

        # Determine if we're betting favorite or underdog
        is_underdog = picked_spread > 0  # Positive spread = underdog

        # Find the opponent team
        picked_keyword = extract_school_keyword(picked_team)
        if picked_keyword in extract_school_keyword(home_team):
            opponent = away_team
        else:
            opponent = home_team
        opponent_keyword = extract_school_keyword(opponent)

        # Find the right market:
        # - If we pick underdog (+spread), bet NO on "opponent wins by over X"
        # - If we pick favorite (-spread), bet YES on "picked_team wins by over X"
        best_market = None
        best_spread_diff = float('inf')

        for market in spread_markets:
            title = market.get("title", "").lower()
            floor_strike = market.get("floor_strike", 0)

            if is_underdog:
                # We want opponent's spread market to bet NO
                if opponent_keyword in title and "wins by" in title:
                    spread_diff = abs(floor_strike - abs(picked_spread))
                    if spread_diff < best_spread_diff:
                        best_spread_diff = spread_diff
                        best_market = market
            else:
                # We want our team's spread market to bet YES
                if picked_keyword in title and "wins by" in title:
                    spread_diff = abs(floor_strike - abs(picked_spread))
                    if spread_diff < best_spread_diff:
                        best_spread_diff = spread_diff
                        best_market = market

        if best_market and best_spread_diff > 0.01:
            # Kalshi spread doesn't match our pick -- skip
            floor_strike = best_market.get("floor_strike", 0)
            print(f"      Kalshi spread mismatch: Kalshi has {floor_strike}, "
                  f"pick spread is {abs(picked_spread)} (diff={best_spread_diff:.1f}) -- skipping")
            return result

        if best_market:
            # Fetch fresh prices from API (not cached data)
            ticker = best_market.get("ticker", "")
            prices = client.get_market_prices(ticker) if ticker else mapper.get_market_prices(best_market)
            title = prices.get("title", "")
            yes_price = prices.get("yes_price", 50)
            no_price = prices.get("no_price", 50)

            if is_underdog:
                kalshi_side = "NO"
                implied_prob = no_price / 100.0
                bet_price = no_price
            else:
                kalshi_side = "YES"
                implied_prob = yes_price / 100.0
                bet_price = yes_price

            # Calculate edge using the correct implied probability
            edge = calculate_edge(model_prob, implied_prob)
            rating = get_rating(edge)
            units = recommended_units(edge, implied_prob)

            result = {
                "Kalshi_Yes": yes_price,
                "Kalshi_No": no_price,
                "Kalshi_Price": bet_price,
                "Edge": edge,
                "Edge_Pct": f"{edge * 100:+.1f}%",
                "Rating": rating.value,
                "Units": units,
                "Kalshi_Ticker": prices.get("ticker", ""),
                "Kalshi_Side": kalshi_side,
                "Kalshi_Title": title,
            }
    except ImportError as e:
        print(f"      Kalshi module not available: {e}")
    except (KeyError, AttributeError) as e:
        print(f"      Kalshi market data format issue: {e}")
    except ZeroDivisionError as e:
        print(f"      Kalshi calculation error (division by zero): {e}")
    except Exception as e:
        print(f"      Unexpected Kalshi error for {away_team} @ {home_team}: {type(e).__name__}: {e}")

    return result


def calculate_production_features(row, h_stats, a_stats):
    """Calculate features needed for prediction."""
    # --- Original Features ---
    # 1. Effective Field Goal %
    row['diff_eFG'] = h_stats.get('season_team_eFG', 0) - a_stats.get('season_team_eFG', 0)

    # 2. Rebounds
    h_orb = h_stats.get('season_team_orb', 0)
    a_orb = a_stats.get('season_team_orb', 0)
    row['diff_Rebound'] = h_orb - a_orb

    # 3. Turnovers
    h_to = h_stats.get('season_team_to', 0)
    a_to = a_stats.get('season_team_to', 0)
    row['diff_TO'] = h_to - a_to

    # 4. Momentum
    row['momentum_gap'] = h_stats.get('roll3_team_eFG', 0) - h_stats.get('season_team_eFG', 0)

    # 5. Cover Margin
    row['roll5_cover_margin'] = h_stats.get('roll5_cover_margin', 0)

    # --- V2 Features ---
    # 6. Games Played (home team's sample size)
    row['prev_games_played'] = h_stats.get('prev_games_played', 10)

    # 7. Opponent Win % (away team's quality)
    row['opp_win_pct'] = a_stats.get('prev_win_pct', 0.5)

    # 8. Blowout Rate (home team's dominance)
    row['prev_blowout_rate'] = h_stats.get('prev_blowout_rate', 0)

    # 9. Recent Margin (home team's recent performance)
    row['prev_roll5_margin'] = h_stats.get('prev_roll5_margin', 0)

    # 10. Volatility (home team's consistency)
    row['prev_volatility'] = h_stats.get('prev_volatility', 10)

    # 11-12. Spread interaction features
    row['spread_abs'] = abs(row.get('spread', 0))
    row['spread_squared'] = row.get('spread', 0) ** 2

    return row

def fetch_games_needing_spreads():
    """Fetch schedule and return games that have no ESPN spread.

    Returns list of dicts with 'away_raw', 'home_raw', 'date', 'id' for each
    game missing a spread, or empty list if all games have spreads.
    """
    schedule = fetch_schedule()
    missing = []
    for g in schedule:
        if not g.get('has_espn_spread', False):
            try:
                eastern = pytz.timezone('US/Eastern')
                local_ts = g['date'].tz_convert(eastern)
                time_str = local_ts.strftime("%m/%d %I:%M %p")
            except (TypeError, AttributeError):
                time_str = g['date'].strftime("%m/%d %I:%M %p")
            missing.append({
                'id': g['id'],
                'away_raw': g['away_raw'],
                'home_raw': g['home_raw'],
                'date': g['date'],
                'time_str': time_str,
                'matchup': f"{g['away_raw']} @ {g['home_raw']}",
            })
    return missing


def get_games_needing_spreads():
    """Return cached list of games needing manual spreads."""
    return _games_needing_spreads


def main(spread_overrides=None):
    """Run prediction engine.

    Args:
        spread_overrides: dict mapping game matchup string to home-team spread float.
            e.g. {"Northwestern Wildcats @ Iowa Hawkeyes": -7.5}
            Convention: negative = home favored, positive = away favored.
    """
    if spread_overrides is None:
        spread_overrides = {}

    print("--- PREDICTION ENGINE (GBM + Sigmoid Calibration, 15 features) ---")

    # Get current Eastern time for dated file naming
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)

    # Load model and data
    try:
        model = joblib.load(MODEL_FILE)
        print(f"   Model loaded: {MODEL_FILE}")
    except (FileNotFoundError, IOError, EOFError) as e:
        print(f"CRITICAL: Model not found or corrupted. Run model.py first. ({e})")
        return

    try:
        df_hist = pd.read_csv(DATA_FILE)
        print(f"   Data loaded: {len(df_hist)} historical games")
    except (FileNotFoundError, IOError, pd.errors.EmptyDataError) as e:
        print(f"CRITICAL: Training data not found or corrupted. Run main.py to download data. ({e})")
        return

    known_teams = df_hist['team'].unique()
    team_stats = get_latest_stats(df_hist)

    # Check data freshness
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    last_data_date = df_hist['date'].max()
    print(f"   Data current through: {last_data_date.strftime('%Y-%m-%d')}")

    days_old = (datetime.now() - last_data_date).days
    if days_old > 2:
        print(f"   WARNING: Data is {days_old} days old. Run main.py to update!")

    # Fetch schedule
    schedule = fetch_schedule()
    games_with_espn_spread = sum(1 for g in schedule if g.get('has_espn_spread', False))
    print(f"   -> Found {len(schedule)} games ({games_with_espn_spread} with ESPN spreads)")

    # Fetch Kalshi markets
    kalshi_client, kalshi_mapper = fetch_kalshi_markets()

    predictions = []
    skipped = []
    games_needing_spreads = []

    for g in schedule:
        # Match team names to historical data
        home_matched = find_best_match(g['home_raw'], known_teams)
        away_matched = find_best_match(g['away_raw'], known_teams)

        # Skip if we can't match teams or don't have stats
        if not home_matched or not away_matched:
            skipped.append(f"{g['away_raw']} @ {g['home_raw']} (Team matching failed)")
            continue

        if home_matched not in team_stats or away_matched not in team_stats:
            skipped.append(f"{g['away_raw']} @ {g['home_raw']} (No historical stats)")
            continue

        # If no ESPN spread, check for manual override
        matchup_key = f"{g['away_raw']} @ {g['home_raw']}"
        if not g.get('has_espn_spread', False):
            if matchup_key in spread_overrides:
                g['spread'] = spread_overrides[matchup_key]
                g['raw_odds'] = f"Manual {g['spread']}"
                g['has_espn_spread'] = True  # Treat as valid
            else:
                games_needing_spreads.append({
                    'id': g['id'],
                    'away_raw': g['away_raw'],
                    'home_raw': g['home_raw'],
                    'matchup': matchup_key,
                })
                skipped.append(f"{matchup_key} (No spread -- needs manual entry)")
                continue

        # Build feature row
        row = {'is_home': 1, 'spread': g['spread']}
        h_stats = team_stats[home_matched]
        a_stats = team_stats[away_matched]
        
        # Calculate rest days for BOTH teams
        home_last_date = pd.to_datetime(h_stats.get('last_game_date', datetime.now()))
        away_last_date = pd.to_datetime(a_stats.get('last_game_date', datetime.now()))
        
        home_actual_rest = max(0, (g['date'].replace(tzinfo=None) - home_last_date).days)
        away_actual_rest = max(0, (g['date'].replace(tzinfo=None) - away_last_date).days)
        
        # For model: use home team's rest, capped at 7 to match training data
        row['rest_days'] = min(home_actual_rest, 7)
        
        # Add production features
        row = calculate_production_features(row, h_stats, a_stats)
        
        # Prepare for model
        cols = model.feature_names_in_
        input_df = pd.DataFrame([row])
        for c in cols:
            if c not in input_df.columns:
                input_df[c] = 0.0

        input_df.columns = input_df.columns.astype(str)

        # Fill any NaN values with defaults
        input_df = input_df.fillna({
            'diff_eFG': 0, 'diff_Rebound': 0, 'diff_TO': 0,
            'momentum_gap': 0, 'roll5_cover_margin': 0,
            'prev_games_played': 10, 'opp_win_pct': 0.5,
            'prev_blowout_rate': 0, 'prev_roll5_margin': 0,
            'prev_volatility': 10, 'is_home': 1, 'spread': 0, 'rest_days': 3,
            'spread_abs': 0, 'spread_squared': 0,
        })
        input_df = input_df.fillna(0)  # Catch any remaining NaNs
        
        # Make prediction
        prob = model.predict_proba(input_df)[0][1]
        conf = max(prob, 1-prob)
        
        # Determine pick - USE ORIGINAL ESPN NAMES
        if prob > 0.5:
            sign = "+" if g['spread'] > 0 else ""
            pick_str = f"{g['home_raw']} {sign}{g['spread']}"  # Original name
            picked_team = g['home_raw']
            picked_spread = g['spread']  # Home team's spread
            picked_team_rest = home_actual_rest  # Picked home team
        else:
            away_spread = -1 * g['spread']
            sign = "+" if away_spread > 0 else ""
            pick_str = f"{g['away_raw']} {sign}{away_spread}"  # Original name
            picked_team = g['away_raw']
            picked_spread = away_spread  # Away team's spread
            picked_team_rest = away_actual_rest  # Picked away team

        # Format time in Eastern
        try:
            local_ts = g['date'].tz_convert('US/Eastern')
            time_str = local_ts.strftime("%m/%d %I:%M %p")
        except (TypeError, AttributeError):
            time_str = g['date'].strftime("%m/%d %I:%M %p")

        # Get Kalshi edge data (now that we know the picked team)
        kalshi_data = get_kalshi_edge(
            kalshi_client,
            kalshi_mapper,
            g['home_raw'],
            g['away_raw'],
            g['date'],
            g['spread'],
            conf,  # Use confidence as model probability
            picked_team,  # Pass picked team to determine YES/NO side
            picked_spread,  # Pass the spread for our pick
        )

        # Calculate line shopping recommendations
        # Build base features (everything except spread)
        base_features = {k: v for k, v in row.items() if k != 'spread'}
        is_home_pick = (prob > 0.5)  # Home pick if model probability > 0.5

        line_shopping = calculate_line_shopping(
            model,
            base_features,
            picked_spread,
            picked_team,
            is_home_pick,
        )

        # CRITICAL FIX: Use ORIGINAL ESPN names for display
        prediction_row = {
            "Date/Time": time_str,
            "Matchup": f"{g['away_raw']} @ {g['home_raw']}",
            "Spread": g['spread'],
            "Pick": pick_str,
            "Conf": conf,
            "Raw Odds": g['raw_odds'],
            "Rest": picked_team_rest,
            # Kalshi edge data
            "Kalshi_Side": kalshi_data.get("Kalshi_Side"),
            "Kalshi_Price": kalshi_data.get("Kalshi_Price"),
            "Kalshi_Title": kalshi_data.get("Kalshi_Title"),
            "Edge": kalshi_data.get("Edge"),
            "Edge_Pct": kalshi_data.get("Edge_Pct"),
            "Rating": kalshi_data.get("Rating"),
            "Units": kalshi_data.get("Units"),
            # Debug fields
            "Home_Matched": home_matched,
            "Away_Matched": away_matched,
            "Kalshi_Ticker": kalshi_data.get("Kalshi_Ticker"),
            # Line shopping data
            "Breakeven_Spread": line_shopping.breakeven_spread,
            "Line_Shopping_Data": line_shopping,
            # Standard book edge (vs -110 odds, for non-Kalshi sportsbooks)
            "Std_Edge": conf - STANDARD_IMPLIED_PROB,
            "Std_Edge_Pct": f"{(conf - STANDARD_IMPLIED_PROB) * 100:+.1f}%",
            "Std_Rating": get_rating(conf - STANDARD_IMPLIED_PROB).value,
            "Std_Units": recommended_units(conf - STANDARD_IMPLIED_PROB, STANDARD_IMPLIED_PROB),
        }

        # VALIDATION: Ensure pick mentions a team that's actually in the matchup
        pick_team_mentioned = pick_str.split()[0] + " " + pick_str.split()[1]
        if g['home_raw'] not in pick_str and g['away_raw'] not in pick_str:
            print(f"      WARNING: Pick '{pick_str}' doesn't match matchup '{prediction_row['Matchup']}'")

        predictions.append(prediction_row)

    # Save predictions
    if predictions:
        pred_df = pd.DataFrame(predictions).sort_values(by="Conf", ascending=False)

        # Drop complex objects before CSV save (can't serialize)
        csv_df = pred_df.drop(columns=["Line_Shopping_Data"], errors="ignore")

        # Save to current file (for app)
        csv_df.to_csv(OUTPUT_FILE, index=False)
        
        # ALSO save to dated archive file (for grading)
        archive_file = OUTPUT_FILE.replace("daily_predictions.csv",
                                          f"predictions_{now_eastern.strftime('%Y%m%d')}.csv")
        csv_df.to_csv(archive_file, index=False)
        
        # Store predictions with line shopping data for app.py access
        global _latest_predictions
        _latest_predictions = pred_df

        print(f"\nSUCCESS: Generated {len(pred_df)} predictions")
        print(f"   Saved to: {OUTPUT_FILE}")
        print(f"   Archive: {archive_file}")

        # Show summary
        print("\nPREDICTION SUMMARY:")
        for _, row in pred_df.head(5).iterrows():
            print(f"   {row['Matchup']}")
            print(f"      Pick: {row['Pick']} (Conf: {row['Conf']:.1%})")

        # Show value bets (STRONG edge from either source)
        value_bets = pred_df[
            (pred_df['Std_Rating'] == 'STRONG') |
            (pred_df['Rating'] == 'STRONG')
        ]
        if len(value_bets) > 0:
            print(f"\nVALUE BETS ({len(value_bets)} found):")
            for _, row in value_bets.iterrows():
                std_rating = row.get('Std_Rating', 'PASS')
                kalshi_rating = row.get('Rating', None) if pd.notna(row.get('Rating')) else None
                print(f"   [STRONG] {row['Pick']}")
                if kalshi_rating == 'STRONG':
                    side = row['Kalshi_Side'] if row['Kalshi_Side'] else "?"
                    print(f"      Kalshi: Buy {side} @ {row['Kalshi_Price']}c | Edge: {row['Edge_Pct']} | {row['Units']:.1f}U")
                if std_rating == 'STRONG':
                    print(f"      Std Book: Edge {row['Std_Edge_Pct']} | {row['Std_Units']:.1f}U")
    else:
        print("\nNo predictions generated.")
    
    # Show skipped games
    if skipped:
        print(f"\nSkipped {len(skipped)} games:")
        for s in skipped[:5]:
            print(f"   - {s}")

    # Store games that still need manual spreads
    global _games_needing_spreads
    _games_needing_spreads = games_needing_spreads
    if games_needing_spreads:
        print(f"\n{len(games_needing_spreads)} game(s) need manual spread entry.")

def get_latest_predictions():
    """Return the latest predictions DataFrame with line shopping data."""
    return _latest_predictions


def check_live_prices():
    """Re-fetch current Kalshi prices for today's STRONG picks and recalculate edge.

    Reads the daily predictions CSV, filters to rows with a Kalshi ticker,
    fetches live prices from the Kalshi API, and prints an updated table
    showing current edge so the user can verify value before placing a bet.
    """
    print("--- LIVE PRICE CHECK ---\n")

    # Load predictions CSV
    if not os.path.exists(OUTPUT_FILE):
        print(f"No predictions file found at {OUTPUT_FILE}")
        print("Run predict.py first (without --check) to generate predictions.")
        return

    df = pd.read_csv(OUTPUT_FILE)

    # Filter to rows that have a Kalshi ticker
    if "Kalshi_Ticker" not in df.columns:
        print("Predictions CSV does not have a Kalshi_Ticker column.")
        return

    has_ticker = df["Kalshi_Ticker"].notna() & (df["Kalshi_Ticker"] != "")
    df_kalshi = df[has_ticker].copy()

    if df_kalshi.empty:
        print("No predictions with Kalshi tickers found.")
        return

    # Filter to STRONG picks only (from either standard or Kalshi rating)
    is_strong = (
        (df_kalshi.get("Std_Rating") == "STRONG") |
        (df_kalshi.get("Rating") == "STRONG")
    )
    df_strong = df_kalshi[is_strong].copy()

    if df_strong.empty:
        print("No STRONG-rated picks with Kalshi tickers found.")
        return

    # Initialize Kalshi client
    api_key = os.getenv("KALSHI_API_KEY")
    if not api_key:
        print("KALSHI_API_KEY not set. Cannot fetch live prices.")
        return

    client = KalshiClient(api_key)

    # Fetch live prices and recalculate edge for each pick
    results = []
    for _, row in df_strong.iterrows():
        ticker = row["Kalshi_Ticker"]
        side = row.get("Kalshi_Side", "YES")
        model_prob = row["Conf"]
        pick = row["Pick"]

        try:
            prices = client.get_market_prices(ticker)
        except Exception as e:
            print(f"   Error fetching {ticker}: {e}")
            results.append({
                "Pick": pick,
                "Model%": f"{model_prob:.1%}",
                "Live Price": "ERR",
                "Edge": "--",
                "Rating": "--",
                "Units": "--",
            })
            continue

        yes_price = prices.get("yes_price")
        no_price = prices.get("no_price")

        if yes_price is None and no_price is None:
            results.append({
                "Pick": pick,
                "Model%": f"{model_prob:.1%}",
                "Live Price": "N/A",
                "Edge": "--",
                "Rating": "--",
                "Units": "--",
            })
            continue

        # Use the correct price based on which side we're betting
        if side == "YES":
            live_price = yes_price
        else:
            live_price = no_price

        implied_prob = live_price / 100.0
        edge = calculate_edge(model_prob, implied_prob)
        rating = get_rating(edge)
        units = recommended_units(edge, implied_prob)

        results.append({
            "Pick": pick,
            "Model%": f"{model_prob:.1%}",
            "Live Price": f"{side} @ {live_price}c",
            "Edge": f"{edge * 100:+.1f}%",
            "Rating": rating.value,
            "Units": f"{units:.1f}U" if units > 0 else "PASS",
        })

    # Print results table
    if not results:
        print("No results to display.")
        return

    # Calculate column widths
    headers = ["Pick", "Model%", "Live Price", "Edge", "Rating", "Units"]
    widths = {}
    for h in headers:
        col_values = [str(r[h]) for r in results]
        widths[h] = max(len(h), max(len(v) for v in col_values))

    # Print header
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for r in results:
        line = "  ".join(str(r[h]).ljust(widths[h]) for h in headers)
        print(line)

    # Summary
    still_strong = sum(1 for r in results if r["Rating"] == "STRONG")
    total = len(results)
    print(f"\n{still_strong}/{total} picks still STRONG at live prices.")

    if still_strong < total:
        print("Some picks have lost edge -- consider skipping those bets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CBB prediction engine -- generate picks or check live prices."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-fetch live Kalshi prices for today's STRONG picks and show updated edge.",
    )
    args = parser.parse_args()

    if args.check:
        check_live_prices()
    else:
        main()