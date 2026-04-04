import argparse
import json
import os
from datetime import datetime, timedelta
from difflib import get_close_matches
import re
import threading

import joblib
import numpy as np
import pandas as pd
import pytz
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed; skipping .env auto-load.")

from kalshi import KalshiClient, MarketMapper
from betting import calculate_edge, get_rating, recommended_units, EdgeRating, STANDARD_IMPLIED_PROB, VALUE_RATINGS, RATING_RANK, kalshi_implied_prob, american_odds_to_implied_prob
from betting.ev_calculator import kalshi_fee_cents
from betting import calculate_line_shopping
from betting.line_shopping import LineShoppingResult
from hasla import HASLA_GAME_FEATURE_COLUMNS, load_snapshot_file as load_hasla_snapshot_file, load_team_map as load_hasla_team_map, matchup_features_for_game as hasla_matchup_features_for_game
from kalshi_game_archive import append_archive_records as append_kalshi_game_archive_records
from kalshi_game_archive import build_game_archive_record
from league_config import get_league_artifact_paths, get_scoreboard_base_url, normalize_league
from model import load_model, use_calibrated_spread_model
from model_win import load_win_model_bundle, predict_home_win_prob
from odds_archive import append_archive_records, build_archive_record
from torvik import TORVIK_GAME_FEATURE_COLUMNS, load_snapshot_file, load_team_map, matchup_features_for_game
from venue import (
    build_team_home_locations,
    compute_distance_advantage,
    load_geocode_cache,
    save_geocode_cache,
)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEAGUE = "mens"
MODEL_FILE = None
WIN_MODEL_FILE = None
DATA_FILE = None
OUTPUT_FILE = None
BASE_URL = None
PREDICTIONS_ARCHIVE_PREFIX = None
TORVIK_SNAPSHOT_FILE = None
TORVIK_MAP_FILE = None
HASLA_SNAPSHOT_FILE = None
HASLA_MAP_FILE = None

# Module-level storage for predictions with line shopping data (for app.py access)
_latest_predictions = {}
_latest_game_predictions = {}

# Games that need manual spread entry (no ESPN spread available)
_games_needing_spreads = {}
_RUNTIME_LOCK = threading.Lock()

GAME_STRONG_MIN_PROB = 0.55
GAME_STRONG_MIN_PRICE = 10
GAME_STRONG_MAX_PRICE = 90
GAME_GOOD_MIN_PROB = 0.52
GAME_GOOD_MIN_PRICE = 15
GAME_GOOD_MAX_PRICE = 85

# --- TEAM MAP (loaded from config file) ---
TEAM_MAP_FILE = os.path.join(BASE_DIR, "team_map.json")
with open(TEAM_MAP_FILE, 'r') as f:
    TEAM_MAP = json.load(f)


def configure_league(league="mens"):
    """Set module-level paths/urls for the requested league."""
    global ACTIVE_LEAGUE, MODEL_FILE, WIN_MODEL_FILE, DATA_FILE, OUTPUT_FILE, BASE_URL, PREDICTIONS_ARCHIVE_PREFIX, TORVIK_SNAPSHOT_FILE, TORVIK_MAP_FILE, HASLA_SNAPSHOT_FILE, HASLA_MAP_FILE
    ACTIVE_LEAGUE = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, ACTIVE_LEAGUE)
    MODEL_FILE = paths["model_file"]
    WIN_MODEL_FILE = paths["win_model_file"]
    DATA_FILE = paths["data_file"]
    OUTPUT_FILE = paths["predictions_file"]
    BASE_URL = get_scoreboard_base_url(ACTIVE_LEAGUE)
    PREDICTIONS_ARCHIVE_PREFIX = paths["predictions_archive_prefix"]
    TORVIK_SNAPSHOT_FILE = paths.get("torvik_snapshot_file")
    TORVIK_MAP_FILE = paths.get("torvik_map_file")
    HASLA_SNAPSHOT_FILE = paths.get("hasla_snapshot_file")
    HASLA_MAP_FILE = paths.get("hasla_map_file")
    return ACTIVE_LEAGUE


# Default to men's config unless caller overrides via --league / function argument.
configure_league("mens")

def find_best_match(name, known_teams):
    """Match ESPN team name to historical data team name."""
    # Exact match in training data takes priority over TEAM_MAP, which may
    # map to a different variant (e.g. "UConn Huskies" -> "Connecticut" is
    # correct for men's but wrong for WBB where the team is "UConn Huskies").
    if name in known_teams:
        return name

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

def _get_std_implied_prob(game, is_home_pick, bet_type="spread"):
    """Get the implied probability for the picked side from real ESPN odds.

    Falls back to STANDARD_IMPLIED_PROB if odds aren't available.
    """
    if bet_type == "spread":
        odds_str = game.get("home_spread_odds") if is_home_pick else game.get("away_spread_odds")
    else:
        odds_str = game.get("home_ml_odds") if is_home_pick else game.get("away_ml_odds")

    if odds_str:
        implied = american_odds_to_implied_prob(odds_str)
        if implied is not None:
            return implied
    return STANDARD_IMPLIED_PROB


def _compute_std_edge(conf, game, is_home_pick, bet_type="spread"):
    """Compute edge vs real sportsbook odds (not the fixed -110 assumption)."""
    return conf - _get_std_implied_prob(game, is_home_pick, bet_type)


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
    failed_dates = []

    # Fetch today through 5 days ahead (Eastern time)
    for days_ahead in range(6):
        target_date = now_eastern + timedelta(days=days_ahead)
        date_str = target_date.strftime("%Y%m%d")
        url = f"{BASE_URL}&dates={date_str}"

        print(f"      Querying ESPN for: {date_str} ({target_date.strftime('%A, %B %d')})")

        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
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
                    print(f"         Could not parse spread from: {details!r}")
                    spread_val = 0.0

                # Extract per-side spread odds and moneyline odds
                point_spread = odds.get('pointSpread', {})
                home_spread_odds = (point_spread.get('home', {}).get('close', {}).get('odds', '')
                                    or point_spread.get('home', {}).get('open', {}).get('odds', ''))
                away_spread_odds = (point_spread.get('away', {}).get('close', {}).get('odds', '')
                                    or point_spread.get('away', {}).get('open', {}).get('odds', ''))
                ml_block = odds.get('moneyline', {})
                home_ml_odds = (ml_block.get('home', {}).get('close', {}).get('odds', '')
                                or ml_block.get('home', {}).get('open', {}).get('odds', ''))
                away_ml_odds = (ml_block.get('away', {}).get('close', {}).get('odds', '')
                                or ml_block.get('away', {}).get('open', {}).get('odds', ''))

                # Neutral site + venue
                is_neutral = int(comp.get('neutralSite', False))
                venue = comp.get('venue', {})
                venue_addr = venue.get('address', {})

                game_id = event['id']
                if not any(g['id'] == game_id for g in games):
                    games.append({
                        'id': game_id,
                        'home_raw': home_raw,  # Keep original ESPN name
                        'away_raw': away_raw,  # Keep original ESPN name
                        'spread': spread_val,  # May be 0 if ESPN doesn't have it
                        'date': game_date,
                        'raw_odds': raw_odds,
                        'has_espn_spread': spread_val != 0.0,
                        'is_neutral': is_neutral,
                        'venue_city': venue_addr.get('city', ''),
                        'venue_state': venue_addr.get('state', ''),
                        'home_spread_odds': home_spread_odds,
                        'away_spread_odds': away_spread_odds,
                        'home_ml_odds': home_ml_odds,
                        'away_ml_odds': away_ml_odds,
                    })
                    
        except requests.RequestException as e:
            print(f"         HTTP/network error fetching {date_str}: {e}")
            failed_dates.append(date_str)
        except ValueError as e:
            print(f"         JSON parse error fetching {date_str}: {e}")
            failed_dates.append(date_str)

    if failed_dates:
        print(f"\n      WARNING: Failed to fetch {len(failed_dates)} date(s): {', '.join(failed_dates)}")
        print(f"      Games for those dates are MISSING from predictions.")

    return sorted(games, key=lambda x: x['date'])

def fetch_kalshi_markets(league=None):
    """Fetch Kalshi college basketball markets and build mapper."""
    target_league = ACTIVE_LEAGUE if league is None else normalize_league(league)
    league_label = "NCAAW" if target_league == "womens" else "NCAAM"
    print(f"   -> Fetching Kalshi markets ({league_label})...")

    api_key = os.getenv("KALSHI_API_KEY")
    if not api_key:
        print("      KALSHI_API_KEY not set. Skipping Kalshi integration.")
        return None, None

    try:
        client = KalshiClient(api_key)
        markets = client.get_college_basketball_markets(league=target_league)

        if markets:
            print(f"      Found {len(markets)} Kalshi {league_label} markets")
            mapper = MarketMapper(markets)
            return client, mapper
        else:
            print(f"      No Kalshi {league_label} markets found")
            return client, None
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
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
            return None, None

        if len(spread_markets) > 1:
            # Multiple spread contracts (alt lines) -- can't reliably
            # identify the main line without volume data, so decline.
            return None, None

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
            yes_price = prices.get("yes_price")
            no_price = prices.get("no_price")

            if is_underdog:
                kalshi_side = "NO"
                bet_price = no_price
            else:
                kalshi_side = "YES"
                bet_price = yes_price

            if bet_price is None:
                print(f"      No Kalshi ask price available for {ticker} -- market may be illiquid")
                return result

            implied_prob = kalshi_implied_prob(bet_price)

            edge = calculate_edge(model_prob, implied_prob)
            rating = get_rating(edge)
            units = recommended_units(edge, implied_prob)

            result = {
                "Kalshi_Yes": yes_price,
                "Kalshi_No": no_price,
                "Kalshi_Price": bet_price,
                "Kalshi_Fee": round(kalshi_fee_cents(bet_price), 1),
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


def _infer_yes_team_from_game_market(market, home_team, away_team):
    """Infer which team maps to YES for a Kalshi GAME market."""
    from kalshi.market_mapper import extract_school_keyword

    home_keyword = extract_school_keyword(home_team).lower()
    away_keyword = extract_school_keyword(away_team).lower()
    title = str(market.get("title", "")).lower()
    rules = str(market.get("rules_primary", "")).lower()

    def _is_match(candidate: str, team_keyword: str) -> bool:
        if not candidate:
            return False
        cleaned = re.sub(r"[^a-z0-9\s]", " ", candidate.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return team_keyword in cleaned or cleaned in team_keyword

    patterns = [
        r"if\s+(.+?)\s+wins",
        r"resolves?\s+to\s+yes\s+if\s+(.+?)\s+wins",
        r"yes\s+if\s+(.+?)\s+wins",
        r"will\s+(.+?)\s+beat",
        r"does\s+(.+?)\s+win",
    ]
    def _clean(text: str) -> str:
        c = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return re.sub(r"\s+", " ", c).strip()

    for pattern in patterns:
        for source in (rules, title):
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip()
            home_match = _is_match(candidate, home_keyword)
            away_match = _is_match(candidate, away_keyword)
            if home_match and away_match:
                # Both keywords match (e.g. "virginia" is substring of
                # "virginia tech"). Prefer the keyword closer in length
                # to the candidate text. On tie (e.g. "kansas st" is
                # equidistant from "kansas" and "kansas state"), prefer the
                # keyword the candidate is an abbreviation of.
                cleaned = _clean(candidate)
                home_dist = abs(len(cleaned) - len(home_keyword))
                away_dist = abs(len(cleaned) - len(away_keyword))
                if away_dist < home_dist:
                    return away_team
                if home_dist < away_dist:
                    return home_team
                # Tied -- candidate is abbreviation of the longer keyword
                if cleaned in away_keyword and cleaned not in home_keyword:
                    return away_team
                if cleaned in home_keyword and cleaned not in away_keyword:
                    return home_team
                return home_team
            if home_match:
                return home_team
            if away_match:
                return away_team

    home_in_title = home_keyword in title
    away_in_title = away_keyword in title
    if home_in_title and not away_in_title:
        return home_team
    if away_in_title and not home_in_title:
        return away_team

    return None


def get_kalshi_game_live_rating(edge, side_prob, price_cents):
    """Apply live GAME filters so edge alone cannot promote tail punts."""
    if edge is None or side_prob is None or price_cents is None:
        return EdgeRating.PASS.value

    edge = float(edge)
    side_prob = float(side_prob)
    price_cents = float(price_cents)

    if (
        edge >= 0.08
        and side_prob >= GAME_STRONG_MIN_PROB
        and GAME_STRONG_MIN_PRICE <= price_cents <= GAME_STRONG_MAX_PRICE
    ):
        return EdgeRating.STRONG.value

    if (
        edge >= 0.04
        and side_prob >= GAME_GOOD_MIN_PROB
        and GAME_GOOD_MIN_PRICE <= price_cents <= GAME_GOOD_MAX_PRICE
    ):
        return EdgeRating.GOOD.value

    return EdgeRating.PASS.value


def get_kalshi_game_edge(
    client,
    mapper,
    home_team,
    away_team,
    game_date,
    model_home_win_prob,
):
    """
    Get best Kalshi GAME market side and edge based on model P(home wins).
    """
    result = {
        "Kalshi_Yes": None,
        "Kalshi_No": None,
        "Kalshi_Price": None,
        "Kalshi_Fee": None,
        "Edge": None,
        "Edge_Pct": None,
        "Rating": None,
        "Units": None,
        "Kalshi_Ticker": None,
        "Kalshi_Side": None,
        "Kalshi_Title": None,
        "Kalshi_Yes_Team": None,
        "Picked_Team": None,
    }

    if not client or not mapper:
        return result

    try:
        all_markets = mapper.find_all_markets_for_game(home_team, away_team, game_date)
        game_markets = [m for m in all_markets if "GAME" in m.get("ticker", "")]
        if not game_markets:
            return result

        best_choice = None
        for market in game_markets:
            yes_team = _infer_yes_team_from_game_market(market, home_team, away_team)
            if not yes_team:
                continue

            ticker = market.get("ticker", "")
            prices = client.get_market_prices(ticker) if ticker else mapper.get_market_prices(market)
            yes_price = prices.get("yes_price")
            no_price = prices.get("no_price")
            if yes_price is None and no_price is None:
                continue

            yes_prob = model_home_win_prob if yes_team == home_team else (1.0 - model_home_win_prob)
            side_candidates = []

            if yes_price is not None:
                implied_yes = kalshi_implied_prob(yes_price)
                edge_yes = calculate_edge(yes_prob, implied_yes)
                side_candidates.append(
                    {
                        "side": "YES",
                        "price": yes_price,
                        "prob": yes_prob,
                        "edge": edge_yes,
                        "picked_team": yes_team,
                    }
                )

            if no_price is not None:
                no_prob = 1.0 - yes_prob
                implied_no = kalshi_implied_prob(no_price)
                edge_no = calculate_edge(no_prob, implied_no)
                picked_team = away_team if yes_team == home_team else home_team
                side_candidates.append(
                    {
                        "side": "NO",
                        "price": no_price,
                        "prob": no_prob,
                        "edge": edge_no,
                        "picked_team": picked_team,
                    }
                )

            if not side_candidates:
                continue

            best_market_side = max(side_candidates, key=lambda c: c["edge"])
            best_market_side.update(
                {
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "ticker": prices.get("ticker", ""),
                    "title": prices.get("title", ""),
                    "yes_team": yes_team,
                }
            )

            if best_choice is None or best_market_side["edge"] > best_choice["edge"]:
                best_choice = best_market_side

        if best_choice is None:
            return result

        rating = get_kalshi_game_live_rating(
            best_choice["edge"],
            best_choice["prob"],
            best_choice["price"],
        )
        units = (
            recommended_units(best_choice["edge"], kalshi_implied_prob(best_choice["price"]))
            if rating in VALUE_RATINGS
            else 0.0
        )
        return {
            "Kalshi_Yes": best_choice["yes_price"],
            "Kalshi_No": best_choice["no_price"],
            "Kalshi_Price": best_choice["price"],
            "Kalshi_Fee": round(kalshi_fee_cents(best_choice["price"]), 1),
            "Edge": best_choice["edge"],
            "Edge_Pct": f"{best_choice['edge'] * 100:+.1f}%",
            "Rating": rating,
            "Units": units,
            "Kalshi_Ticker": best_choice["ticker"],
            "Kalshi_Side": best_choice["side"],
            "Kalshi_Title": best_choice["title"],
            "Kalshi_Yes_Team": best_choice["yes_team"],
            "Picked_Team": best_choice["picked_team"],
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        print(
            f"      Unexpected Kalshi GAME market error for "
            f"{away_team} @ {home_team}: {type(e).__name__}: {e}"
        )
        return result


def calculate_production_features(row, h_stats, a_stats, game_date=None, torvik_context=None):
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

    # Win-model team strength features used by GAME scoring.
    row['prev_win_pct'] = h_stats.get('prev_win_pct', 0.5)
    row['prev_season_team_score'] = h_stats.get('prev_season_team_score', 70.0)
    row['prev_roll3_team_score'] = h_stats.get('prev_roll3_team_score', 70.0)
    row['prev_season_off_rating'] = h_stats.get('prev_season_off_rating', 100.0)
    row['opp_season_off_rating'] = a_stats.get('prev_season_off_rating', 100.0)
    row['off_rating_gap'] = row['prev_season_off_rating'] - row['opp_season_off_rating']
    row['diff_prev_season_team_score'] = h_stats.get('prev_season_team_score', 70.0) - a_stats.get('prev_season_team_score', 70.0)
    row['diff_prev_roll3_team_score'] = h_stats.get('prev_roll3_team_score', 70.0) - a_stats.get('prev_roll3_team_score', 70.0)

    # Spread interaction features.
    row['spread_abs'] = abs(row.get('spread', 0))
    row['spread_squared'] = row.get('spread', 0) ** 2

    for col in TORVIK_GAME_FEATURE_COLUMNS:
        row.setdefault(col, 0.0)
    for col in HASLA_GAME_FEATURE_COLUMNS:
        row.setdefault(col, 0.0)

    if torvik_context and game_date is not None:
        snapshots_df = torvik_context.get("snapshots")
        team_map_df = torvik_context.get("team_map")
        if snapshots_df is not None and team_map_df is not None and not snapshots_df.empty and not team_map_df.empty:
            row.update(
                matchup_features_for_game(
                    home_team=row.get("team_name"),
                    away_team=row.get("opponent_name"),
                    game_date=game_date,
                    snapshots_df=snapshots_df,
                    team_map_df=team_map_df,
                )
            )
        hasla_snapshots = torvik_context.get("hasla_snapshots")
        hasla_team_map = torvik_context.get("hasla_team_map")
        if (
            hasla_snapshots is not None
            and hasla_team_map is not None
            and not hasla_snapshots.empty
            and not hasla_team_map.empty
        ):
            row.update(
                hasla_matchup_features_for_game(
                    home_team=row.get("team_name"),
                    away_team=row.get("opponent_name"),
                    game_date=game_date,
                    snapshots_df=hasla_snapshots,
                    team_map_df=hasla_team_map,
                )
            )

    return row

def _with_runtime_lock(func):
    """Serialize prediction runtime operations that rely on module state."""
    def wrapper(*args, **kwargs):
        with _RUNTIME_LOCK:
            return func(*args, **kwargs)
    return wrapper


@_with_runtime_lock
def fetch_games_needing_spreads(league=None):
    """Fetch schedule and return games that have no ESPN spread.

    Returns list of dicts with 'away_raw', 'home_raw', 'date', 'id' for each
    game missing a spread, or empty list if all games have spreads.
    """
    if league is not None:
        configure_league(league)

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


def get_games_needing_spreads(league="mens"):
    """Return cached list of games needing manual spreads."""
    canonical = normalize_league(league)
    return _games_needing_spreads.get(canonical)


def get_spread_model_label(league="mens"):
    """Return the display label for the configured spread model."""
    return "GBM + Sigmoid Calibration" if use_calibrated_spread_model(league) else "GBM"


@_with_runtime_lock
def main(spread_overrides=None, league="mens"):
    """Run prediction engine.

    Args:
        spread_overrides: dict mapping game matchup string to home-team spread float.
            e.g. {"Northwestern Wildcats @ Iowa Hawkeyes": -7.5}
            Convention: negative = home favored, positive = away favored.
    """
    configure_league(league)

    if spread_overrides is None:
        spread_overrides = {}

    # Get current Eastern time for dated file naming
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)

    # Load model + sigma
    try:
        model, sigma = load_model(league=ACTIVE_LEAGUE)
        feature_count = len(getattr(model, "feature_names_in_", [])) or "unknown"
        model_label = get_spread_model_label(ACTIVE_LEAGUE)
        print(f"--- PREDICTION ENGINE ({ACTIVE_LEAGUE}, {model_label}, {feature_count} features) ---")
        print(f"   Model loaded: {MODEL_FILE} (sigma={sigma:.2f})")
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
    torvik_context = {
        "snapshots": pd.DataFrame(),
        "team_map": pd.DataFrame(),
        "hasla_snapshots": pd.DataFrame(),
        "hasla_team_map": pd.DataFrame(),
    }
    if ACTIVE_LEAGUE == "mens" and TORVIK_SNAPSHOT_FILE and TORVIK_MAP_FILE:
        try:
            torvik_context = {
                "snapshots": load_snapshot_file(TORVIK_SNAPSHOT_FILE),
                "team_map": load_team_map(TORVIK_MAP_FILE, known_teams),
                "hasla_snapshots": load_hasla_snapshot_file(HASLA_SNAPSHOT_FILE) if HASLA_SNAPSHOT_FILE else pd.DataFrame(),
                "hasla_team_map": load_hasla_team_map(HASLA_MAP_FILE, known_teams) if HASLA_MAP_FILE else pd.DataFrame(),
            }
            if not torvik_context["snapshots"].empty:
                latest_snapshot = pd.to_datetime(
                    torvik_context["snapshots"]["snapshot_date"], errors="coerce"
                ).max()
                print(f"   Torvik snapshots current through: {latest_snapshot.strftime('%Y-%m-%d')}")
            if not torvik_context["hasla_snapshots"].empty:
                latest_hasla = pd.to_datetime(
                    torvik_context["hasla_snapshots"]["snapshot_date"], errors="coerce"
                ).max()
                print(f"   Haslametrics snapshots current through: {latest_hasla.strftime('%Y-%m-%d')}")
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
            print(f"   WARNING: priors context unavailable ({e})")

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

    # Optional P(win) bundle for Kalshi GAME markets
    win_bundle = None
    try:
        win_bundle = load_win_model_bundle(league=ACTIVE_LEAGUE)
        has_with_line = win_bundle.get("model_with_line") is not None
        print(
            f"   Win model loaded: {os.path.basename(WIN_MODEL_FILE)} "
            f"(no_line + {'with_line' if has_with_line else 'no_line only'})"
        )
    except (FileNotFoundError, EOFError, IOError, ValueError) as e:
        print(f"   WARNING: Win model unavailable ({e}) -- GAME markets skipped.")

    # Build venue lookups for neutral-site / distance features
    _geo_cache = load_geocode_cache()
    _team_homes = build_team_home_locations(df_hist, league=ACTIVE_LEAGUE) if 'venue_city' in df_hist.columns else {}
    if _team_homes:
        print(f"   Venue distance: {len(_team_homes)} team home locations loaded")

    # Fetch Kalshi markets
    kalshi_client, kalshi_mapper = fetch_kalshi_markets(league=ACTIVE_LEAGUE)

    spread_predictions = []
    game_predictions = []
    skipped = []
    games_needing_spreads = []
    odds_archive_records = []
    kalshi_game_archive_records = []
    _distance_stats = {"total": 0, "nonzero": 0, "neutral": 0}

    for g in schedule:
        matchup_key = f"{g['away_raw']} @ {g['home_raw']}"

        # Match team names to historical data
        home_matched = find_best_match(g['home_raw'], known_teams)
        away_matched = find_best_match(g['away_raw'], known_teams)

        # Skip if we can't match teams or don't have stats
        if not home_matched or not away_matched:
            skipped.append(f"{matchup_key} (Team matching failed)")
            continue

        if home_matched not in team_stats or away_matched not in team_stats:
            skipped.append(f"{matchup_key} (No historical stats)")
            continue

        # Resolve spread availability for spread model path.
        resolved_spread = float(g.get('spread', 0.0) or 0.0)
        spread_source = g.get("raw_odds", "0")
        has_spread_for_spread_model = bool(g.get('has_espn_spread', False))

        if not has_spread_for_spread_model:
            if matchup_key in spread_overrides:
                resolved_spread = float(spread_overrides[matchup_key])
                spread_source = f"Manual {resolved_spread:+.1f}"
                has_spread_for_spread_model = True
            else:
                # Fallback to Kalshi spread markets when ESPN has no spread.
                kalshi_spread, _ = get_kalshi_spread(
                    kalshi_mapper,
                    g['home_raw'],
                    g['away_raw'],
                    g['date'],
                )
                if kalshi_spread is not None:
                    resolved_spread = float(kalshi_spread)
                    spread_source = f"Kalshi {resolved_spread:+.1f}"
                    has_spread_for_spread_model = True
                else:
                    games_needing_spreads.append({
                        'id': g['id'],
                        'away_raw': g['away_raw'],
                        'home_raw': g['home_raw'],
                        'matchup': matchup_key,
                    })
                    skipped.append(f"{matchup_key} (No spread -- spread pick skipped)")

        archive_spread = resolved_spread if has_spread_for_spread_model else None
        odds_archive_records.append(
            build_archive_record(
                league=ACTIVE_LEAGUE,
                game_date=g["date"],
                home_team=g["home_raw"],
                away_team=g["away_raw"],
                spread=archive_spread,
                spread_source=spread_source if has_spread_for_spread_model else None,
                raw_line=spread_source,
            )
        )

        h_stats = team_stats[home_matched]
        a_stats = team_stats[away_matched]

        # Calculate rest days for both teams
        home_last_date = pd.to_datetime(h_stats.get('last_game_date', datetime.now()))
        away_last_date = pd.to_datetime(a_stats.get('last_game_date', datetime.now()))
        home_actual_rest = max(0, (g['date'].replace(tzinfo=None) - home_last_date).days)
        away_actual_rest = max(0, (g['date'].replace(tzinfo=None) - away_last_date).days)

        # Build common feature row (for both spread and game models)
        row = {
            'is_home': 1,
            'spread': resolved_spread,
            'team_name': home_matched,
            'opponent_name': away_matched,
        }
        row['is_neutral'] = g.get('is_neutral', 0)
        # Not in FEATURES yet -- computed for monitoring and future use
        row['distance_advantage'] = compute_distance_advantage(
            _team_homes.get(home_matched),
            _team_homes.get(away_matched),
            g.get('venue_city', ''),
            g.get('venue_state', ''),
            _geo_cache,
        )
        _distance_stats["total"] += 1
        if row['distance_advantage'] != 0:
            _distance_stats["nonzero"] += 1
        if row['is_neutral']:
            _distance_stats["neutral"] += 1
        row['rest_days'] = min(home_actual_rest, 7)
        row = calculate_production_features(
            row,
            h_stats,
            a_stats,
            game_date=g['date'],
            torvik_context=torvik_context,
        )

        # Format game time in Eastern once
        try:
            local_ts = g['date'].tz_convert('US/Eastern')
            time_str = local_ts.strftime("%m/%d %I:%M %p")
        except (TypeError, AttributeError):
            time_str = g['date'].strftime("%m/%d %I:%M %p")

        # Build GAME market picks (Kalshi-only) from P(win) model.
        if win_bundle is not None:
            try:
                home_win_prob, win_variant = predict_home_win_prob(row, win_bundle)
                game_kalshi = get_kalshi_game_edge(
                    kalshi_client,
                    kalshi_mapper,
                    g['home_raw'],
                    g['away_raw'],
                    g['date'],
                    home_win_prob,
                )
                if game_kalshi.get("Kalshi_Ticker"):
                    yes_team = game_kalshi.get("Kalshi_Yes_Team")
                    yes_prob = home_win_prob if yes_team == g['home_raw'] else (1.0 - home_win_prob)
                    side = game_kalshi.get("Kalshi_Side", "YES")
                    side_prob = yes_prob if side == "YES" else (1.0 - yes_prob)
                    pick_line = f"{yes_team} ML {side}"
                    edge = game_kalshi.get("Edge")

                    # Archive all candidates for backtest analysis
                    kalshi_game_archive_records.append(
                        build_game_archive_record(
                            league=ACTIVE_LEAGUE,
                            game_datetime=g["date"],
                            home_team=g["home_raw"],
                            away_team=g["away_raw"],
                            matchup=matchup_key,
                            pick=pick_line,
                            picked_team=game_kalshi.get("Picked_Team"),
                            kalshi_side=side,
                            kalshi_ticker=game_kalshi.get("Kalshi_Ticker"),
                            kalshi_title=game_kalshi.get("Kalshi_Title"),
                            kalshi_yes_team=yes_team,
                            kalshi_yes_price=game_kalshi.get("Kalshi_Yes"),
                            kalshi_no_price=game_kalshi.get("Kalshi_No"),
                            kalshi_price=game_kalshi.get("Kalshi_Price"),
                            kalshi_fee=game_kalshi.get("Kalshi_Fee"),
                            win_model_home_prob=home_win_prob,
                            conf=side_prob,
                            edge=edge,
                            edge_pct=(edge * 100.0) if edge is not None else None,
                            rating=game_kalshi.get("Rating"),
                            units=game_kalshi.get("Units"),
                        )
                    )

                    if game_kalshi.get("Rating") in VALUE_RATINGS and side_prob >= GAME_GOOD_MIN_PROB:
                        game_predictions.append({
                            "Bet_Type": "game",
                            "Date/Time": time_str,
                            "Matchup": matchup_key,
                            "Spread": resolved_spread if has_spread_for_spread_model else np.nan,
                            "Pick": pick_line,
                            "Conf": side_prob,
                            "Raw Odds": "Kalshi GAME",
                            "Rest": home_actual_rest if game_kalshi.get("Picked_Team") == g['home_raw'] else away_actual_rest,
                            "Kalshi_Side": side,
                            "Kalshi_Yes": game_kalshi.get("Kalshi_Yes"),
                            "Kalshi_No": game_kalshi.get("Kalshi_No"),
                            "Kalshi_Price": game_kalshi.get("Kalshi_Price"),
                            "Kalshi_Fee": game_kalshi.get("Kalshi_Fee"),
                            "Kalshi_Title": game_kalshi.get("Kalshi_Title"),
                            "Edge": edge,
                            "Edge_Pct": game_kalshi.get("Edge_Pct"),
                            "Rating": game_kalshi.get("Rating"),
                            "Units": min(game_kalshi.get("Units", 0), 0.5),
                            "Home_Matched": home_matched,
                            "Away_Matched": away_matched,
                            "Kalshi_Ticker": game_kalshi.get("Kalshi_Ticker"),
                            "Kalshi_Yes_Team": yes_team,
                            "Picked_Team": game_kalshi.get("Picked_Team"),
                            "Win_Model_Home_Prob": home_win_prob,
                            "Win_Model_Variant": win_variant,
                            "Std_Edge": 0.0,
                            "Std_Edge_Pct": "",
                            "Std_Rating": "PASS",
                            "Std_Units": 0.0,
                            "Breakeven_Spread": np.nan,
                        })
            except (KeyError, TypeError, ValueError) as e:
                print(f"      WARNING: GAME market prediction failed for {matchup_key}: {e}")

        # Spread pick requires a market spread (ESPN/manual/Kalshi fallback).
        if not has_spread_for_spread_model:
            continue

        # Prepare spread model input
        cols = list(model.feature_names_in_)
        input_df = pd.DataFrame([row])
        for c in cols:
            if c not in input_df.columns:
                input_df[c] = 0.0

        input_df = input_df[cols]
        input_df.columns = input_df.columns.astype(str)
        input_df = input_df.fillna({
            'diff_eFG': 0, 'diff_Rebound': 0, 'diff_TO': 0,
            'momentum_gap': 0, 'roll5_cover_margin': 0,
            'prev_games_played': 10, 'opp_win_pct': 0.5,
            'prev_blowout_rate': 0, 'prev_roll5_margin': 0,
            'prev_volatility': 10, 'is_home': 1, 'is_neutral': 0,
            'distance_advantage': 0, 'spread': 0, 'rest_days': 3,
            'prev_season_team_score': 70,
            'prev_roll3_team_score': 70,
            'diff_prev_season_team_score': 0,
            'diff_prev_roll3_team_score': 0,
            'spread_abs': 0, 'spread_squared': 0,
            'torvik_diff_adj_oe': 0,
            'torvik_diff_adj_de': 0,
            'torvik_diff_barthag': 0,
            'torvik_tempo_gap': 0,
            'torvik_diff_efg': 0,
            'torvik_diff_tor': 0,
            'torvik_diff_orb': 0,
            'torvik_diff_ftr': 0,
            'hasla_diff_rank_strength': 0,
            'hasla_diff_off_rank_strength': 0,
            'hasla_diff_def_rank_strength': 0,
        }).fillna(0)

        prob = model.predict_proba(input_df)[0][1]
        conf = max(prob, 1 - prob)

        if prob > 0.5:
            sign = "+" if resolved_spread > 0 else ""
            pick_str = f"{g['home_raw']} {sign}{resolved_spread}"
            picked_team = g['home_raw']
            picked_spread = resolved_spread
            picked_team_rest = home_actual_rest
        else:
            away_spread = -1 * resolved_spread
            sign = "+" if away_spread > 0 else ""
            pick_str = f"{g['away_raw']} {sign}{away_spread}"
            picked_team = g['away_raw']
            picked_spread = away_spread
            picked_team_rest = away_actual_rest

        kalshi_data = get_kalshi_edge(
            kalshi_client,
            kalshi_mapper,
            g['home_raw'],
            g['away_raw'],
            g['date'],
            resolved_spread,
            conf,
            picked_team,
            picked_spread,
        )

        is_home_pick = (prob > 0.5)
        try:
            line_shopping = calculate_line_shopping(
                conf,
                sigma,
                picked_spread,
                picked_team,
                is_home_pick,
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            print(f"      WARNING: Line shopping failed for {matchup_key}: {type(e).__name__}: {e}")
            line_shopping = LineShoppingResult(
                picked_team=picked_team,
                market_spread=picked_spread,
                breakeven_spread=None,
                recommendations=[],
            )

        prediction_row = {
            "Bet_Type": "spread",
            "Date/Time": time_str,
            "Matchup": matchup_key,
            "Spread": resolved_spread,
            "Pick": pick_str,
            "Conf": conf,
            "Raw Odds": spread_source,
            "Rest": picked_team_rest,
            "Kalshi_Side": kalshi_data.get("Kalshi_Side"),
            "Kalshi_Price": kalshi_data.get("Kalshi_Price"),
            "Kalshi_Fee": kalshi_data.get("Kalshi_Fee"),
            "Kalshi_Title": kalshi_data.get("Kalshi_Title"),
            "Edge": kalshi_data.get("Edge"),
            "Edge_Pct": kalshi_data.get("Edge_Pct"),
            "Rating": kalshi_data.get("Rating"),
            "Units": kalshi_data.get("Units"),
            "Home_Matched": home_matched,
            "Away_Matched": away_matched,
            "Kalshi_Ticker": kalshi_data.get("Kalshi_Ticker"),
            "Breakeven_Spread": line_shopping.breakeven_spread,
            "Line_Shopping_Data": line_shopping,
            "Std_Edge": _compute_std_edge(conf, g, prob > 0.5, "spread"),
            "Std_Edge_Pct": f"{_compute_std_edge(conf, g, prob > 0.5, 'spread') * 100:+.1f}%",
            "Std_Rating": get_rating(_compute_std_edge(conf, g, prob > 0.5, "spread")).value,
            "Std_Units": recommended_units(
                _compute_std_edge(conf, g, prob > 0.5, "spread"),
                _get_std_implied_prob(g, prob > 0.5, "spread"),
            ),
        }

        if g['home_raw'] not in pick_str and g['away_raw'] not in pick_str:
            print(f"      WARNING: Pick '{pick_str}' doesn't match matchup '{prediction_row['Matchup']}'")

        spread_predictions.append(prediction_row)

    spread_df = (
        pd.DataFrame(spread_predictions).sort_values(by="Conf", ascending=False)
        if spread_predictions else pd.DataFrame()
    )
    added_archive_rows = append_archive_records(
        odds_archive_records,
        get_league_artifact_paths(BASE_DIR, ACTIVE_LEAGUE)["odds_archive_file"],
    )
    if added_archive_rows:
        print(f"   -> Archived {added_archive_rows} market line snapshot(s)")
    added_game_archive_rows = append_kalshi_game_archive_records(
        kalshi_game_archive_records,
        get_league_artifact_paths(BASE_DIR, ACTIVE_LEAGUE)["kalshi_game_archive_file"],
    )
    if added_game_archive_rows:
        print(f"   -> Archived {added_game_archive_rows} Kalshi GAME snapshot(s)")
    game_df = (
        pd.DataFrame(game_predictions).sort_values(by="Conf", ascending=False)
        if game_predictions else pd.DataFrame()
    )

    _latest_predictions[ACTIVE_LEAGUE] = spread_df
    _latest_game_predictions[ACTIVE_LEAGUE] = game_df

    # Save combined predictions for app, grading, and live price checks.
    combined_csv_frames = []
    if not spread_df.empty:
        combined_csv_frames.append(spread_df.drop(columns=["Line_Shopping_Data"], errors="ignore"))
    if not game_df.empty:
        combined_csv_frames.append(game_df.drop(columns=["Line_Shopping_Data"], errors="ignore"))

    if combined_csv_frames:
        csv_df = pd.concat(combined_csv_frames, ignore_index=True)
        csv_df = csv_df.sort_values(by="Conf", ascending=False)
    else:
        csv_df = pd.DataFrame()

    # Always write the CSV so stale data from previous runs is cleared.
    csv_df.to_csv(OUTPUT_FILE, index=False)

    if not csv_df.empty:
        archive_file = os.path.join(
            BASE_DIR,
            f"{PREDICTIONS_ARCHIVE_PREFIX}_{now_eastern.strftime('%Y%m%d')}.csv",
        )
        csv_df.to_csv(archive_file, index=False)

        print(
            f"\nSUCCESS: Generated {len(csv_df)} predictions "
            f"({len(spread_df)} spread, {len(game_df)} game)"
        )
        print(f"   Saved to: {OUTPUT_FILE}")
        print(f"   Archive: {archive_file}")

        print("\nPREDICTION SUMMARY:")
        for _, row in csv_df.head(5).iterrows():
            bet_type = row.get("Bet_Type", "spread")
            print(f"   [{bet_type}] {row['Matchup']}")
            print(f"      Pick: {row['Pick']} (Conf: {row['Conf']:.1%})")

        value_bets = csv_df[
            (csv_df['Std_Rating'].isin(VALUE_RATINGS)) |
            (csv_df['Rating'].isin(VALUE_RATINGS))
        ]
        if len(value_bets) > 0:
            print(f"\nVALUE BETS ({len(value_bets)} found):")
            for _, row in value_bets.iterrows():
                std_rating = row.get('Std_Rating', 'PASS')
                if pd.isna(std_rating):
                    std_rating = 'PASS'
                kalshi_rating = row.get('Rating', None) if pd.notna(row.get('Rating')) else None
                std_rank = RATING_RANK.get(std_rating, 0)
                kalshi_rank = RATING_RANK.get(kalshi_rating, 0)
                best_rating = kalshi_rating if kalshi_rank > std_rank else std_rating
                print(f"   [{row.get('Bet_Type', 'spread')}:{best_rating}] {row['Pick']}")
                if kalshi_rating in VALUE_RATINGS:
                    side = row['Kalshi_Side'] if row['Kalshi_Side'] else "?"
                    fee = row.get('Kalshi_Fee', 0) or 0
                    print(f"      Kalshi: Buy {side} @ {row['Kalshi_Price']}c + {fee:.1f}c fee | Edge: {row['Edge_Pct']} | {row['Units']:.1f}U")
                if std_rating in VALUE_RATINGS:
                    print(f"      Std Book: Edge {row['Std_Edge_Pct']} | {row['Std_Units']:.1f}U")
    else:
        print("\nNo predictions generated.")
    
    # Distance feature guardrail
    dt = _distance_stats
    if dt["total"] > 0:
        pct = dt["nonzero"] / dt["total"]
        level = "WARNING" if pct < 0.5 else "INFO"
        print(
            f"\n   [{level}] distance coverage monitor: "
            f"{dt['nonzero']}/{dt['total']} games non-zero ({pct:.0%}), "
            f"{dt['neutral']} neutral-site"
        )
        if pct < 0.5:
            print(f"   [{level}] Low distance coverage -- check venue data and geocode cache")

    # Show skipped games
    if skipped:
        print(f"\nSkipped {len(skipped)} games:")
        for s in skipped[:5]:
            print(f"   - {s}")

    # Persist any newly geocoded venues
    if _geo_cache:
        save_geocode_cache(_geo_cache)

    # Store games that still need manual spreads
    _games_needing_spreads[ACTIVE_LEAGUE] = games_needing_spreads
    if games_needing_spreads:
        print(f"\n{len(games_needing_spreads)} game(s) need manual spread entry.")

def get_latest_predictions(league="mens"):
    """Return the latest predictions DataFrame with line shopping data."""
    canonical = normalize_league(league)
    return _latest_predictions.get(canonical)


def get_latest_game_predictions(league="mens"):
    """Return latest Kalshi GAME market picks."""
    canonical = normalize_league(league)
    return _latest_game_predictions.get(canonical)


@_with_runtime_lock
def check_live_prices(league="mens"):
    """Re-fetch current Kalshi prices for today's value picks and recalculate edge.

    Reads the daily predictions CSV, filters to value-rated rows with a Kalshi
    ticker, fetches live prices from the Kalshi API, and prints an updated table
    showing current edge so the user can verify value before placing a bet.
    """
    configure_league(league)
    print(f"--- LIVE PRICE CHECK ({ACTIVE_LEAGUE}) ---\n")

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

    # Filter to value picks (STRONG or GOOD from either source)
    std_col = df_kalshi["Std_Rating"] if "Std_Rating" in df_kalshi.columns else pd.Series("PASS", index=df_kalshi.index)
    rating_col = df_kalshi["Rating"] if "Rating" in df_kalshi.columns else pd.Series("PASS", index=df_kalshi.index)
    is_value = (
        std_col.isin(VALUE_RATINGS) |
        rating_col.isin(VALUE_RATINGS)
    )
    df_value = df_kalshi[is_value].copy()

    if df_value.empty:
        print("No value-rated picks with Kalshi tickers found.")
        return

    # Initialize Kalshi client
    api_key = os.getenv("KALSHI_API_KEY")
    if not api_key:
        print("KALSHI_API_KEY not set. Cannot fetch live prices.")
        return

    client = KalshiClient(api_key)

    # Fetch live prices and recalculate edge for each pick
    results = []
    for _, row in df_value.iterrows():
        ticker = row["Kalshi_Ticker"]
        side = row.get("Kalshi_Side", "YES")
        model_prob = row["Conf"]
        pick = row["Pick"]

        def _placeholder(live_price_str):
            return {
                "Pick": pick,
                "Model%": f"{model_prob:.1%}",
                "Live Price": live_price_str,
                "Edge": "--",
                "Rating": "--",
                "Units": "--",
            }

        try:
            prices = client.get_market_prices(ticker)
        except (requests.RequestException, ValueError, TypeError) as e:
            print(f"   Error fetching {ticker}: {e}")
            results.append(_placeholder("ERR"))
            continue

        yes_price = prices.get("yes_price")
        no_price = prices.get("no_price")

        if yes_price is None and no_price is None:
            results.append(_placeholder("N/A"))
            continue

        # Use the correct price based on which side we're betting
        live_price = yes_price if side == "YES" else no_price

        if live_price is None:
            results.append(_placeholder(f"{side} N/A"))
            continue

        implied_prob = kalshi_implied_prob(live_price)
        edge = calculate_edge(model_prob, implied_prob)
        rating = get_rating(edge)
        units = recommended_units(edge, implied_prob)

        fee = kalshi_fee_cents(live_price)
        results.append({
            "Pick": pick,
            "Model%": f"{model_prob:.1%}",
            "Live Price": f"{side} @ {live_price}c + {fee:.1f}c fee",
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
    widths = {
        h: max(len(h), *(len(str(r[h])) for r in results))
        for h in headers
    }

    # Print header
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for r in results:
        line = "  ".join(str(r[h]).ljust(widths[h]) for h in headers)
        print(line)

    # Summary
    still_value = sum(1 for r in results if r["Rating"] in VALUE_RATINGS)
    total = len(results)
    print(f"\n{still_value}/{total} picks still STRONG/GOOD at live prices.")

    if still_value < total:
        print("Some picks have lost edge -- consider skipping those bets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CBB prediction engine -- generate picks or check live prices."
    )
    parser.add_argument(
        "--league",
        default="mens",
        help="League to run: mens or womens (aliases supported).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-fetch live Kalshi prices for today's value picks and show updated edge.",
    )
    args = parser.parse_args()

    if args.check:
        check_live_prices(args.league)
    else:
        main(league=args.league)
