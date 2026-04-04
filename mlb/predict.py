"""MLB daily prediction pipeline."""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from betting import (
    american_odds_to_implied_prob,
    calculate_edge,
    get_rating,
    recommended_units,
    kalshi_implied_prob,
    STANDARD_IMPLIED_PROB,
)
from betting.ev_calculator import kalshi_fee_cents
from league_config import (
    get_league_artifact_paths,
    get_scoreboard_base_url,
    normalize_league,
)
from model import load_model, get_feature_list, TARGET_BY_LEAGUE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAGUE = "mlb"

# MLB team name map -- ESPN display names are stable for 30 teams
MLB_TEAM_MAP_FILE = os.path.join(os.path.dirname(__file__), "team_map.json")


def _load_team_map():
    if os.path.exists(MLB_TEAM_MAP_FILE):
        with open(MLB_TEAM_MAP_FILE) as f:
            return json.load(f)
    return {}


TEAM_MAP = _load_team_map()


def get_latest_stats(df):
    """Get the most recent entering stats for each team from training data."""
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    latest_stats = {}

    for team in df["team"].unique():
        last_game = df[df["team"] == team].iloc[-1]
        stats = {}
        for col in df.columns:
            if any(x in col for x in ["season_", "roll", "prev_", "opp_win_pct", "bullpen_era", "pyth_wpct"]):
                stats[col] = last_game[col]
        stats["last_game_date"] = last_game["date"]
        stats["prev_games_played"] = last_game.get("prev_games_played", 10)
        stats["prev_volatility"] = last_game.get("prev_volatility", 2.0)
        stats["prev_win_pct"] = last_game.get("prev_win_pct", 0.5)
        stats["prev_roll10_win_pct"] = last_game.get("prev_roll10_win_pct", 0.5)
        latest_stats[team] = stats

    return latest_stats


def get_latest_pitcher_stats(df):
    """Get the most recent rolling stats for each starting pitcher."""
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    pitcher_stats = {}

    if "starting_pitcher" not in df.columns:
        return pitcher_stats

    for pitcher in df["starting_pitcher"].dropna().unique():
        if not pitcher:
            continue
        pitcher_games = df[df["starting_pitcher"] == pitcher]
        last = pitcher_games.iloc[-1]
        pitcher_stats[pitcher] = {
            "sp_roll_era": last.get("sp_roll_era", float("nan")),
            "sp_roll_whip": last.get("sp_roll_whip", float("nan")),
            "sp_roll_k9": last.get("sp_roll_k9", float("nan")),
            "sp_roll_ip": last.get("sp_roll_ip", float("nan")),
        }

    return pitcher_stats


def find_best_match(name, known_teams):
    """Match ESPN team name to historical data team name."""
    if name in TEAM_MAP:
        return TEAM_MAP[name]
    if name in known_teams:
        return name
    # MLB team names are stable; a simple substring match usually works
    for kt in known_teams:
        if kt in name or name in kt:
            return kt
    from difflib import get_close_matches
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    print(f"      WARNING: Could not match '{name}' to historical data")
    return None


def fetch_schedule(league=LEAGUE):
    """Fetch upcoming MLB games from ESPN."""
    print("   -> Fetching MLB schedule...")
    base_url = get_scoreboard_base_url(league)
    eastern = pytz.timezone("US/Eastern")
    now_eastern = datetime.now(eastern)

    games = []
    for days_ahead in range(3):
        target_date = now_eastern + timedelta(days=days_ahead)
        date_str = target_date.strftime("%Y%m%d")
        url = f"{base_url}&dates={date_str}"
        print(f"      Querying ESPN for: {date_str}")

        try:
            res = requests.get(url, timeout=15)
            res.raise_for_status()
            data = res.json()
        except requests.RequestException as e:
            print(f"      WARNING: Failed to fetch {date_str}: {e}")
            continue
        except ValueError as e:
            print(f"      WARNING: Invalid JSON from ESPN for {date_str}: {e}")
            continue

        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            if not comp.get("competitors"):
                continue

            home = away = None
            for c in comp["competitors"]:
                if c.get("homeAway") == "home":
                    home = c
                elif c.get("homeAway") == "away":
                    away = c

            if home is None or away is None:
                continue

            # Skip completed games
            state = event.get("status", {}).get("type", {}).get("state", "")
            if state == "post":
                continue

            game_date = pd.to_datetime(event["date"])
            home_name = home["team"]["displayName"]
            away_name = away["team"]["displayName"]
            home_abbr = home["team"].get("abbreviation", "")
            away_abbr = away["team"].get("abbreviation", "")

            # Starting pitchers
            home_sp = _extract_sp(home)
            away_sp = _extract_sp(away)

            # Venue
            venue = comp.get("venue", {})

            # Moneyline odds from ESPN/DraftKings
            odds_block = comp.get("odds", [{}])[0] if comp.get("odds") else {}
            ml_block = odds_block.get("moneyline", {})
            home_ml_odds = (ml_block.get("home", {}).get("close", {}).get("odds", "")
                            or ml_block.get("home", {}).get("open", {}).get("odds", ""))
            away_ml_odds = (ml_block.get("away", {}).get("close", {}).get("odds", "")
                            or ml_block.get("away", {}).get("open", {}).get("odds", ""))

            games.append({
                "game_date": game_date,
                "home_team": home_name,
                "away_team": away_name,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "home_sp": home_sp.get("name", "TBD"),
                "home_sp_era": home_sp.get("era", float("nan")),
                "away_sp": away_sp.get("name", "TBD"),
                "away_sp_era": away_sp.get("era", float("nan")),
                "venue": venue.get("fullName", ""),
                "venue_indoor": int(venue.get("indoor", False)),
                "home_ml_odds": home_ml_odds,
                "away_ml_odds": away_ml_odds,
            })

    print(f"      Found {len(games)} upcoming games")
    return games


def _extract_sp(competitor):
    """Extract starting pitcher info from ESPN competitor."""
    probables = competitor.get("probables", [])
    if not probables:
        return {"name": "", "era": float("nan")}
    pitcher = probables[0]
    athlete = pitcher.get("athlete", {})
    era = float("nan")
    for stat in pitcher.get("statistics", []):
        if stat.get("abbreviation") == "ERA" or stat.get("name") == "earnedRunAverage":
            try:
                era = float(stat.get("displayValue", "nan"))
            except (TypeError, ValueError):
                pass
            break
    return {"name": athlete.get("fullName", ""), "era": era}


def build_feature_row(home_stats, away_stats, home_sp_era, away_sp_era,
                      home_sp_name="", away_sp_name="", pitcher_stats=None,
                      game_date=None, venue_name="", venue_indoor=0):
    """Build a feature row for model inference from entering team stats.

    pitcher_stats: dict of {pitcher_name: {sp_roll_era, sp_roll_whip, ...}}
    from the latest training data, representing their entering rolling stats.
    game_date: scheduled game datetime, used for rest day calculation.
    venue_name: stadium name for weather lookup.
    venue_indoor: 1 if dome/retractable, 0 if outdoor.
    """
    if pitcher_stats is None:
        pitcher_stats = {}
    if game_date is None:
        game_date = datetime.now()
    features = get_feature_list(LEAGUE)
    row = {}

    row["is_home"] = 1

    # Rest days relative to the scheduled game date (not "now")
    home_last = home_stats.get("last_game_date")
    if home_last is not None:
        ref_date = pd.to_datetime(game_date).tz_localize(None) if hasattr(pd.to_datetime(game_date), 'tz') else pd.to_datetime(game_date)
        last_date = pd.to_datetime(home_last)
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        rest = (ref_date - last_date).days
        row["rest_days"] = min(max(rest, 0), 7)
    else:
        row["rest_days"] = 1  # MLB teams play almost daily

    # Starting pitcher ESPN entering ERA (point-in-time, clean)
    row["sp_era"] = home_sp_era if not pd.isna(home_sp_era) else 4.50
    row["opp_sp_era"] = away_sp_era if not pd.isna(away_sp_era) else 4.50

    # Starting pitcher rolling stats (from game-log derived data in training CSV)
    home_sp_stats = pitcher_stats.get(home_sp_name, {})
    away_sp_stats = pitcher_stats.get(away_sp_name, {})
    row["sp_roll_era"] = home_sp_stats.get("sp_roll_era", float("nan"))
    row["sp_roll_whip"] = home_sp_stats.get("sp_roll_whip", float("nan"))
    row["sp_roll_k9"] = home_sp_stats.get("sp_roll_k9", float("nan"))
    row["sp_roll_ip"] = home_sp_stats.get("sp_roll_ip", float("nan"))
    row["opp_sp_roll_era"] = away_sp_stats.get("sp_roll_era", float("nan"))

    # Team rolling stats
    for col in ["prev_roll10_runs_per_game", "prev_roll10_runs_allowed",
                 "prev_season_runs_per_game", "prev_season_runs_allowed",
                 "prev_games_played", "prev_win_pct", "prev_roll10_win_pct",
                 "prev_volatility"]:
        row[col] = home_stats.get(col, float("nan"))

    # Opponent quality
    row["opp_win_pct"] = away_stats.get("prev_win_pct", 0.5)

    # Pythagorean win% (from entering team stats)
    home_pyth_season = home_stats.get("prev_season_pyth_wpct", float("nan"))
    away_pyth_season = away_stats.get("prev_season_pyth_wpct", float("nan"))
    row["prev_season_pyth_wpct"] = home_pyth_season if not pd.isna(home_pyth_season) else 0.5
    row["prev_roll10_pyth_wpct"] = home_stats.get("prev_roll10_pyth_wpct", 0.5)
    row["pyth_wpct_diff"] = (
        float(row["prev_season_pyth_wpct"]) - (float(away_pyth_season) if not pd.isna(away_pyth_season) else 0.5)
    )

    # Bullpen ERA differential
    home_bp = home_stats.get("bullpen_era", float("nan"))
    away_bp = away_stats.get("bullpen_era", float("nan"))
    row["bullpen_era_diff"] = (
        (float(away_bp) - float(home_bp))
        if not (pd.isna(home_bp) or pd.isna(away_bp))
        else 0.0
    )

    # Wind speed from venue + game date
    if venue_name and not venue_indoor and game_date is not None:
        try:
            from mlb.weather import fetch_game_weather
            date_str = pd.to_datetime(game_date).strftime("%Y-%m-%d")
            game_time_str = pd.to_datetime(game_date).strftime("%H:%M") if game_date else None
            weather = fetch_game_weather(venue_name, date_str, game_time_str)
            row["wind_speed"] = weather.get("wind_speed", 0.0)
        except Exception:
            row["wind_speed"] = 0.0
    else:
        row["wind_speed"] = 0.0

    # Differentials (game-derived, no leaky aggregates)
    home_rpg = home_stats.get("prev_roll10_runs_per_game", 0)
    away_rpg = away_stats.get("prev_roll10_runs_per_game", 0)
    row["roll10_rpg_diff"] = float(home_rpg or 0) - float(away_rpg or 0)

    home_ra = home_stats.get("prev_roll10_runs_allowed", 0)
    away_ra = away_stats.get("prev_roll10_runs_allowed", 0)
    row["roll10_ra_diff"] = float(away_ra or 0) - float(home_ra or 0)

    # Roll5 short-term form differential
    home_rpg5 = home_stats.get("prev_roll5_runs_per_game", 0)
    away_rpg5 = away_stats.get("prev_roll5_runs_per_game", 0)
    row["roll5_rpg_diff"] = float(home_rpg5 or 0) - float(away_rpg5 or 0)

    row["sp_era_diff"] = (
        (float(away_sp_era) - float(home_sp_era))
        if not (pd.isna(home_sp_era) or pd.isna(away_sp_era))
        else 0.0
    )

    home_sp_roll = home_sp_stats.get("sp_roll_era", 4.5)
    away_sp_roll = away_sp_stats.get("sp_roll_era", 4.5)
    row["sp_roll_era_diff"] = float(away_sp_roll or 4.5) - float(home_sp_roll or 4.5)

    # Fill any missing features with neutral defaults
    for f in features:
        if f not in row or pd.isna(row.get(f)):
            if "pyth_wpct" in f:
                row[f] = 0.5
            elif "bullpen_era_diff" in f:
                row[f] = 0.0
            elif "win_pct" in f:
                row[f] = 0.5
            elif "era" in f:
                row[f] = 4.50
            elif "whip" in f:
                row[f] = 1.30
            elif "k9" in f:
                row[f] = 8.0
            elif "ip" in f:
                row[f] = 5.5
            else:
                row[f] = 0.0

    return row


def _get_kalshi_edge(client, mapper, home_team, away_team, game_date,
                     prob_home_win, pick, game_time=""):
    """Calculate edge against Kalshi GAME market prices.

    For moneyline, each game has two GAME tickers (one YES per team).
    We find the one matching our pick and calculate edge.
    """
    result = {
        "Kalshi_Ticker": None,
        "Kalshi_Side": None,
        "Kalshi_Price": None,
        "Kalshi_Fee": None,
        "Edge": None,
        "Edge_Pct": None,
        "Rating": None,
        "Units": None,
    }

    if not mapper or not client:
        return result

    # Market lookup -- catch API/network errors only
    try:
        market = mapper.find_market(home_team, away_team, game_date, "GAME", game_time=game_time)
    except (requests.RequestException, KeyError) as e:
        print(f"      Kalshi market lookup error for {away_team} @ {home_team}: {e}")
        return result

    if not market:
        return result

    ticker = market.get("ticker", "")
    yes_team = mapper.get_yes_team(ticker)
    if not yes_team:
        return result

    try:
        prices = client.get_market_prices(ticker)
    except (requests.RequestException, KeyError) as e:
        print(f"      Kalshi price fetch error for {ticker}: {e}")
        return result

    yes_price = prices.get("yes_price")
    no_price = prices.get("no_price")
    if yes_price is None:
        return result

    # Determine our side
    if pick == yes_team:
        our_price = yes_price
        our_side = "YES"
    else:
        our_price = no_price if no_price else (100 - yes_price)
        our_side = "NO"

    if our_price is None or our_price <= 0:
        return result

    # Edge calculation -- let bugs propagate here (no catch)
    model_prob = prob_home_win if pick == home_team else (1.0 - prob_home_win)
    implied_prob = kalshi_implied_prob(our_price)
    fee = kalshi_fee_cents(our_price) / 100.0

    edge = calculate_edge(model_prob, implied_prob)
    rating_enum = get_rating(edge)
    rating = rating_enum.value
    units = recommended_units(edge, implied_prob)

    result["Kalshi_Ticker"] = ticker
    result["Kalshi_Side"] = our_side
    result["Kalshi_Price"] = our_price
    result["Kalshi_Fee"] = round(fee, 4)
    result["Edge"] = round(edge, 4)
    result["Edge_Pct"] = f"{edge:.1%}"
    result["Rating"] = rating
    result["Units"] = units

    return result


def generate_predictions(league=LEAGUE):
    """Generate daily MLB predictions."""
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    output_file = paths["predictions_file"]
    model_file = paths["model_file"]

    print(f"=== MLB PREDICTION PIPELINE ===")

    # Load model
    if not os.path.exists(model_file):
        print(f"Model not found: {model_file}")
        print("Run 'python model.py --league mlb' first.")
        return []
    model, sigma = load_model(model_file, league)
    features = get_feature_list(league)
    print(f"   Model loaded ({len(features)} features, sigma={sigma:.2f})")

    # Load training data for team stats
    if not os.path.exists(data_file):
        print(f"Training data not found: {data_file}")
        return []
    df = pd.read_csv(data_file, low_memory=False)
    latest_stats = get_latest_stats(df)
    pitcher_stats = get_latest_pitcher_stats(df)
    known_teams = set(latest_stats.keys())
    print(f"   Stats loaded for {len(known_teams)} teams, {len(pitcher_stats)} pitchers")

    # Fetch Kalshi markets
    kalshi_client = None
    kalshi_mapper = None
    api_key = os.environ.get("KALSHI_API_KEY")
    if not api_key:
        print("   Kalshi: KALSHI_API_KEY not set, skipping")
    else:
        from kalshi import KalshiClient, MLBMarketMapper
        kalshi_client = KalshiClient(api_key)
        try:
            markets = kalshi_client.get_mlb_markets()
            if markets:
                kalshi_mapper = MLBMarketMapper(markets)
                print(f"   Kalshi: {len(markets)} MLB markets loaded")
            else:
                print("   Kalshi: no MLB markets found")
        except requests.RequestException as e:
            print(f"   Kalshi: API error loading markets: {e}")

    # Fetch schedule
    games = fetch_schedule(league)
    if not games:
        print("   No upcoming games found.")
        return []

    predictions = []
    for game in games:
        home_name = find_best_match(game["home_team"], known_teams)
        away_name = find_best_match(game["away_team"], known_teams)

        if home_name is None or away_name is None:
            continue

        home_stats = latest_stats.get(home_name, {})
        away_stats = latest_stats.get(away_name, {})

        row = build_feature_row(
            home_stats, away_stats,
            game["home_sp_era"], game["away_sp_era"],
            home_sp_name=game["home_sp"],
            away_sp_name=game["away_sp"],
            pitcher_stats=pitcher_stats,
            game_date=game["game_date"],
            venue_name=game.get("venue", ""),
            venue_indoor=game.get("venue_indoor", 0),
        )

        # Run inference
        X = pd.DataFrame([row])[features].astype(float)
        prob_home_win = model.predict_proba(X)[:, 1][0]
        prob_away_win = 1.0 - prob_home_win

        conf = max(prob_home_win, prob_away_win)
        pick = game["home_team"] if prob_home_win > 0.5 else game["away_team"]

        # Kalshi edge calculation -- pass game_time for doubleheader disambiguation
        game_time_utc = game["game_date"].strftime("%H:%M") if hasattr(game["game_date"], "strftime") else ""
        kalshi_data = _get_kalshi_edge(
            kalshi_client, kalshi_mapper,
            game["home_team"], game["away_team"],
            game["game_date"],
            prob_home_win, pick,
            game_time=game_time_utc,
        )

        # Standard book (FanDuel/DraftKings) edge from real ESPN moneyline odds
        is_home_pick = prob_home_win > 0.5
        ml_odds_str = game.get("home_ml_odds") if is_home_pick else game.get("away_ml_odds")
        std_implied = american_odds_to_implied_prob(ml_odds_str) if ml_odds_str else None
        if std_implied is None:
            std_implied = STANDARD_IMPLIED_PROB
        std_edge = conf - std_implied
        std_rating = get_rating(std_edge).value
        std_units = recommended_units(std_edge, std_implied) if std_edge > 0 else 0.0

        pred = {
            "Bet_Type": "game",
            "Date/Time": game["game_date"].strftime("%Y-%m-%d %H:%M"),
            "Matchup": f"{game['away_team']} @ {game['home_team']}",
            "Home_SP": game["home_sp"],
            "Away_SP": game["away_sp"],
            "Pick": pick,
            "Prob_Home": round(prob_home_win, 3),
            "Prob_Away": round(prob_away_win, 3),
            "Conf": round(conf, 3),
            "Venue": game["venue"],
            "Std_Edge": round(std_edge, 4),
            "Std_Edge_Pct": f"{std_edge * 100:+.1f}%",
            "Std_Rating": std_rating,
            "Std_Units": round(std_units, 1),
            "Std_Odds": ml_odds_str or "",
            **kalshi_data,
        }
        predictions.append(pred)

        edge_str = ""
        if kalshi_data.get("Edge") is not None:
            edge_str = f" | Edge: {kalshi_data['Edge']:.1%} ({kalshi_data['Rating']})"
        label = "HOME" if prob_home_win > 0.5 else "AWAY"
        print(f"   {game['away_abbr']} @ {game['home_abbr']}: "
              f"{label} {conf:.1%} "
              f"(SP: {game['away_sp']} vs {game['home_sp']})"
              f"{edge_str}")

    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output_file, index=False)

        # Save dated archive for next-day grading
        eastern = pytz.timezone("US/Eastern")
        archive_prefix = paths["predictions_archive_prefix"]
        archive_file = os.path.join(
            BASE_DIR,
            f"{archive_prefix}_{datetime.now(eastern).strftime('%Y%m%d')}.csv",
        )
        pred_df.to_csv(archive_file, index=False)

        print(f"\n   Saved {len(predictions)} predictions to {output_file}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="MLB daily predictions")
    parser.add_argument("--league", default=LEAGUE)
    args = parser.parse_args()
    generate_predictions(args.league)


if __name__ == "__main__":
    main()
