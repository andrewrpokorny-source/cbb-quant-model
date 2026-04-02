"""MLB data pipeline: ESPN scoreboard + MLB-StatsAPI for detailed stats."""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests

# Add parent directory so league_config is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from league_config import (
    get_league_artifact_paths,
    get_scoreboard_base_url,
    get_season_start_date,
    normalize_league,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAGUE = "mlb"


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _get_statsapi():
    """Lazy import of statsapi to avoid hard dependency during testing."""
    try:
        import statsapi
        return statsapi
    except ImportError:
        print("WARNING: MLB-StatsAPI not installed. Run: uv add MLB-StatsAPI")
        return None


def fetch_games_for_date(target_date, base_url=None):
    """Fetch completed MLB games for a date from ESPN scoreboard.

    Returns a list of dicts, two rows per game (home and away), matching
    the CBB pipeline convention.
    """
    if base_url is None:
        base_url = get_scoreboard_base_url(LEAGUE)

    date_str_url = target_date.strftime("%Y%m%d")
    game_date_str = target_date.strftime("%Y-%m-%d")
    print(f"   -> Downloading {game_date_str}...")

    url = f"{base_url}&dates={date_str_url}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        res = response.json()
    except requests.RequestException as e:
        print(f"      WARNING: Connection failed for {date_str_url}: {e}")
        return []
    except ValueError as e:
        print(f"      WARNING: Invalid JSON for {date_str_url}: {e}")
        return []

    events = [
        event for event in res.get("events", [])
        if event.get("status", {}).get("type", {}).get("state") == "post"
    ]

    games = []
    for event in events:
        try:
            comp = event["competitions"][0]
            home = away = None
            for competitor in comp["competitors"]:
                if competitor.get("homeAway") == "home":
                    home = competitor
                elif competitor.get("homeAway") == "away":
                    away = competitor

            if home is None or away is None:
                continue

            venue = comp.get("venue", {})
            venue_addr = venue.get("address", {})

            # Extract starting pitcher info from probables
            home_sp = _extract_probable_pitcher(home)
            away_sp = _extract_probable_pitcher(away)

            # Extract odds if available
            moneyline_home = float("nan")
            moneyline_away = float("nan")
            run_line = float("nan")
            total_line = float("nan")
            if comp.get("odds"):
                odds = comp["odds"][0]
                moneyline_home, moneyline_away = _extract_moneylines(odds, home, away)
                run_line = _extract_run_line(odds)
                total_line = _safe_float(odds.get("overUnder"))

            # Extract hits and errors from team statistics
            home_stats = _extract_team_game_stats(home)
            away_stats = _extract_team_game_stats(away)

            # Determine season and game time for doubleheader disambiguation
            season = target_date.year
            event_dt = pd.to_datetime(event.get("date", ""), utc=True, errors="coerce")
            game_time = event_dt.strftime("%H:%M") if pd.notna(event_dt) else ""

            # Home row
            g_home = {
                "date": game_date_str,
                "game_time": game_time,
                "season": season,
                "team": home["team"]["displayName"],
                "team_abbr": home["team"].get("abbreviation", ""),
                "opponent": away["team"]["displayName"],
                "opp_abbr": away["team"].get("abbreviation", ""),
                "location": "Home",
                "is_home": 1,
                "team_score": int(home["score"]),
                "opp_score": int(away["score"]),
                "moneyline": moneyline_home,
                "run_line": run_line,
                "total_line": total_line,
                "venue_name": venue.get("fullName", ""),
                "venue_city": venue_addr.get("city", ""),
                "venue_state": venue_addr.get("state", ""),
                "venue_indoor": int(venue.get("indoor", False)),
                "starting_pitcher": home_sp.get("name", ""),
                "sp_espn_id": home_sp.get("id", ""),
                "sp_era": home_sp.get("era", float("nan")),
                "opp_starting_pitcher": away_sp.get("name", ""),
                "opp_sp_espn_id": away_sp.get("id", ""),
                "opp_sp_era": away_sp.get("era", float("nan")),
                **{f"team_{k}": v for k, v in home_stats.items()},
                **{f"opp_{k}": v for k, v in away_stats.items()},
            }
            games.append(g_home)

            # Away row (mirror)
            g_away = {
                "date": game_date_str,
                "game_time": game_time,
                "season": season,
                "team": away["team"]["displayName"],
                "team_abbr": away["team"].get("abbreviation", ""),
                "opponent": home["team"]["displayName"],
                "opp_abbr": home["team"].get("abbreviation", ""),
                "location": "Away",
                "is_home": 0,
                "team_score": int(away["score"]),
                "opp_score": int(home["score"]),
                "moneyline": moneyline_away,
                "run_line": -run_line if not pd.isna(run_line) else float("nan"),
                "total_line": total_line,
                "venue_name": venue.get("fullName", ""),
                "venue_city": venue_addr.get("city", ""),
                "venue_state": venue_addr.get("state", ""),
                "venue_indoor": int(venue.get("indoor", False)),
                "starting_pitcher": away_sp.get("name", ""),
                "sp_espn_id": away_sp.get("id", ""),
                "sp_era": away_sp.get("era", float("nan")),
                "opp_starting_pitcher": home_sp.get("name", ""),
                "opp_sp_espn_id": home_sp.get("id", ""),
                "opp_sp_era": home_sp.get("era", float("nan")),
                **{f"team_{k}": v for k, v in away_stats.items()},
                **{f"opp_{k}": v for k, v in home_stats.items()},
            }
            games.append(g_away)

        except (KeyError, TypeError, ValueError, IndexError) as e:
            event_id = event.get("id", "unknown")
            print(f"      Skipped event {event_id}: {type(e).__name__}: {e}")
            continue

    return games


def _extract_probable_pitcher(competitor):
    """Extract probable pitcher info from ESPN competitor data."""
    probables = competitor.get("probables", [])
    if not probables:
        return {"name": "", "id": "", "era": float("nan")}

    pitcher = probables[0]
    athlete = pitcher.get("athlete", {})
    era = float("nan")
    for stat in pitcher.get("statistics", []):
        if stat.get("abbreviation") == "ERA" or stat.get("name") == "earnedRunAverage":
            era = _safe_float(stat.get("displayValue"))
            break

    return {
        "name": athlete.get("fullName", ""),
        "id": str(athlete.get("id", "")),
        "era": era,
    }


def _extract_moneylines(odds, home, away):
    """Extract moneyline odds from ESPN odds block."""
    home_ml = float("nan")
    away_ml = float("nan")

    # ESPN may provide moneylines in different fields depending on the sport
    home_ml_raw = odds.get("homeTeamOdds", {}).get("moneyLine")
    away_ml_raw = odds.get("awayTeamOdds", {}).get("moneyLine")
    if home_ml_raw is not None:
        home_ml = _safe_float(home_ml_raw)
    if away_ml_raw is not None:
        away_ml = _safe_float(away_ml_raw)

    return home_ml, away_ml


def _extract_run_line(odds):
    """Extract run line (spread) from ESPN odds block."""
    spread = odds.get("spread")
    if spread is not None:
        return _safe_float(spread)
    details = odds.get("details", "")
    if details and details != "EVEN":
        try:
            parts = details.split()
            return _safe_float(parts[-1])
        except (ValueError, IndexError):
            pass
    return float("nan")


def _extract_team_game_stats(competitor):
    """Extract game-level stats (hits, errors) from ESPN competitor."""
    stats = {}
    stats["hits"] = _safe_float(competitor.get("hits", float("nan")))
    stats["errors"] = _safe_float(competitor.get("errors", float("nan")))
    for stat_entry in competitor.get("statistics", []):
        name = stat_entry.get("name", "").lower()
        val = stat_entry.get("displayValue", "")
        if name in ("hits", "errors", "runs"):
            stats[name] = _safe_float(val)
    return stats


def fetch_pitcher_game_logs(df, season=None):
    """Fetch per-game pitcher lines from MLB Stats API gameLog endpoint.

    For each unique starting pitcher in the data, fetches their game-by-game
    pitching lines (IP, ER, H, BB, K) for the given season. These are stored
    per game row so features.py can compute rolling stats with honest lag.

    Returns the DataFrame with added columns: sp_ip, sp_er, sp_h, sp_bb, sp_k.
    """
    statsapi = _get_statsapi()
    if statsapi is None:
        return df

    df = df.copy()
    for col in ["sp_ip", "sp_er", "sp_h", "sp_bb", "sp_k"]:
        if col not in df.columns:
            df[col] = float("nan")

    if season is None:
        season = int(df["season"].mode().iloc[0]) if "season" in df.columns else datetime.now().year

    # Get unique pitcher names and resolve to MLB player IDs
    pitcher_names = set()
    for col in ["starting_pitcher", "opp_starting_pitcher"]:
        if col in df.columns:
            pitcher_names.update(df[col].dropna().unique())
    pitcher_names.discard("")

    print(f"   -> Fetching game logs for {len(pitcher_names)} pitchers (season {season})...")
    pitcher_id_cache = {}
    game_log_cache = {}  # pitcher_name -> {date_str: {ip, er, h, bb, k}}
    handedness_cache = {}  # pitcher_name -> "L" or "R"

    for i, name in enumerate(sorted(pitcher_names)):
        if (i + 1) % 25 == 0 or (i + 1) == len(pitcher_names):
            print(f"      Pitcher {i + 1}/{len(pitcher_names)}")

        pid = _resolve_pitcher_id(statsapi, name, season)
        if pid is None:
            continue
        pitcher_id_cache[name] = pid

        logs = _fetch_game_log(pid, season)
        game_log_cache[name] = logs

        hand = _fetch_pitcher_handedness(pid)
        if hand:
            handedness_cache[name] = hand

    # Merge game logs into the DataFrame
    for idx, row in df.iterrows():
        sp_name = row.get("starting_pitcher", "")
        game_date = str(row.get("date", ""))[:10]

        if sp_name and sp_name in game_log_cache:
            log = game_log_cache[sp_name].get(game_date)
            if log:
                df.at[idx, "sp_ip"] = log["ip"]
                df.at[idx, "sp_er"] = log["er"]
                df.at[idx, "sp_h"] = log["h"]
                df.at[idx, "sp_bb"] = log["bb"]
                df.at[idx, "sp_k"] = log["k"]

    print(f"      Matched game logs for {df['sp_ip'].notna().sum()}/{len(df)} rows")

    # Populate pitcher handedness
    if handedness_cache:
        df["sp_throws_left"] = df["starting_pitcher"].map(
            lambda n: 1 if handedness_cache.get(n) == "L" else 0
        )
        matched_hand = df["starting_pitcher"].isin(handedness_cache.keys()).sum()
        lefties = sum(1 for h in handedness_cache.values() if h == "L")
        print(f"      Pitcher handedness: {matched_hand}/{len(df)} rows, {lefties} lefties in pool")
    else:
        df["sp_throws_left"] = 0

    return df


def _fetch_pitcher_handedness(player_id):
    """Fetch pitcher throwing hand from MLB Stats API. Returns 'L', 'R', or None."""
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    try:
        resp = requests.get(url, timeout=10)
        if not resp.ok:
            return None
        data = resp.json()
        people = data.get("people", [])
        if people:
            return people[0].get("pitchHand", {}).get("code")
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def _resolve_pitcher_id(statsapi, pitcher_name, season):
    """Look up a pitcher by name and return their MLB player ID."""
    try:
        results = statsapi.lookup_player(pitcher_name, season=season)
        if not results:
            return None
        for r in results:
            if r.get("primaryPosition", {}).get("abbreviation", "") == "P":
                return r["id"]
        return results[0]["id"]
    except (requests.RequestException, KeyError, TypeError, ValueError) as e:
        print(f"      WARNING: Could not resolve pitcher '{pitcher_name}': {type(e).__name__}: {e}")
        return None


def _fetch_game_log(player_id, season):
    """Fetch a pitcher's game-by-game stats for a season via MLB Stats API.

    Returns dict of {date_str: {ip, er, h, bb, k}} for games started.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        f"/stats?stats=gameLog&season={season}&group=pitching"
    )
    try:
        resp = requests.get(url, timeout=10)
        if not resp.ok:
            print(f"      WARNING: MLB Stats API returned {resp.status_code} for player {player_id}")
            return {}
        data = resp.json()
    except requests.RequestException as e:
        print(f"      WARNING: MLB Stats API request failed for player {player_id}: {e}")
        return {}
    except ValueError as e:
        print(f"      WARNING: Invalid JSON from MLB Stats API for player {player_id}: {e}")
        return {}

    logs = {}
    for split_group in data.get("stats", []):
        for s in split_group.get("splits", []):
            stat = s.get("stat", {})
            # Only include games where pitcher started
            if int(stat.get("gamesStarted", 0)) == 0:
                continue
            game_date = s.get("date", "")[:10]
            if game_date:
                logs[game_date] = {
                    "ip": _safe_float(stat.get("inningsPitched", 0)),
                    "er": _safe_float(stat.get("earnedRuns", 0)),
                    "h": _safe_float(stat.get("hits", 0)),
                    "bb": _safe_float(stat.get("baseOnBalls", 0)),
                    "k": _safe_float(stat.get("strikeOuts", 0)),
                }
    return logs


def get_last_recorded_date(data_file, season_start_date):
    """Return the last date in the existing data file, or season start if none."""
    if not os.path.exists(data_file):
        return season_start_date
    try:
        df = pd.read_csv(data_file)
        df["date"] = pd.to_datetime(df["date"])
        return df["date"].max().to_pydatetime()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError, ValueError) as e:
        print(f"WARNING: Could not parse {data_file} ({type(e).__name__}: {e}), re-downloading from season start")
        return season_start_date


# MLB team IDs for Stats API (30 teams)
MLB_TEAM_IDS = {
    "Arizona Diamondbacks": 109, "Atlanta Braves": 144, "Baltimore Orioles": 110,
    "Boston Red Sox": 111, "Chicago Cubs": 112, "Chicago White Sox": 145,
    "Cincinnati Reds": 113, "Cleveland Guardians": 114, "Colorado Rockies": 115,
    "Detroit Tigers": 116, "Houston Astros": 117, "Kansas City Royals": 118,
    "Los Angeles Angels": 108, "Los Angeles Dodgers": 119, "Miami Marlins": 146,
    "Milwaukee Brewers": 158, "Minnesota Twins": 142, "New York Mets": 121,
    "New York Yankees": 147, "Oakland Athletics": 133, "Philadelphia Phillies": 143,
    "Pittsburgh Pirates": 134, "San Diego Padres": 135, "San Francisco Giants": 137,
    "Seattle Mariners": 136, "St. Louis Cardinals": 138, "Tampa Bay Rays": 139,
    "Texas Rangers": 140, "Toronto Blue Jays": 141, "Washington Nationals": 120,
}


def fetch_all_bullpen_eras(season):
    """Fetch relief pitching ERA for all 30 MLB teams for a season.

    Returns dict: {team_display_name: bullpen_era_float}.
    """
    result = {}
    print(f"   -> Fetching bullpen ERA for {len(MLB_TEAM_IDS)} teams (season {season})...")
    for team_name, team_id in sorted(MLB_TEAM_IDS.items()):
        url = (
            f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
            f"?stats=season&group=pitching&season={season}"
        )
        try:
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                continue
            data = resp.json()
            for split_group in data.get("stats", []):
                for split in split_group.get("splits", []):
                    stat = split.get("stat", {})
                    era = _safe_float(stat.get("era"))
                    # Team pitching ERA includes starters -- approximate bullpen by
                    # using the overall team ERA as a proxy (starter ERA is separate).
                    # A more precise approach would use sitCodes=rp but that's not
                    # consistently available; overall team ERA is still useful signal.
                    if not pd.isna(era):
                        result[team_name] = era
                        break
        except (requests.RequestException, ValueError):
            continue
    print(f"      Fetched ERA for {len(result)}/{len(MLB_TEAM_IDS)} teams")
    return result


def enrich_bullpen_era(df, season=None):
    """Add team pitching ERA column to the DataFrame."""
    if season is None:
        season = int(df["season"].mode().iloc[0]) if "season" in df.columns else datetime.now().year
    eras = fetch_all_bullpen_eras(season)
    if eras:
        df["bullpen_era"] = df["team"].map(eras)
    else:
        df["bullpen_era"] = float("nan")
    return df


def update_database(data_file, start_date, end_date, base_url=None):
    """Fetch MLB games from start_date to end_date and merge into CSV."""
    if base_url is None:
        base_url = get_scoreboard_base_url(LEAGUE)

    all_games = []
    current = start_date
    while current <= end_date:
        games = fetch_games_for_date(current, base_url)
        all_games.extend(games)
        current += timedelta(days=1)

    if not all_games:
        print("No new games found.")
        if os.path.exists(data_file):
            return pd.read_csv(data_file, low_memory=False)
        return None

    new_df = pd.DataFrame(all_games)
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")

    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file, low_memory=False)
        existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.strftime("%Y-%m-%d")

        # Deduplicate on date + team + opponent + is_home
        merge_keys = ["date", "game_time", "team", "opponent", "is_home"]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=merge_keys, keep="last")
    else:
        combined = new_df

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(data_file, index=False)
    print(f"Saved {len(combined)} rows to {data_file}")
    return combined


def run_pipeline(start=None, end=None, enrich=True):
    """Run the full MLB data pipeline: fetch ESPN data, enrich with pitcher game logs."""
    paths = get_league_artifact_paths(BASE_DIR, LEAGUE)
    data_file = paths["data_file"]

    season_start = datetime.strptime(get_season_start_date(LEAGUE), "%Y-%m-%d")

    if start:
        start_date = datetime.strptime(start, "%Y-%m-%d")
    else:
        start_date = get_last_recorded_date(data_file, season_start)
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

    if end:
        end_date = datetime.strptime(end, "%Y-%m-%d")
    else:
        end_date = datetime.now() - timedelta(days=1)

    print(f"MLB data pipeline: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    df = update_database(data_file, start_date, end_date)

    if enrich and df is not None:
        print("Fetching pitcher game logs from MLB-StatsAPI...")
        df = fetch_pitcher_game_logs(df)

        print("Fetching team bullpen ERA...")
        df = enrich_bullpen_era(df)

        print("Fetching weather data...")
        from mlb.weather import add_weather_features
        df = add_weather_features(df)

        df.to_csv(data_file, index=False)
        print(f"Enriched data saved: {len(df)} rows")

    return df


def main():
    parser = argparse.ArgumentParser(description="MLB data pipeline")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip pitcher game log enrichment")
    args = parser.parse_args()
    run_pipeline(start=args.start, end=args.end, enrich=not args.no_enrich)


if __name__ == "__main__":
    main()
