import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests

from league_config import (
    get_league_artifact_paths,
    get_season_start_date,
    get_league_settings,
    get_scoreboard_base_url,
    normalize_league,
)
from womens_net import sync_current_snapshot as sync_womens_net_snapshot

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENUE_METADATA_COLUMNS = ["is_neutral", "venue_city", "venue_state"]
RAW_RATE_COLUMNS = [
    "possessions",
    "team_eFG",
    "team_TO",
    "team_ORB",
    "team_FTR",
    "team_3PR",
    "opp_eFG",
    "opp_TO",
    "opp_ORB",
    "opp_FTR",
    "opp_3PR",
]
SUMMARY_FETCH_WORKERS = 8


def ensure_venue_columns(df):
    """Ensure venue-related columns exist with stable defaults."""
    if "is_neutral" not in df.columns:
        df["is_neutral"] = 0
    else:
        df["is_neutral"] = pd.to_numeric(df["is_neutral"], errors="coerce").fillna(0).astype(int)

    for col in ("venue_city", "venue_state"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)
    return df


def ensure_raw_rate_columns(df):
    """Ensure optional raw box-score rate columns exist."""
    for col in RAW_RATE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def merge_venue_metadata(existing_df, metadata_df):
    """Backfill venue metadata for existing rows using date/team matches."""
    if metadata_df.empty:
        return ensure_venue_columns(existing_df)

    existing_df = ensure_venue_columns(existing_df.copy())
    metadata_df = ensure_venue_columns(metadata_df.copy())

    meta_cols = ["date", "team"] + VENUE_METADATA_COLUMNS
    merged = existing_df.merge(
        metadata_df[meta_cols].drop_duplicates(subset=["date", "team"], keep="last"),
        on=["date", "team"],
        how="left",
        suffixes=("", "_new"),
    )

    merged["is_neutral"] = merged["is_neutral"].where(
        merged["is_neutral"] != 0,
        merged["is_neutral_new"].fillna(0),
    ).fillna(0).astype(int)

    for col in ("venue_city", "venue_state"):
        merged[col] = merged[col].where(
            merged[col].astype(str).str.strip() != "",
            merged[f"{col}_new"].fillna(""),
        )

    drop_cols = [f"{col}_new" for col in VENUE_METADATA_COLUMNS]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged


def merge_raw_rate_data(existing_df, rate_df):
    """Backfill missing raw rate fields using date/team/opponent/is_home matches."""
    if rate_df.empty:
        return ensure_raw_rate_columns(existing_df)

    existing_df = ensure_raw_rate_columns(existing_df.copy())
    rate_df = ensure_raw_rate_columns(rate_df.copy())

    merge_keys = ["date", "team", "opponent", "is_home"]
    merged = existing_df.merge(
        rate_df[merge_keys + RAW_RATE_COLUMNS].drop_duplicates(subset=merge_keys, keep="last"),
        on=merge_keys,
        how="left",
        suffixes=("", "_new"),
    )

    for col in RAW_RATE_COLUMNS:
        merged[col] = merged[col].where(
            merged[col].notna(),
            merged[f"{col}_new"],
        )

    drop_cols = [f"{col}_new" for col in RAW_RATE_COLUMNS]
    return merged.drop(columns=drop_cols, errors="ignore")


def backfill_venue_metadata(df, base_url):
    """Fetch and merge missing venue metadata for historical rows."""
    df = ensure_venue_columns(df.copy())
    missing_mask = (
        (df["is_neutral"] == 0)
        & (df["venue_city"].astype(str).str.strip() == "")
        & (df["venue_state"].astype(str).str.strip() == "")
    )
    target_dates = sorted(pd.to_datetime(df.loc[missing_mask, "date"]).dt.normalize().dropna().unique())
    if not target_dates:
        return df

    print(f"🧭 Backfilling venue metadata for {len(target_dates)} dates...")
    fetched = []
    for idx, ts in enumerate(target_dates, start=1):
        fetched.extend(fetch_games_for_date(pd.Timestamp(ts).to_pydatetime(), base_url))
        if idx % 30 == 0 or idx == len(target_dates):
            print(f"   -> Venue backfill {idx}/{len(target_dates)} dates")

    if not fetched:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    fetched_df = pd.DataFrame(fetched)
    fetched_df["date"] = pd.to_datetime(fetched_df["date"]).dt.strftime("%Y-%m-%d")
    return merge_venue_metadata(df, fetched_df)


def backfill_boxscore_rates(df, base_url):
    """Fetch and merge missing raw rate stats for historical rows."""
    df = ensure_raw_rate_columns(df.copy())
    missing_mask = df[RAW_RATE_COLUMNS].isna().any(axis=1)
    target_dates = sorted(pd.to_datetime(df.loc[missing_mask, "date"]).dt.normalize().dropna().unique())
    if not target_dates:
        return df

    print(f"Backfilling box-score rates for {len(target_dates)} dates...")
    fetched = []
    for idx, ts in enumerate(target_dates, start=1):
        fetched.extend(fetch_games_for_date(pd.Timestamp(ts).to_pydatetime(), base_url))
        if idx % 14 == 0 or idx == len(target_dates):
            print(f"   -> Box-score backfill {idx}/{len(target_dates)} dates")

    if not fetched:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    fetched_df = pd.DataFrame(fetched)
    fetched_df["date"] = pd.to_datetime(fetched_df["date"]).dt.strftime("%Y-%m-%d")
    return merge_raw_rate_data(df, fetched_df)


def get_last_recorded_date(data_file, season_start_date):
    if not os.path.exists(data_file):
        return season_start_date
    try:
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        return df['date'].max().to_pydatetime()
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"WARNING: Data file corrupted ({e}), will re-download from season start")
        return season_start_date
    except Exception as e:
        print(f"WARNING: Could not read last date from {data_file} ({type(e).__name__}: {e})")
        return season_start_date


def fetch_games_for_date(target_date, base_url):
    date_str_url = target_date.strftime("%Y%m%d")
    print(f"   -> Downloading {target_date.strftime('%Y-%m-%d')}...")
    
    url = f"{base_url}&dates={date_str_url}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        res = response.json()
    except requests.RequestException as e:
        print(f"      WARNING: Connection failed for {date_str_url}: {e}")
        return []
    except ValueError as e:
        print(f"      WARNING: Invalid JSON for {date_str_url}: {e}")
        return []

    events = [event for event in res.get('events', []) if event['status']['type']['state'] == 'post']
    rate_by_event = {}
    if events:
        with ThreadPoolExecutor(max_workers=SUMMARY_FETCH_WORKERS) as executor:
            future_map = {
                executor.submit(fetch_boxscore_rates, base_url, event.get('id')): event.get('id')
                for event in events
            }
            for future in as_completed(future_map):
                rate_by_event[future_map[future]] = future.result()

    games = []
    for event in events:
        try:
            comp = event['competitions'][0]
            home = comp['competitors'][0]
            away = comp['competitors'][1]
            boxscore_rates = rate_by_event.get(event.get('id'), {})
            
            # NORMALIZE DATE (No Time)
            game_date_str = target_date.strftime("%Y-%m-%d")

            spread_val = float("nan")
            has_spread_line = False
            try:
                if comp.get('odds'):
                    odds = comp['odds'][0]
                    details = odds.get('details', '')
                    if details == 'EVEN':
                        spread_val = 0.0
                        has_spread_line = True
                    elif details and details != '0':
                        parts = details.split()
                        val = abs(float(parts[-1]))
                        fav = " ".join(parts[:-1])
                        home_abbr = home['team'].get('abbreviation', '')
                        home_name = home['team'].get('displayName', '')
                        is_home_fav = (fav == home_abbr) or (fav == home_name) or (fav in home_name)
                        spread_val = -val if is_home_fav else val
                        has_spread_line = True
            except (ValueError, IndexError, TypeError) as e:
                event_id = event.get('id', 'unknown')
                print(f"      WARNING: Could not parse spread for event {event_id}: {type(e).__name__}: {e}")

            # Neutral site + venue info
            is_neutral = int(comp.get('neutralSite', False))
            venue = comp.get('venue', {})
            venue_addr = venue.get('address', {})
            venue_city = venue_addr.get('city', '')
            venue_state = venue_addr.get('state', '')

            g = {
                'date': game_date_str,
                'team': home['team']['displayName'],
                'opponent': away['team']['displayName'],
                'location': 'Home',
                'team_score': int(home['score']),
                'opp_score': int(away['score']),
                'is_home': 1,
                'spread': spread_val,
                'has_spread_line': has_spread_line,
                'is_neutral': is_neutral,
                'venue_city': venue_city,
                'venue_state': venue_state,
                'ats_win': 0,
                **boxscore_rates.get('home', {}),
            }
            games.append(g)

            g_away = {
                'date': game_date_str,
                'team': away['team']['displayName'],
                'opponent': home['team']['displayName'],
                'location': 'Away',
                'team_score': int(away['score']),
                'opp_score': int(home['score']),
                'is_home': 0,
                'spread': -1 * spread_val,
                'has_spread_line': has_spread_line,
                'is_neutral': is_neutral,
                'venue_city': venue_city,
                'venue_state': venue_state,
                'ats_win': 0,
                **boxscore_rates.get('away', {}),
            }
            games.append(g_away)
        except (KeyError, TypeError, ValueError, IndexError) as e:
            event_id = event.get('id', 'unknown')
            print(f"      Skipped event {event_id}: {type(e).__name__}: {e}")
            continue
        
    return games


def _parse_count_pair(value):
    """Parse a made-attempted string like '24-53'."""
    try:
        made, attempted = str(value).split("-", 1)
        return float(made), float(attempted)
    except (TypeError, ValueError, AttributeError):
        return None, None


def _stat_map(team_entry):
    return {item.get("name"): item.get("displayValue") for item in team_entry.get("statistics", [])}


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_rate_snapshot(team_stats, opp_stats):
    """Convert ESPN team box-score totals into the raw rate fields used by the model."""
    fgm, fga = _parse_count_pair(team_stats.get("fieldGoalsMade-fieldGoalsAttempted"))
    tpm, tpa = _parse_count_pair(team_stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
    _, fta = _parse_count_pair(team_stats.get("freeThrowsMade-freeThrowsAttempted"))
    opp_fgm, opp_fga = _parse_count_pair(opp_stats.get("fieldGoalsMade-fieldGoalsAttempted"))
    opp_tpm, opp_tpa = _parse_count_pair(opp_stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
    _, opp_fta = _parse_count_pair(opp_stats.get("freeThrowsMade-freeThrowsAttempted"))

    orb = _safe_float(team_stats.get("offensiveRebounds"))
    opp_orb = _safe_float(opp_stats.get("offensiveRebounds"))
    opp_drb = _safe_float(opp_stats.get("defensiveRebounds"))
    team_drb = _safe_float(team_stats.get("defensiveRebounds"))
    turnovers = _safe_float(team_stats.get("totalTurnovers") or team_stats.get("turnovers"))
    opp_turnovers = _safe_float(opp_stats.get("totalTurnovers") or opp_stats.get("turnovers"))

    snapshot = {col: pd.NA for col in RAW_RATE_COLUMNS}

    if fga and fga > 0 and fgm is not None and tpm is not None:
        snapshot["team_eFG"] = 100.0 * ((fgm + 0.5 * tpm) / fga)
        snapshot["team_FTR"] = 100.0 * ((fta or 0.0) / fga)
        snapshot["team_3PR"] = 100.0 * ((tpa or 0.0) / fga)

    if opp_fga and opp_fga > 0 and opp_fgm is not None and opp_tpm is not None:
        snapshot["opp_eFG"] = 100.0 * ((opp_fgm + 0.5 * opp_tpm) / opp_fga)
        snapshot["opp_FTR"] = 100.0 * ((opp_fta or 0.0) / opp_fga)
        snapshot["opp_3PR"] = 100.0 * ((opp_tpa or 0.0) / opp_fga)

    if (
        fga is not None and orb is not None and turnovers is not None and fta is not None
    ):
        possessions = fga - orb + turnovers + (0.475 * fta)
        if possessions > 0:
            snapshot["possessions"] = possessions
            snapshot["team_TO"] = 100.0 * (turnovers / possessions)

    if (
        opp_fga is not None and opp_orb is not None and opp_turnovers is not None and opp_fta is not None
    ):
        opp_possessions = opp_fga - opp_orb + opp_turnovers + (0.475 * opp_fta)
        if opp_possessions > 0:
            snapshot["opp_TO"] = 100.0 * (opp_turnovers / opp_possessions)

    if orb is not None and opp_drb is not None and (orb + opp_drb) > 0:
        snapshot["team_ORB"] = 100.0 * (orb / (orb + opp_drb))
    if opp_orb is not None and team_drb is not None and (opp_orb + team_drb) > 0:
        snapshot["opp_ORB"] = 100.0 * (opp_orb / (opp_orb + team_drb))

    return snapshot


def fetch_boxscore_rates(base_url, event_id):
    """Fetch ESPN summary box-score data for one event and return home/away rate snapshots."""
    if not event_id:
        return {}

    summary_url = base_url.replace(
        "scoreboard?groups=50&limit=1000",
        f"summary?event={event_id}",
    )
    try:
        response = requests.get(summary_url, timeout=10)
        response.raise_for_status()
        summary = response.json()
    except requests.RequestException as e:
        print(f"      WARNING: Failed to fetch box-score summary for event {event_id}: {e}")
        return {}
    except ValueError as e:
        print(f"      WARNING: Invalid box-score JSON for event {event_id}: {e}")
        return {}

    teams = summary.get("boxscore", {}).get("teams", [])
    if len(teams) != 2:
        print(f"      WARNING: Incomplete box-score teams for event {event_id}")
        return {}

    entries = {}
    for team_entry in teams:
        side = team_entry.get("homeAway")
        if side not in {"home", "away"}:
            continue
        entries[side] = _stat_map(team_entry)

    if {"home", "away"} - set(entries):
        print(f"      WARNING: Missing home/away box-score split for event {event_id}")
        return {}

    return {
        "home": _build_rate_snapshot(entries["home"], entries["away"]),
        "away": _build_rate_snapshot(entries["away"], entries["home"]),
    }


def update_database(league="mens"):
    league = normalize_league(league)
    settings = get_league_settings(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    base_url = get_scoreboard_base_url(league)
    season_start_date = pd.Timestamp(get_season_start_date(league)).to_pydatetime()

    print(f"--- AUTO-HEALING UPDATER ({settings['label']}) ---")
    
    # 1. Determine Range
    last_date = get_last_recorded_date(data_file, season_start_date)
    # Start from the day AFTER the last record
    start_date = last_date + timedelta(days=1)
    
    # End Yesterday (Strictly ignore Today to avoid partial games)
    end_date = datetime.now() - timedelta(days=1)
    
    # If the database is somehow ahead of reality (timezones), cap it
    if start_date.date() > end_date.date():
        print(f"Data is up to date. Last recorded date: {last_date.date()}")
        if os.path.exists(data_file):
            df_current = pd.read_csv(data_file)
            df_backfilled = backfill_boxscore_rates(df_current, base_url)
            df_backfilled = backfill_venue_metadata(df_backfilled, base_url)
            if not df_backfilled.equals(df_current):
                df_backfilled.to_csv(data_file, index=False)
                print("Box-score and venue metadata backfilled.")
        run_pipeline(league)
        return

    print(f"Filling gap: {start_date.date()} to {end_date.date()}")
    
    new_games = []
    current_date = start_date
    while current_date.date() <= end_date.date():
        daily_games = fetch_games_for_date(current_date, base_url)
        new_games.extend(daily_games)
        current_date += timedelta(days=1)
        
    if new_games:
        print(f"Saving {len(new_games)} new games...")
        
        if os.path.exists(data_file):
            df_old = pd.read_csv(data_file)
            df_old = ensure_venue_columns(df_old)
            df_old = ensure_raw_rate_columns(df_old)
            df_new = pd.DataFrame(new_games)
            df_new = ensure_venue_columns(df_new)
            df_new = ensure_raw_rate_columns(df_new)
            
            # Combine
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            
            # NORMALIZE AND DEDUPLICATE
            df_combined['date'] = pd.to_datetime(df_combined['date']).dt.strftime('%Y-%m-%d')
            df_combined = df_combined.drop_duplicates(subset=['date', 'team'], keep='last')
            df_combined = df_combined.sort_values('date')
            
            df_combined.to_csv(data_file, index=False)
            print("Database updated.")
        else:
            pd.DataFrame(new_games).pipe(ensure_venue_columns).to_csv(data_file, index=False)

    if os.path.exists(data_file):
        df_current = pd.read_csv(data_file)
        df_backfilled = backfill_boxscore_rates(df_current, base_url)
        df_backfilled = backfill_venue_metadata(df_backfilled, base_url)
        if not df_backfilled.equals(df_current):
            df_backfilled.to_csv(data_file, index=False)
            print("Box-score and venue metadata backfilled.")

    run_pipeline(league)


def run_pipeline(league):
    print("\n--- TRIGGERING PIPELINE ---")
    if normalize_league(league) == "womens":
        try:
            report = sync_womens_net_snapshot(league="womens")
            print(
                "0. Synced women's NET snapshots "
                f"(team map: {report['team_map_coverage']:.1%}, "
                f"source teams: {report['source_team_coverage']:.1%})"
            )
        except (FileNotFoundError, requests.RequestException, ValueError) as e:
            print(f"0. WARNING: women's NET sync skipped ({e})")

    print("1. Calculating efficiency stats...")
    subprocess.run([sys.executable, "features.py", "--league", league], check=True)

    print("2. Grading history...")
    subprocess.run([sys.executable, "backtest.py", "--league", league], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update CBB data and run feature/backtest pipeline.")
    parser.add_argument(
        "--league",
        default="mens",
        help="League to update: mens or womens (aliases: men/women/cbb/wcbb).",
    )
    args = parser.parse_args()
    update_database(args.league)
