"""MLB feature engineering with honest lag."""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from league_config import get_league_artifact_paths, normalize_league

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAGUE = "mlb"

# Columns expected in the raw data from mlb/data.py
BASE_COLUMNS = [
    "date", "game_time", "season", "team", "team_abbr", "opponent", "opp_abbr",
    "location", "is_home", "team_score", "opp_score",
    "venue_name", "venue_city", "venue_state", "venue_indoor",
    "starting_pitcher", "sp_espn_id", "sp_era",
    "opp_starting_pitcher", "opp_sp_espn_id", "opp_sp_era",
    "moneyline", "run_line", "total_line",
]

# Per-game pitcher line columns from fetch_pitcher_game_logs()
PITCHER_GAME_LOG_COLUMNS = [
    "sp_ip", "sp_er", "sp_h", "sp_bb", "sp_k",
    "sp_throws_left",
    "bullpen_era",
    "temperature", "wind_speed",
]

# Game-level stat columns from ESPN
GAME_STAT_COLUMNS = [
    "team_hits", "team_errors", "team_runs",
    "opp_hits", "opp_errors", "opp_runs",
]


def compute_pythagorean_wpct(rs, ra, exponent=1.83):
    """Pythagorean expected win% from runs scored and allowed per game.

    Returns 0.5 when both are zero.
    """
    if rs == 0 and ra == 0:
        return 0.5
    rs_pow = rs ** exponent
    ra_pow = ra ** exponent
    denom = rs_pow + ra_pow
    if denom == 0:
        return 0.5
    return rs_pow / denom


def clean_stale_data(df):
    """Drop derived columns so they can be recomputed fresh."""
    print("   -> Cleaning stale columns...")
    keep_cols = set(BASE_COLUMNS + PITCHER_GAME_LOG_COLUMNS + GAME_STAT_COLUMNS)
    drop_list = [col for col in df.columns if col not in keep_cols]
    if drop_list:
        df = df.drop(columns=drop_list)
    return df


def calculate_rolling_stats(df):
    """Compute rolling and season stats with honest lag (.shift(1))."""
    print("   -> Generating rolling averages (honest lag)...")
    # Sort by game_time too so doubleheader games are ordered correctly
    sort_cols = ["team", "date"]
    if "game_time" in df.columns:
        sort_cols.append("game_time")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Ensure numeric types
    for col in ["team_score", "opp_score", "team_hits", "opp_hits",
                 "team_errors", "opp_errors"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Rest days ---
    df["prev_game_date"] = df.groupby("team")["date"].shift(1)
    rest = (pd.to_datetime(df["date"]) - pd.to_datetime(df["prev_game_date"])).dt.days
    df["rest_days"] = rest.fillna(3).clip(lower=0, upper=7)
    df = df.drop(columns=["prev_game_date"])

    # --- Run margin ---
    df["margin"] = df["team_score"] - df["opp_score"]

    # --- Runs per game ---
    df["runs_per_game"] = df["team_score"]
    df["runs_allowed"] = df["opp_score"]

    # --- Hits per game ---
    if "team_hits" in df.columns:
        df["hits_per_game"] = pd.to_numeric(df["team_hits"], errors="coerce")

    # --- Rolling windows: 5-game, 10-game, and season ---
    rolling_cols = ["runs_per_game", "runs_allowed", "margin"]
    if "hits_per_game" in df.columns:
        rolling_cols.append("hits_per_game")

    for col in rolling_cols:
        grp = df.groupby("team")[col]
        df[f"season_{col}"] = grp.expanding().mean().reset_index(level=0, drop=True)
        df[f"roll5_{col}"] = grp.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"roll10_{col}"] = grp.rolling(10, min_periods=3).mean().reset_index(level=0, drop=True)

    # --- Pythagorean win% from run averages ---
    rs_season = df["season_runs_per_game"]
    ra_season = df["season_runs_allowed"]
    rs_pow = rs_season ** 1.83
    ra_pow = ra_season ** 1.83
    denom = rs_pow + ra_pow
    df["season_pyth_wpct"] = (rs_pow / denom).where(denom > 0, 0.5)

    rs_r10 = df["roll10_runs_per_game"]
    ra_r10 = df["roll10_runs_allowed"]
    rs_r10_pow = rs_r10 ** 1.83
    ra_r10_pow = ra_r10 ** 1.83
    denom_r10 = rs_r10_pow + ra_r10_pow
    df["roll10_pyth_wpct"] = (rs_r10_pow / denom_r10).where(denom_r10 > 0, 0.5)

    # Apply honest lag: shift all rolling/season stats by 1 within each team
    lag_prefixes = ["season_", "roll5_", "roll10_"]
    for col in list(df.columns):
        if any(col.startswith(p) for p in lag_prefixes):
            df[f"prev_{col}"] = df.groupby("team")[col].shift(1)

    # --- Games played (sample size) ---
    df["games_played"] = df.groupby("team").cumcount()
    df["prev_games_played"] = df.groupby("team")["games_played"].shift(1).fillna(0)

    # --- Win tracking ---
    df["game_win"] = (df["team_score"] > df["opp_score"]).astype(int)
    df["season_wins"] = df.groupby("team")["game_win"].cumsum()
    df["win_pct"] = df["season_wins"] / (df["games_played"] + 1).clip(lower=1)
    df["prev_win_pct"] = df.groupby("team")["win_pct"].shift(1).fillna(0.5)

    df["roll10_win_pct"] = (
        df.groupby("team")["game_win"]
        .rolling(10, min_periods=3).mean()
        .reset_index(level=0, drop=True)
    )
    df["prev_roll10_win_pct"] = df.groupby("team")["roll10_win_pct"].shift(1).fillna(0.5)

    # --- Score volatility ---
    df["roll10_score_std"] = (
        df.groupby("team")["team_score"]
        .rolling(10, min_periods=3).std()
        .reset_index(level=0, drop=True)
    )
    df["prev_volatility"] = df.groupby("team")["roll10_score_std"].shift(1).fillna(2.0)

    return df


def calculate_pitcher_rolling_stats(df):
    """Compute rolling pitcher stats from game-level lines with honest lag.

    Groups by starting_pitcher and computes rolling ERA, WHIP, K/9, IP/start
    from the per-game columns (sp_ip, sp_er, sp_h, sp_bb, sp_k) stored by
    fetch_pitcher_game_logs(). Applies .shift(1) so each game only sees stats
    from that pitcher's previous starts.
    """
    print("   -> Computing pitcher rolling stats (honest lag)...")

    # Ensure numeric
    for col in PITCHER_GAME_LOG_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sp_ip" not in df.columns or df["sp_ip"].notna().sum() == 0:
        print("      No pitcher game log data found; skipping pitcher rolling stats.")
        for col in ["sp_roll_era", "sp_roll_whip", "sp_roll_k9", "sp_roll_ip"]:
            df[col] = float("nan")
        return df

    sort_cols = ["starting_pitcher", "date"]
    if "game_time" in df.columns:
        sort_cols.append("game_time")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Cumulative sums per pitcher for computing rolling rates
    grp = df.groupby("starting_pitcher")

    # 5-start rolling sums (min_periods=1 so early starts still get values)
    df["_roll_ip"] = grp["sp_ip"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["_roll_er"] = grp["sp_er"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["_roll_h"] = grp["sp_h"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["_roll_bb"] = grp["sp_bb"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["_roll_k"] = grp["sp_k"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["_roll_starts"] = grp["sp_ip"].rolling(5, min_periods=1).count().reset_index(level=0, drop=True)

    # Compute rate stats from rolling sums
    safe_ip = df["_roll_ip"].where(df["_roll_ip"] > 0)
    df["_sp_roll_era"] = 9.0 * df["_roll_er"] / safe_ip
    df["_sp_roll_whip"] = (df["_roll_h"] + df["_roll_bb"]) / safe_ip
    df["_sp_roll_k9"] = 9.0 * df["_roll_k"] / safe_ip
    df["_sp_roll_ip"] = df["_roll_ip"] / df["_roll_starts"].where(df["_roll_starts"] > 0)

    # Honest lag: shift by 1 within each pitcher so we only see prior starts
    df["sp_roll_era"] = grp["_sp_roll_era"].shift(1)
    df["sp_roll_whip"] = grp["_sp_roll_whip"].shift(1)
    df["sp_roll_k9"] = grp["_sp_roll_k9"].shift(1)
    df["sp_roll_ip"] = grp["_sp_roll_ip"].shift(1)

    # Clean up temp columns
    temp_cols = [c for c in df.columns if c.startswith("_roll_") or c.startswith("_sp_roll_")]
    df = df.drop(columns=temp_cols)

    # Re-sort by team+date+time for the rest of the pipeline
    resort_cols = ["team", "date"]
    if "game_time" in df.columns:
        resort_cols.append("game_time")
    df = df.sort_values(resort_cols).reset_index(drop=True)

    matched = df["sp_roll_era"].notna().sum()
    print(f"      Pitcher rolling stats computed: {matched}/{len(df)} rows with prior start data")

    return df


def merge_opponent_stats(df):
    """Merge opponent's entering stats into each row."""
    print("   -> Merging opponent entering stats...")

    opp_cols = {
        "prev_win_pct": "opp_win_pct",
        "prev_roll10_runs_per_game": "opp_prev_roll10_rpg",
        "prev_roll10_runs_allowed": "opp_prev_roll10_ra",
        "prev_season_runs_per_game": "opp_prev_season_rpg",
        "prev_season_runs_allowed": "opp_prev_season_ra",
        "prev_roll10_win_pct": "opp_prev_roll10_win_pct",
        "prev_roll5_runs_per_game": "opp_prev_roll5_rpg",
        "prev_roll5_runs_allowed": "opp_prev_roll5_ra",
        "prev_season_pyth_wpct": "opp_prev_season_pyth_wpct",
        "bullpen_era": "opp_bullpen_era",
    }

    # Use game_time in the join key when available to handle doubleheaders
    has_game_time = "game_time" in df.columns
    join_keys_left = ["date", "opponent"]
    join_keys_right = ["date", "opponent_name"]
    dedup_keys = ["date", "opponent_name"]
    if has_game_time:
        join_keys_left.append("game_time")
        join_keys_right.append("game_time")
        dedup_keys.append("game_time")

    req_cols = ["date", "team"]
    if has_game_time:
        req_cols.append("game_time")
    rename_map = {"team": "opponent_name"}
    for src, dest in opp_cols.items():
        if src in df.columns:
            req_cols.append(src)
            rename_map[src] = dest

    opp_lookup = df[req_cols].copy().rename(columns=rename_map)
    opp_lookup = opp_lookup.drop_duplicates(subset=dedup_keys, keep="last")

    df = pd.merge(
        df, opp_lookup,
        left_on=join_keys_left,
        right_on=join_keys_right,
        how="left",
        suffixes=("", "_dupe"),
    )

    if "opponent_name" in df.columns:
        df = df.drop(columns=["opponent_name"])

    df["opp_win_pct"] = df.get("opp_win_pct", pd.Series(0.5, index=df.index)).fillna(0.5)

    # Also merge opponent starting pitcher rolling stats
    sp_cols = {
        "sp_roll_era": "opp_sp_roll_era",
        "sp_roll_whip": "opp_sp_roll_whip",
        "sp_roll_k9": "opp_sp_roll_k9",
        "sp_roll_ip": "opp_sp_roll_ip",
        "sp_throws_left": "opp_sp_throws_left",
    }
    opp_sp_req = ["date", "team"]
    if has_game_time:
        opp_sp_req.append("game_time")
    opp_sp_rename = {"team": "opp_sp_name"}
    for src, dest in sp_cols.items():
        if src in df.columns:
            opp_sp_req.append(src)
            opp_sp_rename[src] = dest

    min_cols = 3 if has_game_time else 2
    if len(opp_sp_req) > min_cols:
        opp_sp_lookup = df[opp_sp_req].copy().rename(columns=opp_sp_rename)
        sp_dedup_keys = ["date", "opp_sp_name"]
        if has_game_time:
            sp_dedup_keys.append("game_time")
        opp_sp_lookup = opp_sp_lookup.drop_duplicates(subset=sp_dedup_keys, keep="last")
        sp_left = ["date", "opponent"]
        sp_right = ["date", "opp_sp_name"]
        if has_game_time:
            sp_left.append("game_time")
            sp_right.append("game_time")
        df = pd.merge(
            df, opp_sp_lookup,
            left_on=sp_left,
            right_on=sp_right,
            how="left",
            suffixes=("", "_dupe2"),
        )
        if "opp_sp_name" in df.columns:
            df = df.drop(columns=["opp_sp_name"])

    return df


def add_park_factor(df):
    """Add ballpark run factor from venue_name."""
    from mlb.ballpark_factors import get_park_factor
    print("   -> Adding ballpark factors...")
    if "venue_name" in df.columns:
        df["park_factor"] = df["venue_name"].map(get_park_factor).fillna(1.0)
    else:
        df["park_factor"] = 1.0
    return df


def compute_differentials(df):
    """Compute differential features between team and opponent."""
    print("   -> Computing differentials...")

    # Rolling runs scored differential (recent form)
    if "prev_roll10_runs_per_game" in df.columns and "opp_prev_roll10_rpg" in df.columns:
        df["roll10_rpg_diff"] = (
            df["prev_roll10_runs_per_game"].fillna(0)
            - df["opp_prev_roll10_rpg"].fillna(0)
        )
    else:
        df["roll10_rpg_diff"] = 0.0

    # Rolling runs allowed differential (pitching quality gap)
    if "prev_roll10_runs_allowed" in df.columns and "opp_prev_roll10_ra" in df.columns:
        df["roll10_ra_diff"] = (
            df["opp_prev_roll10_ra"].fillna(0)
            - df["prev_roll10_runs_allowed"].fillna(0)
        )
    else:
        df["roll10_ra_diff"] = 0.0

    # Starting pitcher ERA differential (from ESPN point-in-time ERA)
    if "sp_era" in df.columns and "opp_sp_era" in df.columns:
        df["sp_era_diff"] = (
            pd.to_numeric(df["opp_sp_era"], errors="coerce")
            - pd.to_numeric(df["sp_era"], errors="coerce")
        )
    else:
        df["sp_era_diff"] = 0.0

    # Starting pitcher rolling ERA differential (from game-log derived stats)
    if "sp_roll_era" in df.columns and "opp_sp_roll_era" in df.columns:
        df["sp_roll_era_diff"] = (
            df["opp_sp_roll_era"].fillna(4.5)
            - df["sp_roll_era"].fillna(4.5)
        )
    else:
        df["sp_roll_era_diff"] = 0.0

    # Pythagorean win% differential
    if "prev_season_pyth_wpct" in df.columns and "opp_prev_season_pyth_wpct" in df.columns:
        df["pyth_wpct_diff"] = (
            df["prev_season_pyth_wpct"].fillna(0.5)
            - df["opp_prev_season_pyth_wpct"].fillna(0.5)
        )
    else:
        df["pyth_wpct_diff"] = 0.0

    # Roll5 differentials (short-term form)
    if "prev_roll5_runs_per_game" in df.columns and "opp_prev_roll5_rpg" in df.columns:
        df["roll5_rpg_diff"] = (
            df["prev_roll5_runs_per_game"].fillna(0)
            - df["opp_prev_roll5_rpg"].fillna(0)
        )
    else:
        df["roll5_rpg_diff"] = 0.0

    if "prev_roll5_runs_allowed" in df.columns and "opp_prev_roll5_ra" in df.columns:
        df["roll5_ra_diff"] = (
            df["opp_prev_roll5_ra"].fillna(0)
            - df["prev_roll5_runs_allowed"].fillna(0)
        )
    else:
        df["roll5_ra_diff"] = 0.0

    # Bullpen ERA differential (opponent worse = positive for us)
    if "bullpen_era" in df.columns and "opp_bullpen_era" in df.columns:
        df["bullpen_era_diff"] = (
            df["opp_bullpen_era"].fillna(4.0)
            - df["bullpen_era"].fillna(4.0)
        )
    else:
        df["bullpen_era_diff"] = 0.0

    return df


def compute_target(df):
    """Compute the target variable: home_win."""
    print("   -> Computing target variable (home_win)...")
    df["home_win"] = (df["team_score"] > df["opp_score"]).astype(int)
    return df


def run_features(league=LEAGUE):
    """Run the full MLB feature engineering pipeline."""
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Run 'python -m mlb.data' first to fetch game data.")
        return

    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file, low_memory=False)
    print(f"Loaded {len(df)} rows.")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df = clean_stale_data(df)
    df = calculate_rolling_stats(df)
    df = calculate_pitcher_rolling_stats(df)
    df = add_park_factor(df)
    df = merge_opponent_stats(df)
    df = compute_differentials(df)
    df = compute_target(df)

    df.to_csv(data_file, index=False)
    print(f"Saved {len(df)} processed rows to {data_file}")
    return df


def main():
    parser = argparse.ArgumentParser(description="MLB feature engineering")
    parser.add_argument("--league", default=LEAGUE)
    args = parser.parse_args()
    run_features(args.league)


if __name__ == "__main__":
    main()
