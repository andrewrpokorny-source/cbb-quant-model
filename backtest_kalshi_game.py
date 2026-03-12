"""Historical backtest for Kalshi GAME market picks."""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Callable

import pandas as pd

from betting import VALUE_RATINGS, RATING_RANK, get_rating
from betting.ev_calculator import kalshi_fee_cents
from grade_predictions import fetch_completed_games
from kalshi_game_archive import ARCHIVE_COLUMNS
from kalshi.client import KalshiClient
from league_config import get_league_artifact_paths, normalize_league
from settle_bets import determine_bet_result, match_bet_to_game, parse_bet_line


RESULT_COLUMNS = [
    "captured_at",
    "league",
    "game_date",
    "game_datetime",
    "matchup",
    "home_team",
    "away_team",
    "pick",
    "picked_team",
    "kalshi_side",
    "kalshi_ticker",
    "kalshi_price",
    "kalshi_fee",
    "edge",
    "edge_pct",
    "rating",
    "conf",
    "result",
    "profit",
    "stake",
    "payout",
    "roi",
    "edge_bucket",
    "price_bucket",
]


def _parse_game_date_from_kalshi_ticker(ticker: str) -> str | None:
    """Parse a YYYY-MM-DD game date from a Kalshi market ticker."""
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", str(ticker or ""))
    if not match:
        return None
    year_short, month_str, day = match.groups()
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = months.get(month_str)
    if month is None:
        return None
    try:
        return pd.Timestamp(2000 + int(year_short), month, int(day)).strftime("%Y-%m-%d")
    except ValueError:
        return None


def filter_archived_predictions_by_rating(
    archived_inputs: pd.DataFrame,
    min_rating: str = "GOOD",
) -> pd.DataFrame:
    """Filter archived predictions to the minimum actionable rating."""
    threshold = str(min_rating or "GOOD").strip().upper()
    if threshold == "ALL":
        return archived_inputs.copy()
    if threshold not in RATING_RANK:
        raise ValueError(f"Unsupported min_rating '{min_rating}'.")

    filtered = archived_inputs.copy()
    ratings = filtered["rating"].fillna("").astype(str).str.upper()
    keep = ratings.map(lambda rating: RATING_RANK.get(rating, -1) >= RATING_RANK[threshold])
    return filtered[keep].copy()


def load_actual_betting_history_rows(csv_path: str, league: str | None = None) -> pd.DataFrame:
    """Load settled Kalshi GAME bet rows from betting_history.csv."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["platform"] = df["platform"].astype(str).str.lower()
    df["bet_type"] = df["bet_type"].astype(str).str.lower()
    settled = df[
        (df["platform"] == "kalshi")
        & (df["bet_type"] == "game")
        & df["result"].astype(str).str.lower().isin({"win", "loss", "void"})
    ].copy()
    if league is not None:
        settled = settled[settled["league"].astype(str).str.lower() == normalize_league(league)]
    return settled.reset_index(drop=True)


def price_bucket(price_cents) -> str:
    """Bucket a Kalshi ask price into coarse bands."""
    if price_cents is None or pd.isna(price_cents):
        return "unknown"
    price = float(price_cents)
    if price < 25:
        return "00-24"
    if price < 40:
        return "25-39"
    if price < 60:
        return "40-59"
    if price < 75:
        return "60-74"
    return "75-100"


def edge_bucket(edge) -> str:
    """Bucket edge into recommendation-like bands."""
    if edge is None or pd.isna(edge):
        return "unknown"
    return get_rating(float(edge)).value


def calculate_kalshi_contract_outcome(price_cents, result: str) -> tuple[float, float, float]:
    """Return (stake, payout, profit) for one Kalshi contract at the ask price."""
    price = float(price_cents)
    stake = round((price + kalshi_fee_cents(price)) / 100.0, 4)

    if result == "win":
        payout = 1.0
    elif result == "void":
        payout = stake
    else:
        payout = 0.0

    profit = round(payout - stake, 4)
    return stake, payout, profit


def result_from_market_result(side: str, market_result: str | None) -> str | None:
    """Map a Kalshi market_result (yes/no) to win/loss for a chosen side."""
    normalized_side = str(side or "").strip().upper()
    normalized_result = str(market_result or "").strip().lower()
    if normalized_side not in {"YES", "NO"}:
        return None
    if normalized_result not in {"yes", "no"}:
        return None
    if normalized_side == normalized_result.upper():
        return "win"
    return "loss"


def _standardize_archive_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    for col in ARCHIVE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[ARCHIVE_COLUMNS].copy()
    df["source"] = df["source"].fillna(source).replace("", source)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce")
    df["game_datetime"] = pd.to_datetime(df["game_datetime"], utc=True, errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    numeric_cols = [
        "kalshi_yes_price",
        "kalshi_no_price",
        "kalshi_price",
        "kalshi_fee",
        "win_model_home_prob",
        "conf",
        "edge",
        "edge_pct",
        "units",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_kalshi_game_history(archive_file: str) -> pd.DataFrame:
    """Load the dedicated Kalshi GAME archive if present."""
    if not os.path.exists(archive_file):
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    return _standardize_archive_frame(pd.read_csv(archive_file), source="prediction_run")


def load_prediction_game_archives(base_dir: str, league: str) -> pd.DataFrame:
    """Backfill any archived prediction CSVs that already contain GAME picks."""
    paths = get_league_artifact_paths(base_dir, league)
    prefix = paths["predictions_archive_prefix"]
    pattern = os.path.join(base_dir, f"{prefix}_*.csv")

    frames = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        if "Bet_Type" not in df.columns:
            continue
        game_df = df[df["Bet_Type"] == "game"].copy()
        if game_df.empty:
            continue

        file_date = os.path.basename(path).split("_")[-1].split(".")[0]
        captured_at = pd.to_datetime(file_date, format="%Y%m%d", errors="coerce", utc=True)
        if pd.isna(captured_at):
            captured_at = pd.Timestamp.utcnow()

        matchup_split = game_df["Matchup"].fillna("").str.split(" @ ", n=1, expand=True)
        away_team = matchup_split[0].fillna("")
        home_team = matchup_split[1].fillna("")
        file_year = captured_at.year
        game_time = pd.to_datetime(
            str(file_year) + "/" + game_df["Date/Time"].fillna(""),
            format="%Y/%m/%d %I:%M %p",
            errors="coerce",
        )
        # If game date is before the file date, it's a new-year rollover
        needs_bump = game_time < captured_at - pd.Timedelta(days=1)
        if needs_bump.any():
            game_time = game_time.where(
                ~needs_bump,
                pd.to_datetime(
                    str(file_year + 1) + "/" + game_df["Date/Time"].fillna(""),
                    format="%Y/%m/%d %I:%M %p",
                    errors="coerce",
                ),
            )
        standardized = pd.DataFrame(
            {
                "captured_at": captured_at,
                "league": normalize_league(league),
                "game_date": game_time.dt.strftime("%Y-%m-%d"),
                "game_datetime": game_time.dt.tz_localize("US/Eastern", nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC"),
                "matchup": game_df["Matchup"],
                "home_team": home_team,
                "away_team": away_team,
                "pick": game_df["Pick"],
                "picked_team": game_df.get("Picked_Team", pd.Series("", index=game_df.index)),
                "kalshi_side": game_df.get("Kalshi_Side", pd.Series("", index=game_df.index)),
                "kalshi_ticker": game_df.get("Kalshi_Ticker", pd.Series("", index=game_df.index)),
                "kalshi_title": game_df.get("Kalshi_Title", pd.Series("", index=game_df.index)),
                "kalshi_yes_team": game_df.get("Kalshi_Yes_Team", pd.Series("", index=game_df.index)),
                "kalshi_yes_price": game_df.get("Kalshi_Yes", pd.Series(pd.NA, index=game_df.index)),
                "kalshi_no_price": game_df.get("Kalshi_No", pd.Series(pd.NA, index=game_df.index)),
                "kalshi_price": game_df.get("Kalshi_Price", pd.Series(pd.NA, index=game_df.index)),
                "kalshi_fee": game_df.get("Kalshi_Fee", pd.Series(pd.NA, index=game_df.index)),
                "win_model_home_prob": game_df.get("Win_Model_Home_Prob", pd.Series(pd.NA, index=game_df.index)),
                "conf": game_df.get("Conf", pd.Series(pd.NA, index=game_df.index)),
                "edge": game_df.get("Edge", pd.Series(pd.NA, index=game_df.index)),
                "edge_pct": pd.to_numeric(
                    game_df.get("Edge_Pct", pd.Series(pd.NA, index=game_df.index))
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace("+", "", regex=False),
                    errors="coerce",
                ),
                "rating": game_df.get("Rating", pd.Series("", index=game_df.index)),
                "units": game_df.get("Units", pd.Series(pd.NA, index=game_df.index)),
                "source": "predictions_archive",
            }
        )
        frames.append(_standardize_archive_frame(standardized, source="predictions_archive"))

    if not frames:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_backtest_inputs(
    *,
    base_dir: str,
    league: str,
    archive_file: str | None = None,
    include_prediction_archives: bool = True,
) -> pd.DataFrame:
    """Load archived Kalshi GAME snapshots from all supported local sources."""
    paths = get_league_artifact_paths(base_dir, league)
    archive_path = archive_file or paths["kalshi_game_archive_file"]

    frames = [load_kalshi_game_history(archive_path)]
    if include_prediction_archives:
        frames.append(load_prediction_game_archives(base_dir, league))

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined

    combined = combined[combined["league"] == normalize_league(league)].copy()
    combined = combined.drop_duplicates()
    combined = combined.sort_values(["captured_at", "matchup", "kalshi_ticker"], na_position="last")
    return combined


def load_actual_betting_history_results(csv_path: str, league: str | None = None) -> pd.DataFrame:
    """Load settled Kalshi GAME bets from betting_history.csv into result format."""
    settled = load_actual_betting_history_rows(csv_path, league=league)
    if settled.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    settled["captured_at"] = pd.to_datetime(settled["date"], errors="coerce")
    settled["game_date"] = settled["bet_id"].apply(_parse_game_date_from_kalshi_ticker)
    settled["game_date"] = settled["game_date"].fillna(settled["captured_at"].dt.strftime("%Y-%m-%d"))
    settled["stake"] = pd.to_numeric(settled["wager"], errors="coerce")
    settled["payout"] = pd.to_numeric(settled["payout"], errors="coerce")
    settled["profit"] = pd.to_numeric(settled["profit"], errors="coerce")
    settled["roi"] = settled["profit"] / settled["stake"]
    settled["picked_team"] = settled["line"].apply(
        lambda line: (parse_bet_line(line) or {}).get("team", "")
    )
    results = pd.DataFrame(
        {
            "captured_at": settled["captured_at"],
            "league": settled["league"].astype(str).str.lower(),
            "game_date": settled["game_date"],
            "game_datetime": pd.NaT,
            "matchup": settled["game"],
            "home_team": "",
            "away_team": "",
            "pick": settled["line"],
            "picked_team": settled["picked_team"],
            "kalshi_side": "",
            "kalshi_ticker": settled["bet_id"],
            "kalshi_price": pd.NA,
            "kalshi_fee": pd.NA,
            "edge": pd.NA,
            "edge_pct": pd.NA,
            "rating": "actual_bet",
            "conf": pd.NA,
            "result": settled["result"].astype(str).str.lower(),
            "profit": settled["profit"],
            "stake": settled["stake"],
            "payout": settled["payout"],
            "roi": settled["roi"],
            "edge_bucket": "unknown",
            "price_bucket": "unknown",
        }
    )
    return results[RESULT_COLUMNS].reset_index(drop=True)


def compare_actual_bets_to_archived_predictions(
    archived_inputs: pd.DataFrame,
    betting_history_file: str,
    *,
    league: str,
) -> pd.DataFrame:
    """Join settled Kalshi GAME bets to archived model snapshots by ticker."""
    actual = load_actual_betting_history_rows(betting_history_file, league=league)
    if actual.empty or archived_inputs.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    archived = archived_inputs.copy()
    archived = archived[archived["league"] == normalize_league(league)].copy()
    archived = archived[archived["kalshi_ticker"].notna() & (archived["kalshi_ticker"].astype(str) != "")]
    if archived.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    archived = archived.sort_values("captured_at", na_position="last")
    archived = archived.groupby(["league", "kalshi_ticker"], as_index=False, dropna=False).tail(1)

    actual = actual.copy()
    actual["league"] = actual["league"].astype(str).str.lower()
    actual["bet_id"] = actual["bet_id"].astype(str)
    actual["stake"] = pd.to_numeric(actual["wager"], errors="coerce")
    actual["payout"] = pd.to_numeric(actual["payout"], errors="coerce")
    actual["profit"] = pd.to_numeric(actual["profit"], errors="coerce")
    actual["result"] = actual["result"].astype(str).str.lower()
    actual["captured_at_actual"] = pd.to_datetime(actual["date"], errors="coerce")
    actual["game_date"] = actual["bet_id"].apply(_parse_game_date_from_kalshi_ticker)
    actual["game_date"] = actual["game_date"].fillna(actual["captured_at_actual"].dt.strftime("%Y-%m-%d"))

    merged = actual.merge(
        archived,
        left_on=["league", "bet_id"],
        right_on=["league", "kalshi_ticker"],
        how="inner",
        suffixes=("_actual", "_pred"),
    )
    if merged.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    picked_team = merged["picked_team"] if "picked_team" in merged.columns else merged["line"].apply(
        lambda line: (parse_bet_line(line) or {}).get("team", "")
    )
    game_date_pred = merged["game_date_pred"] if "game_date_pred" in merged.columns else merged.get("game_date")
    game_date_actual = merged["game_date_actual"] if "game_date_actual" in merged.columns else merged.get("game_date")

    comparison = pd.DataFrame(
        {
            "captured_at": merged["captured_at"],
            "league": merged["league"],
            "game_date": game_date_pred.fillna(game_date_actual),
            "game_datetime": merged["game_datetime"],
            "matchup": merged["matchup"],
            "home_team": merged["home_team"],
            "away_team": merged["away_team"],
            "pick": merged["pick"],
            "picked_team": picked_team,
            "kalshi_side": merged["kalshi_side"],
            "kalshi_ticker": merged["kalshi_ticker"],
            "kalshi_price": merged["kalshi_price"],
            "kalshi_fee": merged["kalshi_fee"],
            "edge": merged["edge"],
            "edge_pct": merged["edge_pct"],
            "rating": merged["rating"],
            "conf": merged["conf"],
            "result": merged["result"],
            "profit": merged["profit"],
            "stake": merged["stake"],
            "payout": merged["payout"],
            "roi": merged["profit"] / merged["stake"],
            "edge_bucket": merged["edge"].apply(edge_bucket),
            "price_bucket": merged["kalshi_price"].apply(price_bucket),
        }
    )
    return comparison[RESULT_COLUMNS].reset_index(drop=True)


def select_latest_snapshot_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest archived pick for each league/game/ticker combination."""
    if df.empty:
        return df.copy()
    keys = ["league", "game_date", "home_team", "away_team", "kalshi_ticker"]
    ordered = df.sort_values("captured_at", na_position="last")
    return ordered.groupby(keys, as_index=False, dropna=False).tail(1).reset_index(drop=True)


def resolve_backtest_results(
    snapshots: pd.DataFrame,
    *,
    league: str,
    score_fetcher: Callable = fetch_completed_games,
    market_resolver: Callable[[str], dict] | None = None,
) -> pd.DataFrame:
    """Resolve archived Kalshi GAME picks against final scores."""
    if snapshots.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    canonical = normalize_league(league)
    snapshots = snapshots[snapshots["league"] == canonical].copy()
    snapshots["game_date"] = pd.to_datetime(snapshots["game_date"], errors="coerce")

    score_cache = {}
    market_cache = {}
    rows = []
    for record in snapshots.itertuples(index=False):
        if pd.isna(record.game_date):
            continue
        result = None
        if market_resolver and record.kalshi_ticker:
            if record.kalshi_ticker not in market_cache:
                market_cache[record.kalshi_ticker] = market_resolver(record.kalshi_ticker)
            market = market_cache.get(record.kalshi_ticker, {})
            result = result_from_market_result(
                record.kalshi_side,
                market.get("market_result") or market.get("result"),
            )

        if result is None:
            date_key = record.game_date.date()
            if date_key not in score_cache:
                score_cache[date_key] = score_fetcher(pd.Timestamp(date_key).to_pydatetime(), league=canonical)

            completed_games = score_cache.get(date_key, {})
            game_result = completed_games.get((record.home_team, record.away_team))
            if game_result is None:
                game_result = match_bet_to_game(record.matchup, completed_games)
            if game_result is None:
                continue

            parsed = parse_bet_line(record.pick)
            if not parsed:
                continue

            result = determine_bet_result(parsed, game_result, record.matchup)
        if result is None or pd.isna(record.kalshi_price):
            continue

        stake, payout, profit = calculate_kalshi_contract_outcome(record.kalshi_price, result)
        roi = round(profit / stake, 4) if stake else pd.NA
        rows.append(
            {
                "captured_at": record.captured_at,
                "league": record.league,
                "game_date": record.game_date.strftime("%Y-%m-%d"),
                "game_datetime": record.game_datetime,
                "matchup": record.matchup,
                "home_team": record.home_team,
                "away_team": record.away_team,
                "pick": record.pick,
                "picked_team": record.picked_team,
                "kalshi_side": record.kalshi_side,
                "kalshi_ticker": record.kalshi_ticker,
                "kalshi_price": float(record.kalshi_price),
                "kalshi_fee": float(record.kalshi_fee) if not pd.isna(record.kalshi_fee) else kalshi_fee_cents(record.kalshi_price),
                "edge": float(record.edge) if not pd.isna(record.edge) else pd.NA,
                "edge_pct": float(record.edge_pct) if not pd.isna(record.edge_pct) else pd.NA,
                "rating": record.rating,
                "conf": float(record.conf) if not pd.isna(record.conf) else pd.NA,
                "result": result,
                "profit": profit,
                "stake": stake,
                "payout": payout,
                "roi": roi,
                "edge_bucket": edge_bucket(record.edge),
                "price_bucket": price_bucket(record.kalshi_price),
            }
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def summarize_backtest(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build summary tables for the resolved Kalshi GAME backtest."""
    if results.empty:
        empty = pd.DataFrame()
        return {"overall": empty, "by_edge": empty, "by_price": empty, "by_rating": empty}

    def _summarize(group_cols):
        grouped = results.groupby(group_cols, dropna=False).agg(
            bets=("result", "size"),
            wins=("result", lambda s: int((s == "win").sum())),
            losses=("result", lambda s: int((s == "loss").sum())),
            voids=("result", lambda s: int((s == "void").sum())),
            total_profit=("profit", "sum"),
            total_stake=("stake", "sum"),
            avg_edge_pct=("edge_pct", "mean"),
            avg_price=("kalshi_price", "mean"),
            avg_conf=("conf", "mean"),
        ).reset_index()
        grouped["win_rate"] = grouped.apply(
            lambda row: row["wins"] / (row["wins"] + row["losses"])
            if (row["wins"] + row["losses"]) else pd.NA,
            axis=1,
        )
        grouped["roi"] = grouped.apply(
            lambda row: row["total_profit"] / row["total_stake"] if row["total_stake"] else pd.NA,
            axis=1,
        )
        return grouped.sort_values("bets", ascending=False)

    overall = _summarize(["league"])
    return {
        "overall": overall,
        "by_edge": _summarize(["league", "edge_bucket"]),
        "by_price": _summarize(["league", "price_bucket"]),
        "by_rating": _summarize(["league", "rating"]),
    }


def print_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Render the summary tables to stdout."""
    for name, table in summary.items():
        print(f"\n{name.upper()}:")
        if table.empty:
            print("  No resolved bets.")
            continue
        print(table.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest archived Kalshi GAME market picks.")
    parser.add_argument("--league", default="mens", help="mens or womens")
    parser.add_argument("--archive-file", default=None, help="Optional Kalshi GAME archive CSV path")
    parser.add_argument(
        "--betting-history-file",
        default=None,
        help="Optional betting_history.csv path to summarize actual settled Kalshi GAME bets",
    )
    parser.add_argument(
        "--compare-betting-history-file",
        default=None,
        help="Optional betting_history.csv path to join settled Kalshi GAME bets to archived predictions by ticker",
    )
    parser.add_argument(
        "--resolution-source",
        choices=("auto", "kalshi", "espn"),
        default="auto",
        help="Resolve outcomes from Kalshi historical markets when possible, or ESPN scores.",
    )
    parser.add_argument(
        "--min-rating",
        choices=("ALL", "PASS", "MARGINAL", "GOOD", "STRONG"),
        default="GOOD",
        help="Minimum archived Kalshi GAME rating to include in prediction replay backtests.",
    )
    parser.add_argument(
        "--all-snapshots",
        action="store_true",
        help="Use every archived snapshot instead of only the latest per game",
    )
    parser.add_argument(
        "--skip-prediction-archives",
        action="store_true",
        help="Ignore archived prediction CSVs and use only kalshi_game_history.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    league = normalize_league(args.league)
    if args.betting_history_file:
        results = load_actual_betting_history_results(args.betting_history_file, league=league)
        if results.empty:
            print("No settled Kalshi GAME bets found in the betting history file.")
            return
        print_summary(summarize_backtest(results))
        return

    inputs = load_backtest_inputs(
        base_dir=base_dir,
        league=league,
        archive_file=args.archive_file,
        include_prediction_archives=not args.skip_prediction_archives,
    )
    if args.compare_betting_history_file:
        comparisons = compare_actual_bets_to_archived_predictions(
            inputs,
            args.compare_betting_history_file,
            league=league,
        )
        if comparisons.empty:
            print("No settled Kalshi GAME bets matched archived predictions by ticker.")
            return
        print_summary(summarize_backtest(comparisons))
        return

    if inputs.empty:
        print("No archived Kalshi GAME picks found.")
        return

    filtered_inputs = filter_archived_predictions_by_rating(inputs, min_rating=args.min_rating)
    if filtered_inputs.empty:
        print(f"No archived Kalshi GAME picks met min rating {args.min_rating}.")
        return

    snapshots = filtered_inputs if args.all_snapshots else select_latest_snapshot_per_game(filtered_inputs)
    market_resolver = None
    if args.resolution_source in {"auto", "kalshi"}:
        client = KalshiClient()
        market_resolver = client.get_market_any
    results = resolve_backtest_results(
        snapshots,
        league=league,
        market_resolver=market_resolver,
    )
    if results.empty:
        print("No archived Kalshi GAME picks could be resolved against completed scores.")
        return

    print_summary(summarize_backtest(results))


if __name__ == "__main__":
    main()
