"""Grade the MLB market-v2 shadow model against game outcomes.

The shadow model emits MarketV2_* columns alongside production picks but does
not drive bets. This module joins archived predictions to game outcomes from
the training data and computes paired Brier / accuracy / ROI for production,
shadow, and direct no-vig market-only on identical games.

Brier scoring is home-side throughout: (prob_home - I[home_won])^2 for
production, shadow, and market. Identical scoring lets the three deltas be
compared directly.

ROI is unit-stake at the production-pick moneyline (Std_Odds in the
predictions archive). Std_Odds is only persisted for the production side, so
shadow / market ROI are computable only when those picks coincide with
production. Otherwise the row is flagged roi_data_missing and excluded from
the ROI aggregate.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from glob import glob
from typing import Optional

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from mlb.predict import find_best_match  # noqa: E402  reuse production team-name normalizer

DEFAULT_ARCHIVE_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "data", "mlb_training_data_processed.csv")
DEFAULT_LEDGER = os.path.join(BASE_DIR, "data", "mlb_shadow_grader_ledger.csv")

ARCHIVE_RE = re.compile(r"predictions_mlb_(\d{8})\.csv$")

REQUIRED_SHADOW_COLS = (
    "MarketV2_Status",
    "MarketV2_Prob_Home",
    "MarketV2_Pick",
    "MarketV2_Conf",
    "MarketV2_Market_NoVig_Home",
    "MarketV2_Agrees_With_Production",
)

LEDGER_KEY = ["archive_date", "home_team", "away_team", "game_time"]

LEDGER_COLUMNS = [
    "archive_date",
    "game_time",
    "home_team",
    "away_team",
    "matchup",
    "production_pick",
    "shadow_pick",
    "market_pick",
    "agrees_with_production",
    "outcome_status",
    "home_won",
    "production_correct",
    "shadow_correct",
    "market_correct",
    "production_brier",
    "shadow_brier",
    "market_brier",
    "production_prob_home",
    "shadow_prob_home",
    "market_prob_home",
    "std_odds",
    "production_roi_units",
    "shadow_roi_units",
    "market_roi_units",
    "roi_data_missing",
]


def discover_archives(
    archive_dir: str, since: Optional[str] = None
) -> list[tuple[str, str]]:
    """Return sorted [(date_iso, path)] for predictions_mlb_YYYYMMDD.csv files."""
    out = []
    for path in sorted(glob(os.path.join(archive_dir, "predictions_mlb_*.csv"))):
        m = ARCHIVE_RE.search(os.path.basename(path))
        if not m:
            continue
        ds = m.group(1)
        date_iso = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
        if since and date_iso < since:
            continue
        out.append((date_iso, path))
    return out


def has_shadow_columns(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in REQUIRED_SHADOW_COLS)


def parse_matchup(matchup: str) -> tuple[str, str]:
    """Return (away, home) from 'Away @ Home' or ('','') if malformed."""
    if not isinstance(matchup, str):
        return ("", "")
    parts = matchup.split(" @ ", 1)
    if len(parts) != 2:
        return ("", "")
    return parts[0].strip(), parts[1].strip()


def load_outcomes(processed_csv: str) -> pd.DataFrame:
    """Build (date, home_team, away_team) -> home_win lookup from training data."""
    df = pd.read_csv(processed_csv, low_memory=False)
    home = df[df["is_home"] == 1].copy()
    home = home.dropna(subset=["home_win"])
    home["date"] = pd.to_datetime(home["date"]).dt.strftime("%Y-%m-%d")
    out = home[["date", "team", "opponent", "game_time", "home_win"]].copy()
    out = out.rename(columns={"team": "home_team", "opponent": "away_team"})
    out["home_win"] = out["home_win"].astype(int)
    return out


def parse_std_odds(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s.replace("+", "")))
    except (ValueError, TypeError):
        return None


def american_to_payout(odds: int) -> float:
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def _market_only_pick(market_home_prob: float, home_disp: str, away_disp: str) -> str:
    return home_disp if market_home_prob > 0.5 else away_disp


def _bet_roi(won: bool, odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    return american_to_payout(odds) if won else -1.0


def _normalize_team(name: str, known_teams: set[str]) -> Optional[str]:
    if name in known_teams:
        return name
    return find_best_match(name, known_teams)


def _time_to_minutes(value) -> Optional[int]:
    """Best-effort 'HH:MM' -> minutes since midnight; None if unparseable."""
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s or ":" not in s:
        return None
    try:
        h, m = s.split(":")[:2]
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _select_outcome(match: pd.DataFrame, pred_game_time: str) -> Optional[int]:
    """Pick the doubleheader row closest to the prediction's game_time.

    Both the prediction archive and the training data store game_time, but the
    archive uses UTC HH:MM (from ESPN) while the training CSV may use a
    different convention. Absolute clock values may not align, but ordering
    within a doubleheader does, so closest-by-minutes works as a disambiguator.
    """
    if match.empty:
        return None
    if len(match) == 1:
        return int(match.iloc[0]["home_win"])
    pred_minutes = _time_to_minutes(pred_game_time)
    if pred_minutes is None:
        return int(match.iloc[0]["home_win"])
    deltas = match["game_time"].map(
        lambda t: abs((_time_to_minutes(t) or 0) - pred_minutes)
    )
    return int(match.iloc[deltas.values.argmin()]["home_win"])


def grade_archive(
    archive_path: str,
    archive_date: str,
    outcomes: pd.DataFrame,
    known_teams: set[str],
) -> pd.DataFrame:
    """Grade a single dated prediction archive into ledger rows."""
    try:
        df = pd.read_csv(archive_path, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    if not has_shadow_columns(df) or df.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    rows = []
    for _, r in df.iterrows():
        if str(r.get("MarketV2_Status", "")) != "ok":
            continue

        away_disp, home_disp = parse_matchup(r.get("Matchup", ""))
        if not home_disp or not away_disp:
            continue

        home_team = _normalize_team(home_disp, known_teams)
        away_team = _normalize_team(away_disp, known_teams)
        if home_team is None or away_team is None:
            continue

        dt = str(r.get("Date/Time", ""))
        gt = dt.split(" ", 1)[1] if " " in dt else ""

        match = outcomes[
            (outcomes["date"] == archive_date)
            & (outcomes["home_team"] == home_team)
            & (outcomes["away_team"] == away_team)
        ]
        home_won: Optional[int] = _select_outcome(match, gt)

        prob_home = float(r["Prob_Home"])
        shadow_prob_home = float(r["MarketV2_Prob_Home"])
        market_home = float(r["MarketV2_Market_NoVig_Home"])

        production_pick = str(r["Pick"])
        shadow_pick = str(r["MarketV2_Pick"])
        market_pick = _market_only_pick(market_home, home_disp, away_disp)
        std_odds = parse_std_odds(r.get("Std_Odds"))
        std_odds_home = parse_std_odds(r.get("Std_Odds_Home"))
        std_odds_away = parse_std_odds(r.get("Std_Odds_Away"))

        def _odds_for_pick(pick: str) -> Optional[int]:
            """Best moneyline for the picked side, falling back to legacy
            production-only Std_Odds when the new columns are absent."""
            if pick == home_disp and std_odds_home is not None:
                return std_odds_home
            if pick == away_disp and std_odds_away is not None:
                return std_odds_away
            if pick == production_pick:
                return std_odds
            return None

        if home_won is None:
            outcome_status = "outcome_pending"
            production_correct = shadow_correct = market_correct = None
            production_brier = shadow_brier = market_brier = None
            production_roi = shadow_roi = market_roi = None
            roi_data_missing = True
        else:
            outcome_status = "graded"
            home_won_bool = bool(home_won)

            production_correct = (production_pick == home_disp) == home_won_bool
            shadow_correct = (shadow_pick == home_disp) == home_won_bool
            market_correct = (market_pick == home_disp) == home_won_bool

            production_brier = (prob_home - home_won) ** 2
            shadow_brier = (shadow_prob_home - home_won) ** 2
            market_brier = (market_home - home_won) ** 2

            production_roi = _bet_roi(production_correct, _odds_for_pick(production_pick))
            shadow_roi = _bet_roi(shadow_correct, _odds_for_pick(shadow_pick))
            market_roi = _bet_roi(market_correct, _odds_for_pick(market_pick))
            roi_data_missing = (
                production_roi is None or shadow_roi is None or market_roi is None
            )

        rows.append(
            {
                "archive_date": archive_date,
                "game_time": gt,
                "home_team": home_team,
                "away_team": away_team,
                "matchup": r.get("Matchup", ""),
                "production_pick": production_pick,
                "shadow_pick": shadow_pick,
                "market_pick": market_pick,
                "agrees_with_production": bool(
                    r.get("MarketV2_Agrees_With_Production")
                ),
                "outcome_status": outcome_status,
                "home_won": home_won,
                "production_correct": production_correct,
                "shadow_correct": shadow_correct,
                "market_correct": market_correct,
                "production_brier": production_brier,
                "shadow_brier": shadow_brier,
                "market_brier": market_brier,
                "production_prob_home": prob_home,
                "shadow_prob_home": shadow_prob_home,
                "market_prob_home": market_home,
                "std_odds": std_odds,
                "production_roi_units": production_roi,
                "shadow_roi_units": shadow_roi,
                "market_roi_units": market_roi,
                "roi_data_missing": roi_data_missing,
            }
        )

    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def grade(
    archive_dir: str = DEFAULT_ARCHIVE_DIR,
    processed_csv: str = DEFAULT_DATA_FILE,
    since: Optional[str] = None,
) -> pd.DataFrame:
    """Build the full shadow ledger by grading every eligible dated archive."""
    archives = discover_archives(archive_dir, since)
    outcomes = load_outcomes(processed_csv)
    known_teams = set(outcomes["home_team"].unique()) | set(
        outcomes["away_team"].unique()
    )

    frames = [
        grade_archive(path, date_iso, outcomes, known_teams)
        for date_iso, path in archives
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    ledger = pd.concat(frames, ignore_index=True)
    return (
        ledger.sort_values(LEDGER_KEY)
        .drop_duplicates(LEDGER_KEY, keep="last")
        .reset_index(drop=True)
    )


def write_ledger(ledger: pd.DataFrame, path: str = DEFAULT_LEDGER) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    ledger.to_csv(path, index=False)


def aggregate_report(ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {"n_total": 0, "n_graded": 0}

    graded = ledger[ledger["outcome_status"] == "graded"]
    n_graded = len(graded)
    if n_graded == 0:
        return {"n_total": int(len(ledger)), "n_graded": 0}

    out = {
        "n_total": int(len(ledger)),
        "n_graded": int(n_graded),
        "production_brier_mean": float(graded["production_brier"].mean()),
        "shadow_brier_mean": float(graded["shadow_brier"].mean()),
        "market_brier_mean": float(graded["market_brier"].mean()),
        "shadow_minus_market_brier": float(
            (graded["shadow_brier"] - graded["market_brier"]).mean()
        ),
        "production_minus_market_brier": float(
            (graded["production_brier"] - graded["market_brier"]).mean()
        ),
        "shadow_minus_production_brier": float(
            (graded["shadow_brier"] - graded["production_brier"]).mean()
        ),
        "production_accuracy": float(graded["production_correct"].mean()),
        "shadow_accuracy": float(graded["shadow_correct"].mean()),
        "market_accuracy": float(graded["market_correct"].mean()),
        "agreement_rate": float(graded["agrees_with_production"].mean()),
    }

    roi = graded[~graded["roi_data_missing"]]
    out["n_roi_eligible"] = int(len(roi))
    out["n_roi_missing"] = int(n_graded - len(roi))
    if len(roi) > 0:
        out["production_roi_units"] = float(roi["production_roi_units"].sum())
        out["shadow_roi_units"] = float(roi["shadow_roi_units"].sum())
        out["market_roi_units"] = float(roi["market_roi_units"].sum())
    return out


def format_report(report: dict) -> str:
    n_total = report.get("n_total", 0)
    n_graded = report.get("n_graded", 0)
    if n_graded == 0:
        return f"Shadow grader: 0 graded games (total ledger rows: {n_total})."

    lines = [
        f"Shadow grader: {n_graded} graded games "
        f"(total {n_total}, ROI eligible {report['n_roi_eligible']}, "
        f"missing {report['n_roi_missing']})",
        f"  Brier mean: production={report['production_brier_mean']:.4f}  "
        f"shadow={report['shadow_brier_mean']:.4f}  "
        f"market={report['market_brier_mean']:.4f}",
        f"  Brier deltas: shadow-market={report['shadow_minus_market_brier']:+.4f}  "
        f"production-market={report['production_minus_market_brier']:+.4f}  "
        f"shadow-production={report['shadow_minus_production_brier']:+.4f}",
        f"  Accuracy: production={report['production_accuracy']:.3f}  "
        f"shadow={report['shadow_accuracy']:.3f}  "
        f"market={report['market_accuracy']:.3f}",
        f"  Agreement (shadow vs production): {report['agreement_rate']:.3f}",
    ]
    if "production_roi_units" in report:
        lines.append(
            f"  ROI (units): production={report['production_roi_units']:+.2f}  "
            f"shadow={report['shadow_roi_units']:+.2f}  "
            f"market={report['market_roi_units']:+.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Grade MLB shadow vs production picks.")
    p.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR)
    p.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    p.add_argument("--ledger", default=DEFAULT_LEDGER)
    p.add_argument("--since", default=None, help="YYYY-MM-DD lower bound on archive dates.")
    p.add_argument("--no-write", action="store_true")
    p.add_argument(
        "--no-report", action="store_true", help="Suppress aggregate report at end."
    )
    args = p.parse_args()

    ledger = grade(args.archive_dir, args.data_file, args.since)
    if not args.no_write:
        write_ledger(ledger, args.ledger)
        print(f"Wrote {len(ledger)} ledger rows to {args.ledger}")
    if not args.no_report:
        print(format_report(aggregate_report(ledger)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
