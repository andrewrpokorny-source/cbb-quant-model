"""Backfill MLB moneylines from ESPN's core odds endpoint.

The normal ESPN scoreboard endpoint drops odds for many completed games. The
core competition odds endpoint retains provider odds for historical events:

    https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{event_id}/competitions/{competition_id}/odds

This script discovers ESPN event IDs from the scoreboard by date, fetches the
core odds for each event, and can emit:

1. A game-level odds CSV.
2. An enriched copy of the MLB training CSV with row-wise ``moneyline`` filled.

It does not modify the source CSV unless an explicit output path is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "mlb_training_data_processed.csv"
DEFAULT_ODDS_OUTPUT = REPO_ROOT / "data" / "mlb_espn_odds_backfill.csv"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "espn_mlb_odds"

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/"
    "scoreboard?limit=200&dates={date_yyyymmdd}"
)
CORE_ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/"
    "events/{event_id}/competitions/{competition_id}/odds?lang=en&region=us"
)


@dataclass(frozen=True)
class EspnEvent:
    date: str
    game_time: str
    event_id: str
    competition_id: str
    home_abbr: str
    away_abbr: str
    home_team: str
    away_team: str


def _cache_path(cache_dir: Path, namespace: str, key: str) -> Path:
    return cache_dir / namespace / f"{key}.json"


def _get_json(
    session: requests.Session,
    url: str,
    cache_path: Path | None,
    sleep_seconds: float,
) -> dict:
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())
    response = session.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return data


def _american_from_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("american", "alternateDisplayValue", "displayValue"):
            raw = value.get(key)
            if raw not in (None, "", "EVEN"):
                return _american_from_value(raw)
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None
    if not math.isfinite(number) or number == 0:
        return None
    return float(number)


def _float_from_value(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, dict):
        for key in ("value", "line", "american", "alternateDisplayValue"):
            parsed = _float_from_value(value.get(key))
            if math.isfinite(parsed):
                return parsed
        return float("nan")
    text = str(value).strip().lower().replace("+", "")
    if not text:
        return float("nan")
    if text.startswith(("o", "u")):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _extract_moneyline(team_odds: dict, basis_order: list[str]) -> tuple[float | None, str | None]:
    for basis in basis_order:
        moneyline = (team_odds.get(basis) or {}).get("moneyLine")
        american = _american_from_value(moneyline)
        if american is not None:
            return american, basis
    american = _american_from_value(team_odds.get("moneyLine"))
    if american is not None:
        return american, "top_level"
    return None, None


def _extract_spread_line(team_odds: dict, basis: str | None) -> float:
    if not basis:
        return float("nan")
    return _float_from_value((team_odds.get(basis) or {}).get("pointSpread"))


def _provider_matches(item: dict, provider: str | None) -> bool:
    if not provider:
        return True
    wanted = provider.strip().lower()
    got = str((item.get("provider") or {}).get("name", "")).strip().lower()
    return got == wanted


def _select_odds_item(
    items: list[dict],
    provider: str | None,
    basis_order: list[str],
) -> tuple[dict | None, str | None, str | None, str | None, str | None]:
    for item in items:
        if not _provider_matches(item, provider):
            continue
        home_odds = item.get("homeTeamOdds") or {}
        away_odds = item.get("awayTeamOdds") or {}
        home_ml, home_basis = _extract_moneyline(home_odds, basis_order)
        away_ml, away_basis = _extract_moneyline(away_odds, basis_order)
        if home_ml is not None and away_ml is not None:
            return item, home_ml, away_ml, home_basis, away_basis
    return None, None, None, None, None


def fetch_scoreboard_events(
    session: requests.Session,
    date: pd.Timestamp,
    cache_dir: Path | None,
    sleep_seconds: float,
) -> list[EspnEvent]:
    date_yyyymmdd = date.strftime("%Y%m%d")
    cache_path = _cache_path(cache_dir, "scoreboard", date_yyyymmdd) if cache_dir else None
    data = _get_json(
        session,
        SCOREBOARD_URL.format(date_yyyymmdd=date_yyyymmdd),
        cache_path,
        sleep_seconds,
    )
    events = []
    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        home = away = None
        for competitor in comp.get("competitors", []):
            if competitor.get("homeAway") == "home":
                home = competitor
            elif competitor.get("homeAway") == "away":
                away = competitor
        if not home or not away:
            continue
        event_dt = pd.to_datetime(event.get("date", ""), utc=True, errors="coerce")
        game_time = event_dt.strftime("%H:%M") if pd.notna(event_dt) else ""
        events.append(
            EspnEvent(
                date=date.strftime("%Y-%m-%d"),
                game_time=game_time,
                event_id=str(event.get("id", "")),
                competition_id=str(comp.get("id") or event.get("id", "")),
                home_abbr=str(home.get("team", {}).get("abbreviation", "")),
                away_abbr=str(away.get("team", {}).get("abbreviation", "")),
                home_team=str(home.get("team", {}).get("displayName", "")),
                away_team=str(away.get("team", {}).get("displayName", "")),
            )
        )
    return events


def fetch_event_odds(
    session: requests.Session,
    event: EspnEvent,
    cache_dir: Path | None,
    sleep_seconds: float,
    provider: str | None,
    basis_order: list[str],
) -> dict:
    cache_path = _cache_path(cache_dir, "odds", event.event_id) if cache_dir else None
    data = _get_json(
        session,
        CORE_ODDS_URL.format(event_id=event.event_id, competition_id=event.competition_id),
        cache_path,
        sleep_seconds,
    )
    items = data.get("items", [])
    item, home_ml, away_ml, home_basis, away_basis = _select_odds_item(
        items, provider, basis_order
    )
    provider_info = (item or {}).get("provider") or {}
    home_odds = (item or {}).get("homeTeamOdds") or {}
    away_odds = (item or {}).get("awayTeamOdds") or {}
    return {
        "date": event.date,
        "game_time": event.game_time,
        "event_id": event.event_id,
        "competition_id": event.competition_id,
        "provider_id": provider_info.get("id", ""),
        "provider_name": provider_info.get("name", ""),
        "home_team": event.home_team,
        "away_team": event.away_team,
        "home_abbr": event.home_abbr,
        "away_abbr": event.away_abbr,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "home_price_basis": home_basis,
        "away_price_basis": away_basis,
        "home_run_line": _extract_spread_line(home_odds, home_basis),
        "away_run_line": _extract_spread_line(away_odds, away_basis),
        "total_line": _float_from_value((item or {}).get("overUnder")),
        "over_odds": _american_from_value((item or {}).get("overOdds")),
        "under_odds": _american_from_value((item or {}).get("underOdds")),
        "odds_items_count": len(items),
    }


def collect_espn_odds(
    source_df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    provider: str | None,
    basis_order: list[str],
    cache_dir: Path | None,
    sleep_seconds: float,
) -> pd.DataFrame:
    dates = pd.to_datetime(source_df["date"], errors="coerce").dropna()
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]
    unique_dates = sorted(dates.dt.normalize().unique())

    rows = []
    with requests.Session() as session:
        for date in unique_dates:
            events = fetch_scoreboard_events(session, pd.Timestamp(date), cache_dir, sleep_seconds)
            for event in events:
                rows.append(
                    fetch_event_odds(
                        session,
                        event,
                        cache_dir,
                        sleep_seconds,
                        provider,
                        basis_order,
                    )
                )
    return pd.DataFrame(rows)


def enrich_training_rows(source_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    out = source_df.copy()
    required = {"date", "game_time", "team_abbr", "opp_abbr", "moneyline", "run_line", "total_line"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Source CSV missing required columns: {missing}")

    home_map = {}
    away_map = {}
    for row in odds_df.to_dict("records"):
        home_key = (row["date"], row["game_time"], row["home_abbr"], row["away_abbr"])
        away_key = (row["date"], row["game_time"], row["away_abbr"], row["home_abbr"])
        home_map[home_key] = row
        away_map[away_key] = row

    for idx, row in out.iterrows():
        key = (
            str(row["date"]),
            str(row.get("game_time", "")),
            str(row.get("team_abbr", "")),
            str(row.get("opp_abbr", "")),
        )
        if key in home_map:
            odds = home_map[key]
            out.at[idx, "moneyline"] = odds["home_moneyline"]
            out.at[idx, "run_line"] = odds["home_run_line"]
            out.at[idx, "total_line"] = odds["total_line"]
        elif key in away_map:
            odds = away_map[key]
            out.at[idx, "moneyline"] = odds["away_moneyline"]
            out.at[idx, "run_line"] = odds["away_run_line"]
            out.at[idx, "total_line"] = odds["total_line"]
    return out


def _coverage_report(odds_df: pd.DataFrame, enriched: pd.DataFrame | None = None) -> dict:
    games = int(len(odds_df))
    complete_ml = odds_df["home_moneyline"].notna() & odds_df["away_moneyline"].notna()
    report = {
        "games": games,
        "complete_moneyline_games": int(complete_ml.sum()),
        "complete_moneyline_game_share": float(complete_ml.mean()) if games else 0.0,
    }
    if enriched is not None:
        moneyline = pd.to_numeric(enriched["moneyline"], errors="coerce")
        report.update(
            {
                "source_rows": int(len(enriched)),
                "enriched_moneyline_rows": int(moneyline.notna().sum()),
                "enriched_moneyline_row_share": float(moneyline.notna().mean())
                if len(enriched)
                else 0.0,
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--odds-output", type=Path, default=DEFAULT_ODDS_OUTPUT)
    parser.add_argument("--enriched-output", type=Path)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--provider", help="Optional provider name filter, e.g. 'ESPN BET'.")
    parser.add_argument(
        "--basis-order",
        default="close",
        help="Comma-separated price basis preference.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"Source CSV not found: {args.source}")

    source_df = pd.read_csv(args.source, low_memory=False)
    basis_order = [part.strip() for part in args.basis_order.split(",") if part.strip()]
    cache_dir = None if args.no_cache else args.cache_dir
    odds_df = collect_espn_odds(
        source_df=source_df,
        start_date=args.start_date,
        end_date=args.end_date,
        provider=args.provider,
        basis_order=basis_order,
        cache_dir=cache_dir,
        sleep_seconds=args.sleep,
    )
    enriched = enrich_training_rows(source_df, odds_df)
    print(json.dumps(_coverage_report(odds_df, enriched), indent=2))

    if args.dry_run:
        return

    args.odds_output.parent.mkdir(parents=True, exist_ok=True)
    odds_df.to_csv(args.odds_output, index=False)
    print(f"Wrote odds backfill: {args.odds_output}")
    if args.enriched_output:
        args.enriched_output.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(args.enriched_output, index=False)
        print(f"Wrote enriched source CSV: {args.enriched_output}")


if __name__ == "__main__":
    main()
