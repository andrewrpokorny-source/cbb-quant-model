"""Fetch settled Kalshi CBB positions and log them to betting_history.csv."""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from kalshi.client import KalshiClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BETTING_HISTORY = os.path.join(BASE_DIR, "betting_history.csv")
SYNC_STATE_FILE = os.path.join(BASE_DIR, ".kalshi_sync_state.json")

CSV_HEADERS = [
    "date", "platform", "game", "bet_type", "line", "odds",
    "wager", "result", "payout", "profit", "bet_id", "league",
]

CBB_PREFIXES = ("KXNCAAMB", "KXNCAAWB")


def _league_from_ticker(ticker: str) -> str:
    if ticker.startswith("KXNCAAWB"):
        return "womens"
    return "mens"


def _bet_type_from_ticker(ticker: str) -> str:
    upper = ticker.upper()
    if "SPREAD" in upper:
        return "spread"
    if "GAME" in upper:
        return "game"
    if "TOTAL" in upper:
        return "total"
    return "other"


def _clean_game_title(title: str) -> str:
    """Extract a clean game name from a Kalshi market title."""
    # Titles look like "Team A vs Team B: Spread -5.5" or similar
    # Strip trailing colon-delimited qualifier
    parts = title.split(":")
    return parts[0].strip()


def _read_existing_rows(csv_path: str) -> list[dict]:
    """Read all rows from the CSV."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _existing_bet_ids(rows: list[dict]) -> set[str]:
    """Extract non-empty bet_id values from rows."""
    return {r.get("bet_id", "").strip() for r in rows if r.get("bet_id", "").strip()}


def _find_pending_kalshi_match(
    rows: list[dict], ticker: str, wager: float, title: str = "",
    side: str = "",
) -> int | None:
    """Find a pending Kalshi row that matches this settlement.

    Kalshi bets logged via share URLs don't set bet_id, so we match on
    platform=Kalshi + result=pending + approximate wager, plus discriminators
    extracted from the ticker abbreviation, market title spread value, and
    the YES/NO side.  Returns the row index, or None if no unique match.
    """
    # Extract team abbreviation from ticker tail (e.g. "SMC8" -> "SMC")
    tail = ticker.rsplit("-", 1)[-1] if "-" in ticker else ""
    ticker_team = re.match(r"([A-Z]+)", tail)
    ticker_abbr = ticker_team.group(1).upper() if ticker_team else ""

    # Extract spread number from market title (e.g. "... wins by over 8.5 Points")
    spread_match = re.search(r"(\d+\.?\d*)\s+Points", title, re.IGNORECASE)
    title_spread = float(spread_match.group(1)) if spread_match else None

    side_upper = side.upper()

    # First pass: find pending Kalshi rows matching wager
    wager_matches: list[int] = []
    for i, row in enumerate(rows):
        if row.get("result", "").strip().lower() != "pending":
            continue
        if row.get("platform", "").strip().upper() != "KALSHI":
            continue
        try:
            row_wager = round(float(row.get("wager", 0)), 2)
        except (TypeError, ValueError):
            continue
        if abs(row_wager - wager) > 0.02:
            continue
        wager_matches.append(i)

    if not wager_matches:
        return None
    if len(wager_matches) == 1:
        return wager_matches[0]

    # Multiple wager matches -- narrow by YES/NO side (word boundary match
    # to avoid "NO" matching inside team names like "NORTHWESTERN")
    if side_upper:
        side_re = re.compile(r"\b" + re.escape(side_upper) + r"\b")
        side_matches = [
            i for i in wager_matches
            if side_re.search(rows[i].get("line", "").upper())
        ]
        if len(side_matches) == 1:
            return side_matches[0]
        if side_matches:
            wager_matches = side_matches

    # Narrow by ticker abbreviation
    if ticker_abbr and len(wager_matches) > 1:
        abbr_matches = [
            i for i in wager_matches
            if ticker_abbr in rows[i].get("line", "").upper()
        ]
        if len(abbr_matches) == 1:
            return abbr_matches[0]
        if abbr_matches:
            wager_matches = abbr_matches

    # Narrow by spread number from title
    if title_spread is not None and len(wager_matches) > 1:
        spread_matches = []
        for i in wager_matches:
            row_spread = re.search(r"(\d+\.?\d*)", rows[i].get("line", ""))
            if row_spread and abs(float(row_spread.group(1)) - title_spread) < 0.1:
                spread_matches.append(i)
        if len(spread_matches) == 1:
            return spread_matches[0]

    # Could not disambiguate
    return None


def _ensure_csv_headers(csv_path: str):
    """Create CSV with headers if missing, or migrate to add league column."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header is None:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        return

    missing = [col for col in CSV_HEADERS if col not in header]
    if not missing:
        return

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    defaults = {"bet_id": "", "league": ""}
    tmp_path = csv_path + ".migrate_tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            for col in missing:
                row.setdefault(col, defaults.get(col, ""))
            writer.writerow({h: row.get(h, "") for h in CSV_HEADERS})
    os.replace(tmp_path, csv_path)
    print(f"Migrated CSV: added {missing} column(s) to {len(rows)} rows")


def _reconstruct_line(title: str, side: str) -> str:
    """Best-effort line reconstruction from market title + side."""
    # e.g. "Duke vs UNC: Duke -5.5" -> "Duke -5.5" for YES side
    # For game markets, side is YES (home team wins) or NO
    parts = title.split(":")
    if len(parts) >= 2:
        qualifier = parts[-1].strip()

        # Game/winner market -- extract teams from the base title
        if qualifier.lower() == "winner":
            team_parts = re.split(r"\s+(?:at|vs\.?)\s+", parts[0].strip(), flags=re.IGNORECASE)
            if len(team_parts) == 2:
                if side == "YES":
                    return f"{team_parts[1].strip()} ML"
                return f"{team_parts[0].strip()} ML"

        # Spread markets: qualifier is like "Team -5.5"
        if side == "YES":
            return qualifier
        # NO side on a spread means the opposite team at the flipped line.
        # e.g. title "Duke vs UNC: Duke -5.5", NO side -> "UNC +5.5"
        m = re.search(r"(.+?)\s+([+-]?\d+\.?\d*)\s*$", qualifier)
        if m:
            team, spread_val = m.group(1), float(m.group(2))
            if spread_val == 0:
                return qualifier
            # Find the opposite team from the base title
            team_parts = re.split(r"\s+(?:at|vs\.?)\s+", parts[0].strip(), flags=re.IGNORECASE)
            if len(team_parts) == 2:
                opp = team_parts[1].strip() if team.strip().lower() == team_parts[0].strip().lower() else team_parts[0].strip()
                return f"{opp} {-spread_val:+.1f}"
            return f"{team} {-spread_val:+.1f}"
        return f"{qualifier} (NO)"
    # Game market -- no colon; try "Team A at/vs Team B Winner?"
    m = re.match(r"(.+?)\s+(?:at|vs\.?)\s+(.+?)(?:\s+Winner\??)?$", title, re.IGNORECASE)
    if m:
        if side == "YES":
            return f"{m.group(2).strip()} ML"
        return f"{m.group(1).strip()} ML"
    return f"{side} side"


def _parse_date(settled_time: str) -> str:
    if not settled_time:
        return ""
    try:
        dt = datetime.fromisoformat(settled_time.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return settled_time[:10] if len(settled_time) >= 10 else ""


def _result_from_profit(profit: float) -> str:
    if profit > 0:
        return "win"
    if profit < 0:
        return "loss"
    return "void"


def _parse_settlement(s: dict) -> list[dict]:
    """Parse a single Kalshi settlement into one or more row-ready dicts.

    When only one side was traded, returns a single entry.  When both YES
    and NO were filled, returns two entries with revenue assigned by market
    outcome: the winning side receives $1/contract and the losing side $0.
    """
    yes_count = s.get("yes_count", 0) or 0
    no_count = s.get("no_count", 0) or 0
    yes_cost = s.get("yes_total_cost", 0) or 0
    no_cost = s.get("no_total_cost", 0) or 0
    revenue = s.get("revenue", 0) or 0
    market_result = s.get("market_result", "")  # "yes", "no", or "all_no" / "all_yes"
    date_str = _parse_date(s.get("settled_time", ""))

    if yes_count == 0 and no_count == 0:
        return []

    # Single side -- common case
    if yes_count > 0 and no_count == 0:
        wager = yes_cost / 100
        payout = revenue / 100
        profit = round(payout - wager, 2)
        return [{"side": "YES", "wager": wager, "payout": payout,
                 "profit": profit, "result": _result_from_profit(profit), "date": date_str}]

    if no_count > 0 and yes_count == 0:
        wager = no_cost / 100
        payout = revenue / 100
        profit = round(payout - wager, 2)
        return [{"side": "NO", "wager": wager, "payout": payout,
                 "profit": profit, "result": _result_from_profit(profit), "date": date_str}]

    # Both sides filled -- assign revenue by market outcome.
    # In a binary Kalshi market, the winning side pays $1/contract and
    # the losing side pays $0.
    yes_won = market_result.lower() in ("yes", "all_yes")
    yes_rev_cents = yes_count * 100 if yes_won else 0
    no_rev_cents = no_count * 100 if not yes_won else 0

    entries = []
    for side, cost, rev in [("YES", yes_cost, yes_rev_cents), ("NO", no_cost, no_rev_cents)]:
        wager = cost / 100
        payout = rev / 100
        profit = round(payout - wager, 2)
        entries.append({"side": side, "wager": wager, "payout": payout,
                        "profit": profit, "result": _result_from_profit(profit), "date": date_str})
    return entries


def _read_sync_ts() -> int | None:
    """Read the last-synced epoch timestamp, or None if never synced."""
    if not os.path.exists(SYNC_STATE_FILE):
        return None
    try:
        with open(SYNC_STATE_FILE) as f:
            return json.load(f).get("last_sync_ts")
    except (json.JSONDecodeError, OSError):
        return None


def _write_sync_ts(ts: int):
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump({"last_sync_ts": ts}, f)


def settle_to_csv(days: int = 30, dry_run: bool = False) -> dict:
    """Fetch Kalshi CBB settlements and log new ones to betting_history.csv.

    Uses a persisted sync timestamp so that settlements are never missed
    regardless of how long between runs.  The ``days`` parameter is only
    used as a fallback when no prior sync state exists.

    Returns dict with keys: logged (list[dict]), settled (int), skipped (int),
    error (str|None).  ``settled`` counts pending rows that were updated in
    place rather than appended.
    """
    client = KalshiClient()

    if not client.private_key or not client.api_key:
        return {"logged": [], "settled": 0, "skipped": 0, "error": "Kalshi credentials not configured"}

    now_ts = int(datetime.now(timezone.utc).timestamp())
    last_sync = _read_sync_ts()
    if last_sync is not None:
        # Overlap by 1 day to catch any edge-case timing gaps
        min_ts = last_sync - 86400
    else:
        min_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    try:
        settlements = client.get_settlements(min_ts=min_ts)
    except RuntimeError as e:
        return {"logged": [], "settled": 0, "skipped": 0, "error": str(e)}

    cbb = [s for s in settlements if any(s.get("ticker", "").startswith(p) for p in CBB_PREFIXES)]
    if not cbb:
        if not dry_run:
            _write_sync_ts(now_ts)
        return {"logged": [], "settled": 0, "skipped": 0, "error": None}

    _ensure_csv_headers(BETTING_HISTORY)
    existing_rows = _read_existing_rows(BETTING_HISTORY)
    existing_ids = _existing_bet_ids(existing_rows)

    new_rows = []
    settled_count = 0
    skipped = 0
    rows_modified = False

    for s in cbb:
        ticker = s.get("ticker", "")
        if ticker in existing_ids:
            skipped += 1
            continue

        entries = _parse_settlement(s)
        if not entries:
            skipped += 1
            continue

        market = client.get_market(ticker)
        title = market.get("title", ticker)

        for parsed in entries:
            # For dual-side fills, use side-specific bet_id suffix
            bet_id = ticker if len(entries) == 1 else f"{ticker}:{parsed['side']}"
            if bet_id in existing_ids:
                skipped += 1
                continue

            # Check for a pending Kalshi bet that matches this settlement
            pending_idx = _find_pending_kalshi_match(
                existing_rows, ticker, parsed["wager"], title, parsed["side"],
            )
            if pending_idx is not None:
                existing_rows[pending_idx]["result"] = parsed["result"]
                existing_rows[pending_idx]["payout"] = f"{parsed['payout']:.2f}"
                existing_rows[pending_idx]["profit"] = f"{parsed['profit']:.2f}"
                existing_rows[pending_idx]["bet_id"] = bet_id
                if not existing_rows[pending_idx].get("league", "").strip():
                    existing_rows[pending_idx]["league"] = _league_from_ticker(ticker)
                settled_count += 1
                rows_modified = True
                continue

            row = {
                "date": parsed["date"],
                "platform": "Kalshi",
                "game": _clean_game_title(title),
                "bet_type": _bet_type_from_ticker(ticker),
                "line": _reconstruct_line(title, parsed["side"]),
                "odds": "",
                "wager": f"{parsed['wager']:.2f}",
                "result": parsed["result"],
                "payout": f"{parsed['payout']:.2f}",
                "profit": f"{parsed['profit']:.2f}",
                "bet_id": bet_id,
                "league": _league_from_ticker(ticker),
            }
            new_rows.append(row)

    if not dry_run:
        if rows_modified:
            tmp_path = BETTING_HISTORY + ".settle_tmp"
            with open(tmp_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow({h: r.get(h, "") for h in CSV_HEADERS})
            os.replace(tmp_path, BETTING_HISTORY)

        if new_rows:
            with open(BETTING_HISTORY, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                for row in new_rows:
                    writer.writerow(row)

        _write_sync_ts(now_ts)

    return {"logged": new_rows, "settled": settled_count, "skipped": skipped, "error": None}


def settle(days: int, dry_run: bool):
    """CLI wrapper around settle_to_csv with printed output."""
    result = settle_to_csv(days=days, dry_run=dry_run)

    if result["error"]:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    logged = result["logged"]
    settled = result["settled"]
    skipped = result["skipped"]
    total = len(logged) + settled + skipped

    if total == 0:
        print("No CBB settlements found.")
        return

    for row in logged:
        prefix = "[DRY RUN] " if dry_run else ""
        print(f"  {prefix}ADD: {row['bet_id']} | {row['game']} | {row['result']} | "
              f"wager=${float(row['wager']):.2f} profit=${float(row['profit']):+.2f}")

    if settled:
        print(f"  Updated {settled} pending bet(s) with settlement results")
    if skipped:
        print(f"  Skipped {skipped} already-logged settlement(s)")

    if dry_run:
        print(f"\nDry run complete. {len(logged)} new, {settled} updated.")
    elif logged or settled:
        print(f"\nDone. {len(logged)} appended, {settled} pending updated.")


def main():
    parser = argparse.ArgumentParser(description="Settle Kalshi CBB bets to betting_history.csv")
    parser.add_argument("--days", type=int, default=30, help="Look back N days (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be logged without writing")
    args = parser.parse_args()
    settle(args.days, args.dry_run)


if __name__ == "__main__":
    main()
