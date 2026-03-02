"""Fetch settled Kalshi CBB positions and log them to betting_history.csv."""

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from kalshi.client import KalshiClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BETTING_HISTORY = os.path.join(BASE_DIR, "betting_history.csv")

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


def _find_pending_kalshi_match(rows: list[dict], ticker: str, wager: float) -> int | None:
    """Find a pending Kalshi row that matches this settlement.

    Kalshi bets logged via share URLs don't set bet_id, so we match on
    platform=Kalshi + result=pending + approximate wager. The ticker's
    embedded team abbreviation is also checked against the line field.
    Returns the row index, or None.
    """
    # Extract team abbreviation from ticker tail (e.g. "SMC8" -> "SMC")
    tail = ticker.rsplit("-", 1)[-1] if "-" in ticker else ""
    ticker_team = re.match(r"([A-Z]+)", tail)
    ticker_abbr = ticker_team.group(1).upper() if ticker_team else ""

    candidates: list[int] = []
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
        # If we have a ticker abbreviation, check it appears in the line
        if ticker_abbr:
            row_line = row.get("line", "").upper()
            if ticker_abbr in row_line:
                candidates.append(i)
                continue
        # No ticker abbr or no line match -- still a wager-only candidate
        candidates.append(i)

    return candidates[0] if len(candidates) == 1 else None


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
        # Spread markets: qualifier is like "Team -5.5"
        if side == "YES":
            return qualifier
        # NO side on a spread means the opposite
        # Try to flip the spread sign
        m = re.search(r"(.+?)\s+([+-]?\d+\.?\d*)\s*$", qualifier)
        if m:
            team, spread_val = m.group(1), float(m.group(2))
            return f"{team} {-spread_val:+.1f}" if spread_val != 0 else qualifier
        return f"{qualifier} (NO)"
    # Game market -- no spread in title
    return f"{side} side"


def _parse_settlement(s: dict) -> dict | None:
    """Parse a single Kalshi settlement into a row-ready dict.

    Uses combined cost/revenue across both YES and NO sides so that
    positions with fills on both sides are handled correctly.
    """
    yes_count = s.get("yes_count", 0) or 0
    no_count = s.get("no_count", 0) or 0
    yes_cost = s.get("yes_total_cost", 0) or 0
    no_cost = s.get("no_total_cost", 0) or 0
    revenue = s.get("revenue", 0) or 0

    if yes_count == 0 and no_count == 0:
        return None

    # Combine both sides for net cost and determine primary side for line display
    wager_cents = yes_cost + no_cost
    if yes_count > 0 and no_count > 0:
        side = "YES" if yes_count >= no_count else "NO"
    elif yes_count > 0:
        side = "YES"
    else:
        side = "NO"

    wager = wager_cents / 100
    payout = revenue / 100
    profit = round(payout - wager, 2)

    if profit > 0:
        result = "win"
    elif profit < 0:
        result = "loss"
    else:
        result = "void"

    settled_time = s.get("settled_time", "")
    if settled_time:
        try:
            dt = datetime.fromisoformat(settled_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = settled_time[:10] if len(settled_time) >= 10 else ""
    else:
        date_str = ""

    return {
        "side": side,
        "wager": wager,
        "payout": payout,
        "profit": profit,
        "result": result,
        "date": date_str,
    }


def settle_to_csv(days: int = 30, dry_run: bool = False) -> dict:
    """Fetch Kalshi CBB settlements and log new ones to betting_history.csv.

    Returns dict with keys: logged (list[dict]), settled (int), skipped (int),
    error (str|None).  ``settled`` counts pending rows that were updated in
    place rather than appended.
    """
    client = KalshiClient()

    if not client.private_key or not client.api_key:
        return {"logged": [], "settled": 0, "skipped": 0, "error": "Kalshi credentials not configured"}

    min_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    settlements = client.get_settlements(min_ts=min_ts)

    cbb = [s for s in settlements if any(s.get("ticker", "").startswith(p) for p in CBB_PREFIXES)]
    if not cbb:
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

        parsed = _parse_settlement(s)
        if parsed is None:
            skipped += 1
            continue

        market = client.get_market(ticker)
        title = market.get("title", ticker)

        # Check for a pending Kalshi bet that matches this settlement
        pending_idx = _find_pending_kalshi_match(existing_rows, ticker, parsed["wager"])
        if pending_idx is not None:
            # Update the existing pending row in place
            existing_rows[pending_idx]["result"] = parsed["result"]
            existing_rows[pending_idx]["payout"] = f"{parsed['payout']:.2f}"
            existing_rows[pending_idx]["profit"] = f"{parsed['profit']:.2f}"
            existing_rows[pending_idx]["bet_id"] = ticker
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
            "bet_id": ticker,
            "league": _league_from_ticker(ticker),
        }
        new_rows.append(row)

    if not dry_run:
        if rows_modified:
            # Rewrite CSV with updated pending rows
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
    parser.add_argument("--days", type=int, default=7, help="Look back N days (default: 7)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be logged without writing")
    args = parser.parse_args()
    settle(args.days, args.dry_run)


if __name__ == "__main__":
    main()
