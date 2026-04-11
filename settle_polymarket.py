"""Fetch settled Polymarket positions and log them to betting_history.csv.

Uses the Polymarket Gamma API to query closed positions for a wallet address.
Mirrors settle_kalshi.py's approach: parse positions, match to pending bets,
append new rows to the shared CSV ledger.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from polymarket.client import PolymarketClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BETTING_HISTORY = os.path.join(BASE_DIR, "data/betting_history.csv")
SYNC_STATE_FILE = os.path.join(BASE_DIR, ".polymarket_sync_state.json")

CSV_HEADERS = [
    "date", "platform", "game", "bet_type", "line", "odds",
    "wager", "result", "payout", "profit", "bet_id", "league",
]


def _read_sync_ts() -> str | None:
    """Read the last-synced timestamp, or None if never synced."""
    if not os.path.exists(SYNC_STATE_FILE):
        return None
    try:
        with open(SYNC_STATE_FILE) as f:
            return json.load(f).get("last_sync_ts")
    except (json.JSONDecodeError, OSError):
        return None


def _write_sync_ts(ts: str):
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump({"last_sync_ts": ts}, f)


def _ensure_csv():
    """Create the CSV file with headers if it doesn't exist."""
    if not os.path.exists(BETTING_HISTORY):
        os.makedirs(os.path.dirname(BETTING_HISTORY), exist_ok=True)
        with open(BETTING_HISTORY, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


def _existing_bet_ids() -> set[str]:
    """Load all existing bet_id values from the CSV."""
    ids = set()
    if not os.path.exists(BETTING_HISTORY):
        return ids
    with open(BETTING_HISTORY, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row.get("bet_id", "").strip()
            if bid:
                ids.add(bid)
    return ids


def _parse_position(pos: dict) -> dict | None:
    """Parse a single closed Polymarket position into a CSV-ready dict.

    Returns None if the position can't be parsed.
    """
    # Fields per https://docs.polymarket.com/api-reference/core/get-closed-positions-for-a-user
    token_id = pos.get("conditionId") or pos.get("token_id") or pos.get("id", "")
    asset = pos.get("asset", "")  # specific token (YES or NO) traded
    title = pos.get("title") or pos.get("question") or pos.get("name", "")
    outcome = pos.get("outcome") or pos.get("result", "")

    # Cost and payout
    cost = pos.get("cost") or pos.get("avgPrice", 0)
    payout = pos.get("payout") or pos.get("redeemed", 0)
    try:
        cost = float(cost)
        payout = float(payout)
    except (TypeError, ValueError):
        cost = 0.0
        payout = 0.0

    profit = round(payout - cost, 2)
    result = "win" if profit > 0 else ("loss" if profit < 0 else "push")

    # Date
    date_str = ""
    raw_date = pos.get("created_at") or pos.get("date") or pos.get("timestamp", "")
    if raw_date:
        try:
            dt = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = str(raw_date)[:10]

    # Bet type inference from title
    title_lower = title.lower()
    if "spread" in title_lower or "wins by" in title_lower:
        bet_type = "spread"
    elif "total" in title_lower or "over" in title_lower or "under" in title_lower:
        bet_type = "total"
    else:
        bet_type = "game"

    # Side
    side = pos.get("side", "YES").upper()

    return {
        "date": date_str,
        "platform": "Polymarket",
        "game": title,
        "bet_type": bet_type,
        "line": f"{title} {side}",
        "odds": "n/a",
        "wager": round(cost, 2),
        "result": result,
        "payout": round(payout, 2),
        "profit": profit,
        "bet_id": f"poly_{token_id}_{outcome or side}_{asset[-8:]}" if asset else f"poly_{token_id}_{outcome or side}",
        "league": "",
    }


def settle_to_csv(wallet_address: str | None = None, dry_run: bool = False) -> dict:
    """Fetch closed Polymarket positions and append to betting_history.csv.

    Returns summary dict with counts.
    """
    wallet = wallet_address or os.getenv("POLYMARKET_WALLET_ADDRESS")
    if not wallet:
        print("POLYMARKET_WALLET_ADDRESS not set. Cannot settle Polymarket positions.")
        return {"settled": 0, "skipped": 0, "errors": 0}

    proxy = os.getenv("POLYMARKET_PROXY")
    if not proxy:
        print("POLYMARKET_PROXY not set. Cannot reach Polymarket.")
        return {"settled": 0, "skipped": 0, "errors": 0}

    client = PolymarketClient(proxy_url=proxy)
    positions = client.get_closed_positions(wallet)

    if not positions:
        print("No closed Polymarket positions found.")
        return {"settled": 0, "skipped": 0, "errors": 0}

    _ensure_csv()
    existing_ids = _existing_bet_ids()

    settled = 0
    skipped = 0
    errors = 0

    rows_to_write = []
    for pos in positions:
        try:
            row = _parse_position(pos)
            if row is None:
                errors += 1
                continue

            if row["bet_id"] in existing_ids:
                skipped += 1
                continue

            rows_to_write.append(row)
            settled += 1
        except Exception as e:
            logger.warning("Failed to parse Polymarket position: %s", e)
            errors += 1

    if rows_to_write and not dry_run:
        with open(BETTING_HISTORY, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            for row in rows_to_write:
                writer.writerow(row)

        _write_sync_ts(datetime.now(timezone.utc).isoformat())
        print(f"Polymarket: {settled} settled, {skipped} already recorded, {errors} errors")
    elif dry_run:
        print(f"Polymarket (dry run): {settled} would be settled, {skipped} already recorded")
        for row in rows_to_write:
            print(f"  {row['date']} | {row['game'][:50]} | {row['result']} | ${row['profit']:.2f}")
    else:
        print("No new Polymarket positions to settle.")

    return {"settled": settled, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Settle Polymarket positions")
    parser.add_argument("--wallet", help="Polygon wallet address (overrides env)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    settle_to_csv(wallet_address=args.wallet, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
