"""Fetch settled Kalshi CBB positions and log them to betting_history.csv."""

import argparse
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

from kalshi.client import KalshiClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BETTING_HISTORY = os.path.join(BASE_DIR, "data/betting_history.csv")
SYNC_STATE_FILE = os.path.join(BASE_DIR, ".kalshi_sync_state.json")

CSV_HEADERS = [
    "date", "platform", "game", "bet_type", "line", "odds",
    "wager", "result", "payout", "profit", "bet_id", "league",
]

CBB_PREFIXES = ("KXNCAAMB", "KXNCAAWB")
MLB_PREFIXES = ("KXMLB",)
ALL_PREFIXES = CBB_PREFIXES + MLB_PREFIXES


def _league_from_ticker(ticker: str) -> str:
    if ticker.startswith("KXNCAAWB"):
        return "womens"
    if ticker.startswith("KXMLB"):
        return "mlb"
    return "mens"


def _bet_type_from_ticker(ticker: str) -> str:
    upper = ticker.upper()
    if "SPREAD" in upper:
        return "spread"
    if "GAME" in upper:
        return "moneyline"
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
        legacy_path = os.path.join(BASE_DIR, os.path.basename(csv_path))
        if os.path.exists(legacy_path):
            raise FileNotFoundError(
                f"{os.path.basename(csv_path)} found at legacy location ({legacy_path}) "
                f"but not at expected location ({csv_path}). "
                f"Run 'python migrate_data.py' to migrate."
            )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
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


def _game_ml_label(team_parts: list[str], side: str, yes_team: str) -> str:
    """Return '{Team} ML' for a game/winner market given two team names."""
    if yes_team:
        yt = yes_team.lower()
        # Match yes_team to one of the two title teams
        scores = []
        for t in team_parts:
            tl = t.lower()
            if yt == tl or yt in tl or tl in yt:
                scores.append(1)
            else:
                scores.append(0)
        if scores[0] > scores[1]:
            yes_idx = 0
        elif scores[1] > scores[0]:
            yes_idx = 1
        else:
            yes_idx = 1  # fallback to legacy home-team assumption
        if side == "YES":
            return f"{team_parts[yes_idx]} ML"
        return f"{team_parts[1 - yes_idx]} ML"
    # No yes_team provided -- fall back to legacy positional assumption
    if side == "YES":
        return f"{team_parts[1]} ML"
    return f"{team_parts[0]} ML"


def _reconstruct_line(title: str, side: str, yes_team: str = "") -> str:
    """Best-effort line reconstruction from market title + side.

    Args:
        yes_team: The YES-side team name (from market ``yes_sub_title``).
                  Used for game/winner markets to avoid guessing from
                  title position, which breaks when YES != home team.
    """
    # e.g. "Duke vs UNC: Duke -5.5" -> "Duke -5.5" for YES side
    # For game markets, side is YES (yes_team wins) or NO
    parts = title.split(":")
    if len(parts) >= 2:
        qualifier = parts[-1].strip()
        base_title = ":".join(parts[:-1]).strip()

        # Game/winner market -- extract teams from the base title
        if qualifier.lower() == "winner":
            team_parts = re.split(r"\s+(?:at|vs\.?)\s+", base_title, flags=re.IGNORECASE)
            if len(team_parts) == 2:
                return _game_ml_label(team_parts, side, yes_team)

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
            team_parts = re.split(r"\s+(?:at|vs\.?)\s+", base_title, flags=re.IGNORECASE)
            if len(team_parts) == 2:
                team_lower = team.strip().lower()
                t0 = team_parts[0].strip()
                t1 = team_parts[1].strip()
                t0_match = team_lower in t0.lower()
                t1_match = team_lower in t1.lower()
                if t0_match and t1_match:
                    # Both match (e.g. "virginia" in "virginia tech" and "virginia").
                    # The qualifier matches the closer-length team; opposite is the other.
                    if abs(len(team_lower) - len(t1.lower())) < abs(len(team_lower) - len(t0.lower())):
                        opp = t0
                    else:
                        opp = t1
                elif t0_match:
                    opp = t1
                elif t1_match:
                    opp = t0
                else:
                    logger.warning(
                        "Qualifier team %r matches neither base team (%r, %r) in title %r; "
                        "using first base team as fallback",
                        team, t0, t1, title,
                    )
                    opp = t0
                # Strip prefix noise from multi-colon titles (e.g. "NCAA: Duke" -> "Duke")
                if ":" in opp:
                    opp = opp.rsplit(":", 1)[-1].strip()
                return f"{opp} {-spread_val:+.1f}"
            logger.warning(
                "Could not extract opposite team from title %r (split produced %d parts); "
                "falling back to same-team flipped spread",
                title, len(team_parts),
            )
            return f"{team} {-spread_val:+.1f}"
        return f"{qualifier} (NO)"
    # Game market -- no colon; try "Team A at/vs Team B Winner?"
    m = re.match(r"(.+?)\s+(?:at|vs\.?)\s+(.+?)(?:\s+Winner\??)?$", title, re.IGNORECASE)
    if m:
        team_parts = [m.group(1).strip(), m.group(2).strip()]
        return _game_ml_label(team_parts, side, yes_team)
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


def _pnl_from_fills(fills: list[dict], settlement_revenue_cents: int, date_str: str) -> dict:
    """Compute actual P&L from fills + settlement revenue.

    Handles cases where contracts were bought and sold before settlement.
    The settlement API's total_cost fields include costs of sold contracts
    but revenue only includes settlement payouts, not sell proceeds.
    Fills give us the complete picture.
    """
    total_bought_cents = 0
    total_sold_cents = 0
    total_fees_cents = 0
    buy_yes_cents = 0
    buy_no_cents = 0

    for f in fills:
        count = int(float(f.get("count_fp", 0) or 0))
        action = f.get("action", "")
        side = f.get("side", "")

        if side == "yes":
            price_cents = round(float(f.get("yes_price_dollars", 0) or 0) * 100)
        else:
            price_cents = round(float(f.get("no_price_dollars", 0) or 0) * 100)

        cost = count * price_cents
        fee_dollars = float(f.get("fee_cost", 0) or 0)
        fee_cents = round(fee_dollars * 100)

        if action == "buy":
            total_bought_cents += cost
            if side == "yes":
                buy_yes_cents += cost
            else:
                buy_no_cents += cost
        elif action == "sell":
            total_sold_cents += cost

        total_fees_cents += fee_cents

    total_in = total_sold_cents + settlement_revenue_cents
    net_cents = total_in - total_bought_cents - total_fees_cents

    wager = total_bought_cents / 100
    payout = total_in / 100
    profit = round(net_cents / 100, 2)
    side = "NO" if buy_no_cents >= buy_yes_cents else "YES"

    return {"side": side, "wager": wager, "payout": payout,
            "profit": profit, "result": _result_from_profit(profit), "date": date_str}


def _safe_dollars_to_cents(value) -> int | None:
    """Convert a dollar-denomination string to integer cents.

    Returns None if value is None, empty, or not a valid number.
    """
    if value is None:
        return None
    try:
        return round(float(value) * 100)
    except (ValueError, TypeError):
        return None


def _parse_settlement(s: dict, fills: list[dict] | None = None) -> list[dict]:
    """Parse a single Kalshi settlement into one or more row-ready dicts.

    When only one side was traded, returns a single entry.  When both YES
    and NO were filled, uses fills data to compute actual P&L (accounting
    for intermediate sells that the settlement API's total_cost fields
    don't subtract).
    """
    yes_count = int(float(s.get("yes_count_fp") or s.get("yes_count") or 0))
    no_count = int(float(s.get("no_count_fp") or s.get("no_count") or 0))
    # Kalshi API returns costs in dollars (*_dollars fields, string type).
    # Convert to cents for internal math. Fall back to legacy cents fields.
    yes_cost = _safe_dollars_to_cents(s.get("yes_total_cost_dollars"))
    if yes_cost is None:
        yes_cost = int(s.get("yes_total_cost", 0) or 0)
    no_cost = _safe_dollars_to_cents(s.get("no_total_cost_dollars"))
    if no_cost is None:
        no_cost = int(s.get("no_total_cost", 0) or 0)
    # Revenue may also migrate to dollars in the future
    revenue_dollars = _safe_dollars_to_cents(s.get("revenue_dollars"))
    if revenue_dollars is not None:
        revenue = revenue_dollars
    else:
        revenue = int(s.get("revenue", 0) or 0)
    market_result = s.get("market_result", "")
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

    # Both sides traded.  Check whether the revenue matches what we'd
    # expect if every counted winning contract settled normally.  If it
    # does, the counts are trustworthy and we can split into two entries.
    # If not, there were intermediate sells that inflated the counts and
    # we need fills for accurate P&L.
    yes_won = market_result.lower() in ("yes", "all_yes")
    winning_count = yes_count if yes_won else no_count
    expected_revenue = winning_count * 100

    if revenue == expected_revenue:
        # Clean dual-side hold -- split into per-side entries.
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

    # Revenue mismatch -- intermediate sells inflated the counts.
    # Use fills for accurate P&L.
    if fills:
        return [_pnl_from_fills(fills, revenue, date_str)]

    # Fallback when fills unavailable: use settlement revenue directly.
    # This may understate profit when sell proceeds aren't in revenue.
    total_cost = (yes_cost + no_cost) / 100
    payout = revenue / 100
    profit = round(payout - total_cost, 2)
    side = "NO" if no_cost >= yes_cost else "YES"
    return [{"side": side, "wager": total_cost, "payout": payout,
             "profit": profit, "result": _result_from_profit(profit), "date": date_str}]


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

    # Warn if Kalshi API fields have changed again
    if settlements:
        sample = settlements[0]
        expected_fields = {"yes_total_cost_dollars", "no_total_cost_dollars", "yes_count_fp", "revenue"}
        # If revenue_dollars appears, the revenue field has migrated too
        if "revenue_dollars" in sample:
            print("  WARNING: Kalshi API now has 'revenue_dollars' -- revenue field may have migrated")
        missing = expected_fields - set(sample.keys())
        if missing:
            print(f"  WARNING: Kalshi settlement API fields changed -- missing: {missing}")
            print(f"  Available fields: {sorted(sample.keys())}")

    cbb = [s for s in settlements if any(s.get("ticker", "").startswith(p) for p in ALL_PREFIXES)]
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
    parse_failed = 0
    rows_modified = False

    for s in cbb:
        ticker = s.get("ticker", "")
        if ticker in existing_ids:
            skipped += 1
            continue

        # Fetch fills for dual-side settlements (both YES and NO traded)
        yes_count = int(float(s.get("yes_count_fp") or s.get("yes_count") or 0))
        no_count = int(float(s.get("no_count_fp") or s.get("no_count") or 0))
        fills = None
        if yes_count > 0 and no_count > 0:
            try:
                fills = client.get_fills(ticker=ticker)
            except Exception as e:
                logger.warning("Failed to fetch fills for %s: %s", ticker, e)

        entries = _parse_settlement(s, fills=fills)
        if not entries:
            parse_failed += 1
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
                "line": _reconstruct_line(title, parsed["side"], market.get("yes_sub_title", "")),
                "odds": "",
                "wager": f"{parsed['wager']:.2f}",
                "result": parsed["result"],
                "payout": f"{parsed['payout']:.2f}",
                "profit": f"{parsed['profit']:.2f}",
                "bet_id": bet_id,
                "league": _league_from_ticker(ticker),
            }
            new_rows.append(row)

    # Detect possible API schema change: new settlements exist but none parsed
    unseen = len(cbb) - skipped + parse_failed  # tickers not in existing_ids
    error = None
    if parse_failed > 0 and parse_failed == unseen and not new_rows and settled_count == 0:
        sample_keys = sorted(cbb[0].keys()) if cbb else []
        error = (
            f"Possible Kalshi API schema change: {parse_failed}/{len(cbb)} "
            f"new settlements failed to parse. Sample keys: {sample_keys}"
        )

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

    return {"logged": new_rows, "settled": settled_count, "skipped": skipped, "error": error}


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
