"""
Telegram bot for logging bets and settling them.

Screenshot a bet slip -> share to this bot -> auto-parsed and logged.
Commands: /settle, /pending, /today, /record

Usage:
    python telegram_bot.py
"""

import os
import re
import sys
import csv
import glob
import json
import errno
import fcntl
import asyncio
import base64
import shutil
import tempfile
import logging
import logging.handlers
import functools
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import urlparse, parse_qs

import httpx
import pytz
import pandas as pd
from dotenv import load_dotenv
from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from ocrmac import ocrmac

from betting import VALUE_RATINGS
from kalshi.market_mapper import normalize_team_name
from settle_bets import settle_pending_bets
from settle_kalshi import settle_to_csv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BETTING_HISTORY = os.path.join(BASE_DIR, "betting_history.csv")
DAILY_PREDICTIONS = os.path.join(BASE_DIR, "daily_predictions.csv")
PERF_FILE = os.path.join(BASE_DIR, "performance_log.csv")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Guard against concurrent /settle invocations (e.g. double-tap)
_settle_lock = asyncio.Lock()

# User authorization: comma-separated list of allowed Telegram user IDs
_allowed_users_str = os.getenv("TELEGRAM_ALLOWED_USERS", "")
ALLOWED_USER_IDS = set(int(uid.strip()) for uid in _allowed_users_str.split(",") if uid.strip())


def _parse_edge_pct(s):
    """Parse an edge string like '+4.2%' into a float (4.2). Returns 0.0 on failure."""
    try:
        val = float(str(s).replace("%", "").replace("+", ""))
        return val if val == val else 0.0  # NaN != NaN
    except (ValueError, TypeError):
        return 0.0


def authorized_only(func):
    """Decorator to restrict access to authorized users only."""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id if update.effective_user else None
        if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
            logger.warning(f"Unauthorized access attempt from user {user_id}")
            await update.message.reply_text("Unauthorized. Contact the bot owner for access.")
            return
        return await func(update, context)
    return wrapper

LOG_FILE = os.path.join(BASE_DIR, "telegram_bot.log")
PARSE_AUDIT_FILE = os.getenv("BET_PARSE_AUDIT_FILE", os.path.join(BASE_DIR, "telegram_parse_audit.jsonl"))
MAX_AUDIT_TEXT_CHARS = 12000


class _RedactTokenFilter(logging.Filter):
    """Strip bot tokens from log output (covers both msg and args)."""
    _pattern = re.compile(r"/bot\d+:[A-Za-z0-9_-]+/")

    def filter(self, record):
        # Format eagerly so tokens in args are also redacted
        if record.args:
            record.msg = str(record.msg) % record.args
            record.args = None
        record.msg = self._pattern.sub("/bot***REDACTED***/", str(record.msg))
        return True


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
        ),
    ],
)
for handler in logging.root.handlers:
    handler.addFilter(_RedactTokenFilter())
logger = logging.getLogger(__name__)

# CSV headers for betting_history.csv
CSV_HEADERS = ["date", "platform", "game", "bet_type", "line", "odds", "wager", "result", "payout", "profit", "bet_id", "league"]

# Short team abbreviations that are valid (not junk OCR text)
VALID_SHORT_TEAMS = {
    "TCU", "USC", "LSU", "SMU", "UCF", "UNC", "UAB", "FIU", "BYU", "UIC",
    "UTSA", "UTEP", "UNLV", "ETSU", "NJIT", "UNO", "URI", "UNI", "SIUE",
    "UMBC", "LIU", "SIU", "NIU", "WKU", "FAU", "FDU", "UNCW", "VMI",
}

# Known junk lines from DraftKings/FanDuel UI that OCR picks up.
# NOTE: Do NOT include platform names here -- they're needed for platform detection.
JUNK_LINES = {
    "my bets", "betting groups", "my pools", "open", "live", "settled",
    "won", "lost", "share", "the crown is yours", "pm", "am",
    "sportsbook", "final", "spread", "straight", "parlay",
    "won on fanduel", "returned", "finished", "all sports", "rewards",
    "home", "account", "saved", "live now", "spread betting", "total wager",
}

# Regex patterns for junk OCR lines
JUNK_PATTERNS = [
    re.compile(r"^\d{1,2}:\d{2}\s*(AM|PM)?$", re.IGNORECASE),  # timestamps
    re.compile(r"^Bet ID:\s", re.IGNORECASE),  # DK bet IDs
    re.compile(r"^BET ID:\s", re.IGNORECASE),  # FD bet IDs
    re.compile(r"^Placed:\s", re.IGNORECASE),  # "Placed:" lines
    re.compile(r"^PLACED:\s", re.IGNORECASE),  # FD placed lines
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),  # dates like 1/30/26
    re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}", re.IGNORECASE),
    re.compile(r"^\d[\d\s]*$"),  # score lines (just digits and spaces)
    re.compile(r"^\$\d+\.\d+\s*\+\s*$"),  # balance display with + sign like "$12.10 +"
    re.compile(r"^\$\d{2,}\.\d+$"),  # large balances like "$42.44" (2+ digits before decimal)
    re.compile(r"^[•+]\s*Share", re.IGNORECASE),  # share buttons
    re.compile(r"^\d+\s+Share", re.IGNORECASE),  # "4 Share"
]

# FanDuel settled-slip regexes (applied to raw OCR text before cleaning)
FD_BET_ID_RE = re.compile(r"BET ID:\s*(\S+)", re.IGNORECASE)
FD_GAME_DATE_RE = re.compile(
    r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2}),\s*\d{1,2}:\d{2}(?:AM|PM)\s*ET",
    re.IGNORECASE,
)

# Cross-screenshot dedup for FanDuel BET IDs within a single bot session.
# Not persisted across restarts -- CSV-level dedup in append_bet() provides
# the durable duplicate check.
_SEEN_FD_BET_IDS: set[str] = set()

# Common OCR misreads to correct
OCR_CORRECTIONS = {
    "lowa": "Iowa",
}


def _clean_ocr_text(lines: list[str]) -> list[str]:
    """Remove known junk lines from OCR output before parsing."""
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip known junk (case-insensitive)
        if stripped.lower() in JUNK_LINES:
            continue
        # Skip regex-matched junk
        if any(p.match(stripped) for p in JUNK_PATTERNS):
            continue
        # Skip short ALL-CAPS abbreviations (2-5 chars) unless valid team name
        if stripped.isupper() and 2 <= len(stripped) <= 5 and stripped not in VALID_SHORT_TEAMS:
            continue
        # Apply OCR corrections
        for wrong, right in OCR_CORRECTIONS.items():
            if wrong in stripped:
                stripped = stripped.replace(wrong, right)
        cleaned.append(stripped)
    return cleaned


def _detect_fd_settled(raw_text: str) -> bool:
    """Detect whether raw OCR text is from a FanDuel settled screenshot."""
    lower = raw_text.lower()
    if "won on fanduel" in lower or "lost on fanduel" in lower:
        return True
    if re.search(r"placed:\s*\d+/\d+/\d{4}", lower):
        return True
    # "RETURNED" appears as a standalone line on FanDuel void slips
    if re.search(r"^\s*returned\s*$", lower, re.MULTILINE) and "sportsbook" in lower:
        return True
    # "Finished" format: settled tab with scores instead of WON/LOST banners
    if re.search(r"^\s*finished\s*$", lower, re.MULTILINE):
        return True
    return False


def ensure_csv_exists():
    """Create betting_history.csv with headers if it doesn't exist.

    Also migrates existing CSVs that lack the bet_id column.
    """
    if not os.path.exists(BETTING_HISTORY):
        with open(BETTING_HISTORY, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        return

    # Migrate: add missing columns (bet_id, league)
    with open(BETTING_HISTORY, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header is None:
        return

    missing = [col for col in CSV_HEADERS if col not in header]
    if not missing:
        return

    with open(BETTING_HISTORY, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    logger.info("Migrating betting_history.csv: adding %s column(s) (%d rows)", missing, len(rows))
    tmp_path = BETTING_HISTORY + ".migrate_tmp"
    defaults = {"bet_id": "", "league": ""}
    try:
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for row in rows:
                for col in missing:
                    row.setdefault(col, defaults.get(col, ""))
                writer.writerow({h: row.get(h, "") for h in CSV_HEADERS})
        os.replace(tmp_path, BETTING_HISTORY)
        logger.info("Migration complete: %s column(s) added", missing)
    except Exception:
        logger.exception("CSV migration failed -- original file preserved")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _normalize_line(line_str: str) -> str:
    """Normalize a bet line field for duplicate comparison.

    Handles embedded newlines from garbled OCR by extracting the actual
    spread line (e.g. 'Team +/-X.X') from the end of multi-line strings.
    """
    line_str = str(line_str).strip()
    if "\n" in line_str:
        # Search from end for the actual spread line
        spread_re = re.compile(r"^(.+?)\s+[+-]\d+\.?\d*$")
        for part in reversed(line_str.split("\n")):
            part = part.strip()
            if spread_re.match(part):
                return part.upper()
        return line_str.split("\n")[-1].strip().upper()
    return line_str.upper()


def _normalize_game(game_str: str) -> str:
    """Normalize game strings for safer matching."""
    game = str(game_str or "").strip()
    if not game or game.lower() in {"nan", "none"}:
        return ""
    game = re.sub(r"\s+(?:vs\.?|@|at)\s+", " vs ", game, flags=re.IGNORECASE)
    game = re.sub(r"\s+", " ", game).strip()
    return game.upper()


def _normalize_wager(wager_val) -> str:
    """Normalize wager to fixed-point string for key comparison."""
    try:
        return f"{float(str(wager_val).strip()):.2f}"
    except (TypeError, ValueError):
        return str(wager_val).strip()


def _normalize_odds(odds_val) -> str:
    """Normalize American odds strings (e.g. -110, +120, n/a)."""
    odds = str(odds_val or "").strip().replace(" ", "")
    if not odds or odds.lower() in {"n/a", "nan", "none"}:
        return "N/A"
    try:
        odds_int = int(float(odds))
        return f"{odds_int:+d}" if odds_int != 0 else "0"
    except (TypeError, ValueError):
        return odds.upper()


def _bet_identity(row: dict) -> tuple[str, str, str, str, str, str]:
    """Identity key for duplicate detection."""
    return (
        str(row.get("date", "")).strip(),
        str(row.get("platform", "")).strip().upper(),
        _normalize_game(row.get("game", "")),
        _normalize_line(row.get("line", "")),
        _normalize_odds(row.get("odds", "")),
        _normalize_wager(row.get("wager", "")),
    )


def _audit_bet_fields(bet: dict) -> dict:
    """Extract stable bet fields for audit snapshots."""
    try:
        wager = float(bet.get("wager", 0))
    except (TypeError, ValueError):
        wager = bet.get("wager", 0)

    return {
        "platform": str(bet.get("platform", "")),
        "game": str(bet.get("game", "")),
        "bet_type": str(bet.get("bet_type", "")),
        "line": str(bet.get("line", "")),
        "odds": str(bet.get("odds", "")),
        "wager": wager,
        "result": str(bet.get("result", "")),
        "bet_id": str(bet.get("bet_id", "")),
    }


def _sanitize_audit_payload(value):
    """Recursively sanitize and truncate values before JSONL audit writes."""
    if isinstance(value, dict):
        return {k: _sanitize_audit_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_audit_payload(v) for v in value]
    if isinstance(value, str) and len(value) > MAX_AUDIT_TEXT_CHARS:
        return value[:MAX_AUDIT_TEXT_CHARS] + "... [truncated]"
    return value


def _append_parse_audit(entry: dict):
    """Append a parse/selection event to the sidecar JSONL audit log."""
    eastern = pytz.timezone("US/Eastern")
    payload = _sanitize_audit_payload(dict(entry))
    payload["timestamp"] = datetime.now(eastern).isoformat()

    try:
        parent = os.path.dirname(PARSE_AUDIT_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)

        line = json.dumps(payload, ensure_ascii=True)
        with open(PARSE_AUDIT_FILE, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (OSError, TypeError, ValueError) as e:
        logger.warning("Could not write parse audit entry: %s", e)


def update_bet_result(bet: dict) -> str | None:
    """
    Update a pending bet with its result.

    Primary match key is platform + line + wager. If multiple candidates match,
    use game/odds/date to disambiguate. If still ambiguous, do not update.
    """
    if not os.path.exists(BETTING_HISTORY):
        return None

    new_platform = str(bet.get("platform", "")).strip().upper()
    new_line = _normalize_line(bet.get("line", ""))
    new_wager = _normalize_wager(bet.get("wager", ""))
    new_game = _normalize_game(bet.get("game", ""))
    new_odds = _normalize_odds(bet.get("odds", ""))
    new_date = str(bet.get("date", "")).strip()
    new_bet_id = str(bet.get("bet_id", "")).strip()

    with open(BETTING_HISTORY, "r+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            reader = list(csv.DictReader(f))

            candidates: list[tuple[int, int, str]] = []
            for i, row in enumerate(reader):
                if str(row.get("result", "")).strip().lower() != "pending":
                    continue

                row_platform = str(row.get("platform", "")).strip().upper()
                if row_platform != new_platform:
                    continue
                if _normalize_line(row.get("line", "")) != new_line:
                    continue
                if _normalize_wager(row.get("wager", "")) != new_wager:
                    continue

                score = 0
                row_game = _normalize_game(row.get("game", ""))
                row_odds = _normalize_odds(row.get("odds", ""))
                row_date = str(row.get("date", "")).strip()

                # BET ID match is a strong signal
                row_bet_id = str(row.get("bet_id", "")).strip()
                if new_bet_id and row_bet_id and new_bet_id == row_bet_id:
                    score += 10

                if new_game and row_game:
                    if new_game == row_game:
                        score += 2
                    elif new_game in row_game or row_game in new_game:
                        score += 1

                if new_odds != "N/A" and row_odds != "N/A" and new_odds == row_odds:
                    score += 1

                if new_date and row_date == new_date:
                    score += 2

                candidates.append((score, i, row_date))

            if not candidates:
                return None

            best_score = max(score for score, _, _ in candidates)
            top = [(i, row_date) for score, i, row_date in candidates if score == best_score]

            if len(top) == 1:
                matched_idx = top[0][0]
            else:
                # Tie-break by most recent date if unique, otherwise avoid wrong update.
                dated = []
                for i, row_date in top:
                    try:
                        dt = datetime.strptime(row_date, "%Y-%m-%d")
                    except (TypeError, ValueError):
                        dt = datetime.min
                    dated.append((dt, i))

                latest_date = max(dt for dt, _ in dated)
                finalists = [i for dt, i in dated if dt == latest_date]
                if len(finalists) == 1:
                    matched_idx = finalists[0]
                else:
                    try:
                        wager_display = f"{float(bet.get('wager', 0)):.2f}"
                    except (TypeError, ValueError):
                        wager_display = str(bet.get("wager", ""))
                    return (
                        f"Ambiguous match for {bet.get('line', '')} ${wager_display} on "
                        f"{bet.get('platform', '')}. Multiple pending bets match."
                    )

            # Update the matched row
            reader[matched_idx]["result"] = bet["result"]
            reader[matched_idx]["payout"] = bet.get("payout", "")
            reader[matched_idx]["profit"] = bet.get("profit", "")

            # Backfill bet_id and game if the existing row has empty values
            if bet.get("bet_id") and not str(reader[matched_idx].get("bet_id", "")).strip():
                reader[matched_idx]["bet_id"] = bet["bet_id"]
            if bet.get("game") and not str(reader[matched_idx].get("game", "")).strip():
                reader[matched_idx]["game"] = bet["game"]

            # Write back while still holding the lock
            f.seek(0)
            f.truncate()
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(reader)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    result = bet["result"]
    profit = float(bet.get("profit", 0) or 0)
    return f"Updated: {bet['line']} -> {result.upper()} ({profit:+.2f}U)"


def _find_matching_pending_bet(partial_bet: dict) -> dict | None:
    """Match a partial settled bet against pending bets in history.

    Matches by platform + wager, then narrows by team name if needed.
    Returns the matched row as a dict, or None.
    """
    if not os.path.exists(BETTING_HISTORY):
        return None

    try:
        df = pd.read_csv(BETTING_HISTORY)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error("Could not read %s for pending bet matching: %s", BETTING_HISTORY, e)
        return None

    pending = df[df["result"] == "pending"]
    if len(pending) == 0:
        return None

    platform = partial_bet["platform"].upper()
    try:
        wager = float(partial_bet["wager"])
    except (ValueError, TypeError, KeyError) as e:
        logger.warning("Invalid wager in partial bet for matching: %s", e)
        return None

    matches = pending[
        (pending["platform"].str.upper() == platform)
        & (abs(pending["wager"].astype(float) - wager) < 0.01)
    ]

    if len(matches) == 0:
        return None

    # Score each candidate by how many extracted teams match it
    teams = partial_bet.get("_teams", [])
    if teams:
        scored: list[tuple[int, int]] = []  # (score, iloc_index)
        for idx in range(len(matches)):
            row = matches.iloc[idx]
            line_upper = str(row.get("line", "")).upper()
            game_upper = str(row.get("game", "")).upper()
            score = sum(
                1
                for team in teams
                if team.upper() in line_upper or team.upper() in game_upper
            )
            scored.append((score, idx))

        best = max(s for s, _ in scored)
        if best > 0:
            winners = [i for s, i in scored if s == best]
            if len(winners) == 1:
                return matches.iloc[winners[0]].to_dict()
            logger.info(
                f"Tied team match ({best} tokens) across {len(winners)} "
                f"candidates, teams={teams}"
            )
            return None

    # No teams extracted (or no team matched any candidate)
    if len(matches) == 1:
        return matches.iloc[0].to_dict()

    logger.info(
        f"Could not narrow pending bet match: {len(matches)} candidates, "
        f"teams={teams}"
    )
    return None


def _find_kalshi_pending_by_spread(bet: dict) -> str | None:
    """Fallback matcher for Kalshi bets where the ticker-derived team name
    doesn't match the pending bet's full team name.

    Matches pending Kalshi bets by wager + spread number + side (YES/NO),
    ignoring the team name portion of the line. If exactly one pending bet
    matches, swaps the line to the pending row's line and calls
    update_bet_result. Returns the update message, or None on 0 or 2+ matches.
    """
    if bet.get("platform", "").upper() != "KALSHI":
        return None

    if not os.path.exists(BETTING_HISTORY):
        return None

    # Extract spread number and side from the settled bet's line
    # Expected format: "SMC -8.5 YES" or "Saint Mary's -8.5 YES"
    line_match = re.search(r"[-+]?\d+(?:\.\d+)?\s+(YES|NO)", bet.get("line", ""), re.IGNORECASE)
    if not line_match:
        return None

    settled_spread_str = re.search(r"([-+]?\d+(?:\.\d+)?)", bet.get("line", ""))
    if not settled_spread_str:
        return None

    settled_spread = float(settled_spread_str.group(1))
    settled_side = line_match.group(1).upper()
    settled_wager = _normalize_wager(bet.get("wager", ""))

    df = pd.read_csv(BETTING_HISTORY)
    pending = df[
        (df["result"] == "pending")
        & (df["platform"].str.upper() == "KALSHI")
    ]

    candidates = []
    for idx, row in pending.iterrows():
        if _normalize_wager(row.get("wager", "")) != settled_wager:
            continue
        row_line = str(row.get("line", ""))
        row_match = re.search(r"([-+]?\d+(?:\.\d+)?)\s+(YES|NO)", row_line, re.IGNORECASE)
        if not row_match:
            continue
        row_spread = float(row_match.group(1))
        row_side = row_match.group(2).upper()
        if abs(row_spread - settled_spread) < 0.01 and row_side == settled_side:
            candidates.append(row)

    if len(candidates) != 1:
        return None

    # Swap the line to the pending row's line so update_bet_result can match
    original_line = bet["line"]
    bet["line"] = str(candidates[0]["line"])
    result = update_bet_result(bet)
    if result is None:
        bet["line"] = original_line  # restore on failure
    return result


def append_bet(bet: dict):
    """Append a bet row to betting_history.csv. Returns None if duplicate.

    Uses file locking to prevent race conditions when multiple messages
    arrive simultaneously.
    """
    ensure_csv_exists()
    eastern = pytz.timezone("US/Eastern")
    today = datetime.now(eastern).strftime("%Y-%m-%d")

    row = {
        "date": bet.get("date", today),
        "platform": bet.get("platform", "Unknown"),
        "game": bet.get("game", ""),
        "bet_type": bet.get("bet_type", "spread"),
        "line": bet.get("line", ""),
        "odds": bet.get("odds", "n/a"),
        "wager": bet.get("wager", 0),
        "result": bet.get("result", "pending"),
        "payout": bet.get("payout", ""),
        "profit": bet.get("profit", ""),
        "bet_id": bet.get("bet_id", ""),
        "league": bet.get("league", ""),
    }
    parse_audit_id = bet.get("_parse_audit_id")

    # Use file locking to prevent race conditions
    with open(BETTING_HISTORY, "r+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Check for duplicates while holding the lock
            new_key = _bet_identity(row)
            new_is_pending = str(row.get("result", "")).strip().lower() == "pending"
            new_bet_id = str(row.get("bet_id", "")).strip()
            reader = csv.DictReader(f)
            for existing in reader:
                # BET ID dedup: same non-empty bet_id is always a duplicate
                existing_bet_id = str(existing.get("bet_id", "")).strip()
                if new_bet_id and existing_bet_id and new_bet_id == existing_bet_id:
                    logger.info(
                        "Duplicate bet_id skipped: %s %s %s",
                        row["line"],
                        row["wager"],
                        row["platform"],
                    )
                    if parse_audit_id:
                        _append_parse_audit(
                            {
                                "event": "ocr_duplicate_skipped",
                                "parse_id": parse_audit_id,
                                "bet": _audit_bet_fields(row),
                            }
                        )
                    return None

                if _bet_identity(existing) != new_key:
                    continue

                existing_is_pending = (
                    str(existing.get("result", "")).strip().lower() == "pending"
                )

                # Allow logging a new pending bet if only a settled version exists.
                # For non-pending rows, dedupe against other non-pending rows.
                if (new_is_pending and existing_is_pending) or (
                    not new_is_pending and not existing_is_pending
                ):
                    logger.info(
                        "Duplicate bet skipped: %s %s %s",
                        row["line"],
                        row["wager"],
                        row["platform"],
                    )
                    if parse_audit_id:
                        _append_parse_audit(
                            {
                                "event": "ocr_duplicate_skipped",
                                "parse_id": parse_audit_id,
                                "bet": _audit_bet_fields(row),
                            }
                        )
                    return None

            # Seek to end and append while still holding the lock
            f.seek(0, 2)
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    if parse_audit_id:
        _append_parse_audit(
            {
                "event": "ocr_bet_logged",
                "parse_id": parse_audit_id,
                "bet": _audit_bet_fields(row),
            }
        )

    return row


def _ocr_sort_key(result):
    """Sort key for OCR results: top-to-bottom, left-to-right reading order."""
    return (-round(result[2][1] * 200), result[2][0])


def parse_bet_screenshot(image_bytes: bytes) -> list[dict]:
    """
    Use macOS native OCR (ocrmac) to extract text from a bet slip screenshot,
    then parse the text into structured bet data.
    Returns a list of parsed bet dicts.
    """
    parse_id = uuid.uuid4().hex[:12]

    # Write bytes to a temp file for ocrmac
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        results = ocrmac.OCR(tmp_path, recognition_level="accurate").recognize()
        try:
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            saved_path = os.path.join(SCREENSHOT_DIR, f"{parse_id}.jpg")
            shutil.copy2(tmp_path, saved_path)
            logger.info("Saved screenshot to %s", saved_path)
        except OSError as save_err:
            logger.warning("Could not save screenshot for re-testing: %s", save_err)
    finally:
        os.unlink(tmp_path)

    # Sort OCR results by y-coordinate (descending: y=1 is top in macOS coords)
    # so text flows top-to-bottom in visual reading order.  Without this,
    # ocrmac returns all left-column text first then right-column text,
    # which breaks card splitting on multi-card settled screenshots.
    # Quantize y to ~0.5% bands so items on the same visual line sort
    # left-to-right by x instead of by sub-pixel y jitter.
    results = sorted(results, key=_ocr_sort_key)

    # Combine OCR text lines (confidence threshold 0.5)
    lines = [text for text, confidence, bbox in results if confidence >= 0.5]
    logger.info(f"OCR raw lines:\n{chr(10).join(lines)}")

    # Detect platform from RAW text (before cleaning removes identifiers)
    raw_text_for_detection = "\n".join(lines)
    platform = _detect_platform(raw_text_for_detection)
    logger.info(f"Detected platform: {platform}")

    is_fd_settled = platform == "FanDuel" and _detect_fd_settled(raw_text_for_detection)

    # Clean junk lines before parsing
    cleaned = _clean_ocr_text(lines)
    cleaned_text = "\n".join(cleaned)
    logger.info(f"OCR cleaned text:\n{cleaned_text}")

    bets = _parse_bet_slip_text(
        cleaned_text, platform=platform, raw_text=raw_text_for_detection,
        is_fd_settled=is_fd_settled,
    )
    for bet in bets:
        bet["_parse_audit_id"] = parse_id

    _append_parse_audit(
        {
            "event": "ocr_parse",
            "parse_id": parse_id,
            "detected_platform": platform,
            "raw_ocr_text": raw_text_for_detection,
            "cleaned_ocr_text": cleaned_text,
            "parsed_candidates": [_audit_bet_fields(b) for b in bets],
        }
    )

    return bets


def _parse_dk_blocks(text: str) -> list[dict]:
    """Parse DraftKings bet slip using block-based approach.

    DK OCR produces structured blocks like:
        TEAM SPREAD ODDS
        Spread
        Wager: $X.XX [Paid: $X.XX]
        TEAM1
        TEAM2
        Final Score ...
    """
    # Match spread lines: "TEAM SPREAD [1|] ODDS"
    # OCR often inserts "1", "|", or "•" between spread and odds
    spread_line_re = re.compile(
        r"^(.+?)\s*([+-]\d+\.?\d*)\s*(?:[1|•]\s*)?([+-]\s*\d{3,})\s*$", re.MULTILINE
    )

    matches = list(spread_line_re.finditer(text))
    if not matches:
        return []

    skip_teams = {
        "SPREAD", "DRAFTKINGS", "SPORTSBOOK", "LOST", "WON", "PUSH", "VOID", "OT",
        "SETTLED", "OPEN", "LIVE", "MY BETS", "MY POOLS", "BETTING GROUPS",
        "SHARE", "THE CROWN IS YOURS", "FINAL",
    }
    bets = []

    for i, match in enumerate(matches):
        team = match.group(1).strip()
        spread = match.group(2)
        odds = match.group(3).replace(" ", "")  # fix OCR spaces in odds like "- 115"

        # Text between this spread line and the next one
        block_start = match.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[block_start:block_end]

        # Extract wager -- require "Wager:" prefix
        # Handle formats: "Wager: $1.00", "Wager:\n$1.00", "Wager: $1.00 / Paid: $1.89"
        wager_match = re.search(r"Wager:\s*[\n\s]*\$?(\d+\.?\d*)", block)
        wager = float(wager_match.group(1)) if wager_match else 0

        # Extract matchup from ALL-CAPS lines (DK lists teams on separate lines)
        team_lines = re.findall(r"^([A-Z][A-Z &'.]+)$", block, re.MULTILINE)
        teams = [
            t.strip() for t in team_lines
            if t.strip().upper() not in skip_teams
            and (len(t.strip()) >= 4 or t.strip().upper() in VALID_SHORT_TEAMS)
        ]
        game = f"{teams[0]} vs {teams[1]}" if len(teams) >= 2 else ""

        bets.append({
            "platform": "DraftKings",
            "game": game,
            "bet_type": "spread",
            "line": f"{team} {spread}",
            "odds": odds,
            "wager": wager,
        })

    return bets


def _parse_fd_settled_fallback(raw_text: str) -> dict | None:
    """Parse a FanDuel settled slip when OCR missed the header spread line.

    Settled FD slips have score sections and financial details that we can
    extract even when the styled header (team + spread) is unreadable.
    Works on raw (uncleaned) OCR text to preserve score lines and team names.
    Returns a partial bet dict with _partial=True, or None.
    """
    lines = raw_text.split("\n")

    # Look for score lines (2+ digit groups, e.g. "26 34 60") and adjacent team names
    score_re = re.compile(r"^\d+(?:\s+\d+){1,}$")
    teams = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if score_re.match(stripped) and i > 0:
            prev = lines[i - 1].strip()
            if (
                prev
                and re.match(r"^[A-Za-z]", prev)
                and not re.match(r"^\$", prev)
                and prev.lower() not in JUNK_LINES
            ):
                teams.append(prev)

    # Need at least one team and a dollar amount to proceed
    amounts = [
        float(m) for m in re.findall(r"\$(\d+\.?\d{0,2})", raw_text)
        if 0.25 <= float(m) <= 500
    ]
    if not teams or not amounts:
        return None

    wager = amounts[0]
    payout = amounts[1] if len(amounts) >= 2 else 0

    odds_match = re.search(r"([+-]\d{3,})", raw_text)
    odds = odds_match.group(1) if odds_match else "n/a"

    # Determine result from payout
    lower_text = raw_text.lower()
    if payout > wager:
        result = "win"
    elif "returned" in lower_text:
        if abs(payout - wager) < 0.01:
            result = "void"
        else:
            result = "loss"
    elif payout == 0:
        result = "loss"
    elif abs(payout - wager) < 0.01:
        result = "void"
    else:
        result = "loss"

    profit = round(payout - wager, 2)
    game = f"{teams[0]} vs {teams[1]}" if len(teams) >= 2 else ""

    return {
        "platform": "FanDuel",
        "game": game,
        "bet_type": "spread",
        "line": "",
        "odds": odds,
        "wager": wager,
        "payout": payout,
        "result": result,
        "profit": profit,
        "_partial": True,
        "_teams": teams,
    }


def _parse_fd_blocks(text: str) -> list[dict]:
    """Parse FanDuel bet slip using block-based approach.

    FanDuel OCR layout (spread and odds on SEPARATE lines):
        Team Name +/-X.X
        Team1 @ Team2
        $X.XX           <- wager
        BET ID: ...
        -NNN            <- odds on own line
        $X.XX           <- payout
    """
    # FD spread lines have NO odds on the same line: "Team +/-X.X" alone
    # Parens allow FanDuel's "(W)" women's league suffix
    spread_line_re = re.compile(
        r"^([A-Za-z][A-Za-z &'.()\-]+?)\s+([+-]\d+\.?\d*)\s*$", re.MULTILINE
    )
    matches = list(spread_line_re.finditer(text))
    if not matches:
        return []

    bets = []
    for i, match in enumerate(matches):
        team = match.group(1).strip()
        spread = match.group(2)

        # Detect women's league from FanDuel "(W)" suffix
        league = ""
        if re.search(r"\(W\)\s*$", team):
            league = "womens"
            team = re.sub(r"\s*\(W\)\s*$", "", team).strip()

        # Block of text between this spread line and the next
        block_start = match.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[block_start:block_end]

        # Matchup: "Team1 @ Team2"
        matchup_match = re.search(
            r"([A-Za-z][A-Za-z &'.()\-]+?)\s+@\s+([A-Za-z][A-Za-z &'.()\-]+)",
            block,
        )
        if matchup_match:
            raw_g1 = matchup_match.group(1).strip()
            raw_g2 = matchup_match.group(2).strip()
            if not league and ("(W)" in raw_g1 or "(W)" in raw_g2):
                league = "womens"
            g1 = re.sub(r"\s*\(W\)\s*$", "", raw_g1)
            g2 = re.sub(r"\s*\(W\)\s*$", "", raw_g2)
            game = f"{g1} vs {g2}"
        else:
            game = ""

        # Wager: first dollar amount in reasonable range
        wager_amounts = re.findall(r"\$(\d+\.?\d{0,2})", block)
        wager = 0
        for amt_str in wager_amounts:
            amt = float(amt_str)
            if 0.25 <= amt <= 500:
                wager = amt
                break

        # Odds: standalone American odds line (e.g. "-108") in the block
        odds_match = re.search(r"^([+-]\d{3,})\s*$", block, re.MULTILINE)
        odds = odds_match.group(1) if odds_match else "n/a"

        bets.append({
            "platform": "FanDuel",
            "game": game,
            "bet_type": "spread",
            "line": f"{team} {spread}",
            "odds": odds,
            "wager": wager,
            "league": league,
        })

    return bets


def _parse_fd_settled_cards(raw_text: str) -> list[dict]:
    """Parse a FanDuel settled screenshot into individual bet dicts.

    Splits on PLACED: lines (real OCR output), falling back to
    FANDUEL/SPORTSBOOK banners for legacy fixtures.
    """
    placed_re = re.compile(r"^\s*PLACED:\s*\d+/\d+/\d{4}", re.IGNORECASE | re.MULTILINE)

    if placed_re.search(raw_text):
        # Split after each PLACED: line (each card ends with PLACED)
        lines = raw_text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        for line in lines:
            current.append(line)
            if placed_re.match(line):
                chunks.append("\n".join(current))
                current = []
        # Remaining lines after last PLACED (truncated card at bottom)
        if current:
            chunks.append("\n".join(current))
        cards = chunks
    else:
        # Fallback: split at FANDUEL ... SPORTSBOOK banners
        cards = re.split(r"FANDUEL\s*\n\s*SPORTSBOOK", raw_text, flags=re.IGNORECASE)
        if cards:
            cards = cards[1:]
        if not cards:
            logger.warning(
                "FD settled card splitting found no card boundaries "
                "(no PLACED: lines and no FANDUEL/SPORTSBOOK banners)"
            )

    bets = []
    for card_text in cards:
        bet = _parse_single_fd_settled_card(card_text)
        if bet is not None:
            bets.append(bet)
    return bets


def _resolve_game_date(team_name: str) -> str:
    """Look up the game date for a team from prediction files.

    Searches men's and women's daily and dated prediction files
    for a matching team in the Pick or Matchup column. Returns YYYY-MM-DD or "".
    """
    if not team_name:
        return ""

    team_upper = team_name.upper()
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    # Build list of prediction files to search (most recent first)
    # Include both men's and women's prediction files
    pred_files = []
    for base in [DAILY_PREDICTIONS, os.path.join(BASE_DIR, "daily_predictions_wbb.csv")]:
        if os.path.exists(base):
            pred_files.append(base)
    archive_files = []
    for pattern in ("predictions_*.csv", "predictions_wbb_*.csv"):
        archive_files.extend(glob.glob(os.path.join(BASE_DIR, pattern)))

    def _archive_sort_key(path: str) -> str:
        match = re.search(r"(\d{8})", os.path.basename(path))
        return match.group(1) if match else ""

    for fname in sorted(archive_files, key=_archive_sort_key, reverse=True):
        if fname not in pred_files:
            pred_files.append(fname)

    for fpath in pred_files:
        try:
            df = pd.read_csv(fpath)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            logger.warning("Could not read prediction file %s for date resolution: %s", fpath, e)
            continue
        if "Date/Time" not in df.columns or "Matchup" not in df.columns:
            logger.warning(
                "Prediction file %s missing expected columns (has: %s); skipping",
                fpath, list(df.columns),
            )
            continue

        for _, row in df.iterrows():
            matchup = str(row.get("Matchup", "")).upper()
            pick = str(row.get("Pick", "")).upper()
            if team_upper in matchup or team_upper in pick:
                dt_str = str(row["Date/Time"]).strip()
                # Format: "MM/DD HH:MM AM|PM"
                m = re.match(r"(\d{1,2})/(\d{1,2})", dt_str)
                if m:
                    month, day = int(m.group(1)), int(m.group(2))
                    year = now.year
                    if month == 12 and now.month <= 2:
                        year -= 1
                    return f"{year}-{month:02d}-{day:02d}"
                logger.warning(
                    "Found team %r in %s but Date/Time %r did not match expected MM/DD format",
                    team_name, fpath, dt_str,
                )
    return ""


def find_unlogged_strong_games(target_date) -> list[dict]:
    """Find STRONG-rated games on target_date that have no FanDuel/DraftKings bet logged.

    Scans prediction archives for both men's and women's leagues.
    Returns list of dicts with matchup, pick, edge, units, league info.
    """
    target_mm = target_date.month
    target_dd = target_date.day

    def _archive_sort_key_local(path: str) -> str:
        match = re.search(r"(\d{8})", os.path.basename(path))
        return match.group(1) if match else ""

    # Collect STRONG games from prediction archives, keyed by matchup to deduplicate
    strong_games = {}

    league_prefixes = [
        ("predictions", "mens"),
        ("predictions_wbb", "womens"),
    ]

    for prefix, league in league_prefixes:
        archive_files = glob.glob(os.path.join(BASE_DIR, f"{prefix}_*.csv"))
        # Exclude wbb files from men's glob (predictions_*.csv also matches predictions_wbb_*)
        if prefix == "predictions":
            archive_files = [f for f in archive_files if "_wbb_" not in os.path.basename(f)]
        archive_files.sort(key=_archive_sort_key_local, reverse=True)

        for fpath in archive_files[:7]:
            try:
                df = pd.read_csv(fpath)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
                logger.warning("Skipping unreadable prediction archive: %s", fpath, exc_info=True)
                continue
            if "Date/Time" not in df.columns:
                continue

            for _, row in df.iterrows():
                dt_str = str(row.get("Date/Time", "")).strip()
                m = re.match(r"(\d{1,2})/(\d{1,2})", dt_str)
                if not m:
                    continue
                month, day = int(m.group(1)), int(m.group(2))
                if (month, day) != (target_mm, target_dd):
                    continue

                std_rating = str(row.get("Std_Rating", "")).strip()
                kalshi_rating = str(row.get("Rating", "")).strip()
                if std_rating != "STRONG" and kalshi_rating != "STRONG":
                    continue

                matchup = str(row.get("Matchup", "")).strip()
                matchup_key = matchup.upper()

                new_entry = {
                    "matchup": matchup,
                    "pick": str(row.get("Pick", "")),
                    "std_edge_pct": str(row.get("Std_Edge_Pct", "")),
                    "std_units": float(row.get("Std_Units", 0) or 0),
                    "league": league,
                    "bet_type": str(row.get("Bet_Type", "spread")),
                }

                # Prefer rows with actual Std edge data (spread rows over Kalshi game rows)
                existing = strong_games.get(matchup_key)
                if existing and existing["std_units"] > 0 and new_entry["std_units"] == 0:
                    continue

                strong_games[matchup_key] = new_entry

    if not strong_games:
        return []

    # Check betting_history.csv for already-logged FanDuel/DraftKings bets
    target_str = target_date.strftime("%Y-%m-%d")
    if not os.path.exists(BETTING_HISTORY):
        return list(strong_games.values())

    try:
        hist = pd.read_csv(BETTING_HISTORY)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        logger.warning("Could not read %s -- treating all STRONG games as unlogged", BETTING_HISTORY, exc_info=True)
        return list(strong_games.values())

    logged = hist[
        (hist["date"] == target_str)
        & (hist["platform"].isin(["FanDuel", "DraftKings"]))
    ]

    if logged.empty:
        return list(strong_games.values())

    # Build list of normalized team pairs from logged games
    logged_team_pairs = []
    for game_str in logged["game"].dropna().unique():
        parts = re.split(r"\s+vs\s+", str(game_str), flags=re.IGNORECASE)
        if len(parts) == 2:
            logged_team_pairs.append(
                tuple(normalize_team_name(p.strip()).upper() for p in parts)
            )

    def _teams_match(pred_team: str, logged_team: str) -> bool:
        """Check if two normalized team names refer to the same school."""
        if pred_team == logged_team:
            return True
        # One may be a substring of the other (e.g. "Tulsa" vs "Tulsa Golden Hurricane")
        return pred_team in logged_team or logged_team in pred_team

    unlogged = []
    for info in strong_games.values():
        parts = re.split(r"\s+@\s+", info["matchup"])
        if len(parts) != 2:
            unlogged.append(info)
            continue

        away_norm = normalize_team_name(parts[0].strip()).upper()
        home_norm = normalize_team_name(parts[1].strip()).upper()

        found = any(
            (_teams_match(away_norm, t1) and _teams_match(home_norm, t2))
            or (_teams_match(away_norm, t2) and _teams_match(home_norm, t1))
            for t1, t2 in logged_team_pairs
        )
        if not found:
            unlogged.append(info)

    return unlogged


async def _reminder_check_unlogged(context) -> None:
    """Daily job: remind user about unlogged STRONG bets from yesterday."""
    eastern = pytz.timezone("US/Eastern")
    yesterday = (datetime.now(eastern) - timedelta(days=1)).date()

    try:
        unlogged = find_unlogged_strong_games(yesterday)
    except Exception:
        logger.exception("Error checking for unlogged STRONG games")
        return

    if not unlogged:
        return

    lines = [f"Unlogged STRONG bets from {yesterday.strftime('%m/%d')}:\n"]
    for g in unlogged:
        league_tag = "[W] " if g["league"] == "womens" else ""
        lines.append(
            f"  {league_tag}{g['matchup']}\n"
            f"    Pick: {g['pick']}  |  {g['std_edge_pct']}  |  {g['std_units']:.1f}U"
        )
    lines.append("\nDid you place any FanDuel/DraftKings bets on these? Send a bet slip or log manually.")

    msg = "\n".join(lines)

    for user_id in ALLOWED_USER_IDS:
        try:
            await context.bot.send_message(chat_id=user_id, text=msg)
        except Exception:
            logger.exception("Failed to send reminder to user %d", user_id)


def _parse_single_fd_settled_card(card_text: str) -> dict | None:
    """Parse a single FanDuel settled card from raw OCR text."""
    # 1. Extract BET ID (before cleaning)
    bet_id_match = FD_BET_ID_RE.search(card_text)
    bet_id = bet_id_match.group(1) if bet_id_match else ""

    # 2. Cross-screenshot dedup check (don't mark as seen yet; incomplete cards shouldn't burn IDs)
    if bet_id and bet_id in _SEEN_FD_BET_IDS:
        logger.info("Cross-screenshot dedup: skipping already-seen bet_id %s", bet_id)
        return {"_skipped": True, "_skip_reason": "dedup", "_settled": True,
                "platform": "FanDuel", "bet_id": bet_id}

    # 3. Extract game date
    date_match = FD_GAME_DATE_RE.search(card_text)
    game_date = ""
    if date_match:
        month_str = date_match.group(1).upper()
        day = int(date_match.group(2))
        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        month = month_map.get(month_str)
        if month is None:
            logger.warning(
                "Unrecognized month abbreviation %r in FD settled card (bet_id=%s), "
                "falling back to month=1",
                month_str, bet_id,
            )
            month = 1
        eastern = pytz.timezone("US/Eastern")
        now = datetime.now(eastern)
        year = now.year
        # Handle year boundary: Dec game viewed in Jan
        if month == 12 and now.month <= 2:
            year -= 1
        game_date = f"{year}-{month:02d}-{day:02d}"

    # 3b. Extract PLACED date as a last-resort fallback (used only if prediction
    # lookup also fails -- see step 6b below).
    placed_date = ""
    if not game_date:
        placed_date_match = re.search(
            r"PLACED:\s*(\d{1,2})/(\d{1,2})/(\d{4})", card_text, re.IGNORECASE
        )
        if placed_date_match:
            p_month = int(placed_date_match.group(1))
            p_day = int(placed_date_match.group(2))
            p_year = int(placed_date_match.group(3))
            if 1 <= p_month <= 12 and 1 <= p_day <= 31 and 2020 <= p_year <= 2030:
                placed_date = f"{p_year}-{p_month:02d}-{p_day:02d}"
            else:
                logger.warning(
                    "FD settled card: PLACED date out of range month=%d day=%d "
                    "year=%d (bet_id=%s)",
                    p_month, p_day, p_year, bet_id,
                )

    # 4. Detect result from raw text
    lower = card_text.lower()
    if "won on fanduel" in lower:
        result = "win"
    elif "lost on fanduel" in lower or re.search(r"^\s*lost\s*$", lower, re.MULTILINE):
        result = "loss"
    elif re.search(r"^\s*returned\s*$", lower, re.MULTILINE):
        # RETURNED present -- distinguish loss ($0.00 payout) from void (wager refunded).
        # Extract dollar amounts in betting range; if two found and approximately
        # equal, wager was refunded (void). Otherwise loss.
        ret_amts = re.findall(r"\$(\d+\.?\d{0,2})", card_text)
        ret_in_range = [float(m) for m in ret_amts if 0.25 <= float(m) <= 500]
        if len(ret_in_range) >= 2 and abs(ret_in_range[1] - ret_in_range[0]) < 0.01:
            result = "void"
        else:
            result = "loss"
        logger.info(
            "FD settled card: RETURNED -> %s (bet_id=%s, amounts_in_range=%s)",
            result, bet_id, ret_in_range,
        )
    else:
        # No keyword markers found.  If there's no BET ID either, this is
        # almost certainly a trailing open/unsettled card (or betslip bar
        # remnant) rather than a settled card whose keywords OCR missed.
        if not bet_id:
            result = "pending"
            logger.info(
                "FD settled card: no result keywords and no BET ID, "
                "treating as unsettled/pending"
            )
        else:
            # Has a BET ID but no keywords -- infer result from payout vs wager.
            # Covers "Finished" format (scores visible) and cases where OCR
            # misses colored "WON/LOST ON FANDUEL" text on dark backgrounds.
            # Assumes first dollar amount in [0.25, 500] is wager and second is payout.
            all_amts = re.findall(r"\$(\d+\.?\d{0,2})", card_text)
            amts_in_range = [float(m) for m in all_amts if 0.25 <= float(m) <= 500]
            if len(amts_in_range) >= 2:
                w, p = amts_in_range[0], amts_in_range[1]
                if p > w:
                    result = "win"
                elif p < w:
                    result = "loss"
                else:
                    result = "void"
                logger.info(
                    "FD settled card: no WON/LOST keyword, inferred result=%s "
                    "from wager=$%.2f payout=$%.2f (bet_id=%s)",
                    result, w, p, bet_id,
                )
            else:
                result = "pending"
                logger.warning(
                    "FD settled card: no result keywords and < 2 dollar amounts, "
                    "defaulting to pending (bet_id=%s, amounts=%s)",
                    bet_id, amts_in_range,
                )

    # 5. Clean and parse structured fields
    lines = card_text.split("\n")
    cleaned = _clean_ocr_text(lines)

    # Parens allow FanDuel's "(W)" women's league suffix
    spread_re = re.compile(r"^([A-Za-z][A-Za-z &'.()\-]+?)\s+([+-]\d+\.?\d*)\s*$")
    matchup_re = re.compile(r"([A-Za-z][A-Za-z &'.()\-]+?)\s+@\s+([A-Za-z][A-Za-z &'.()\-]+)")

    team = ""
    spread = ""
    game = ""
    league = ""
    wager = 0.0
    odds = "n/a"

    for cl in cleaned:
        cl = cl.strip()
        if not team:
            sm = spread_re.match(cl)
            if sm:
                team = sm.group(1).strip()
                spread = sm.group(2)
                # Detect women's league from FanDuel "(W)" suffix
                if re.search(r"\(W\)\s*$", team):
                    league = "womens"
                    team = re.sub(r"\s*\(W\)\s*$", "", team).strip()
                continue
        if not game:
            mm = matchup_re.search(cl)
            if mm:
                raw_g1 = mm.group(1).strip()
                raw_g2 = mm.group(2).strip()
                if not league and ("(W)" in raw_g1 or "(W)" in raw_g2):
                    league = "womens"
                g1 = re.sub(r"\s*\(W\)\s*$", "", raw_g1)
                g2 = re.sub(r"\s*\(W\)\s*$", "", raw_g2)
                game = f"{g1} vs {g2}"
                continue

    # Fallback game detection: in the FanDuel "Finished" format, team names
    # appear between "SPREAD BETTING" and the wager ($X.XX / TOTAL WAGER).
    if not game:
        raw_lines = card_text.split("\n")
        game_teams = []
        in_team_zone = False
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if "SPREAD BETTING" in stripped.upper():
                in_team_zone = True
                continue
            if in_team_zone:
                if (
                    stripped.startswith("$")
                    or "TOTAL WAGER" in stripped.upper()
                    or "BET ID" in stripped.upper()
                ):
                    break
                if (
                    stripped
                    and re.match(r"^[A-Za-z]", stripped)
                    and stripped.lower() not in JUNK_LINES
                    and len(stripped) >= 3
                ):
                    game_teams.append(stripped)
        if len(game_teams) >= 2:
            # Detect women's league from "(W)" in team names
            if not league and any("(W)" in t for t in game_teams):
                league = "womens"
            gt0 = re.sub(r"\s*\(W\)\s*$", "", game_teams[0]).strip()
            gt1 = re.sub(r"\s*\(W\)\s*$", "", game_teams[1]).strip()
            game = f"{gt0} vs {gt1}"

    # Wager: first dollar amount in range from cleaned text
    for cl in cleaned:
        wager_amounts = re.findall(r"\$(\d+\.?\d{0,2})", cl)
        for amt_str in wager_amounts:
            amt = float(amt_str)
            if 0.25 <= amt <= 500:
                wager = amt
                break
        if wager > 0:
            break

    # Odds: standalone American odds line
    for cl in cleaned:
        odds_match = re.match(r"^([+-]\d{3,})\s*$", cl.strip())
        if odds_match:
            odds = odds_match.group(1)
            break

    # 6. Extract payout from raw text (second dollar amount in range)
    all_amounts = re.findall(r"\$(\d+\.?\d{0,2})", card_text)
    raw_amounts = [float(m) for m in all_amounts if 0.25 <= float(m) <= 500]
    if len(raw_amounts) < 2:
        logger.warning(
            "FD settled card: expected >= 2 dollar amounts, found %d (bet_id=%s)",
            len(raw_amounts), bet_id,
        )
    payout = raw_amounts[1] if len(raw_amounts) >= 2 else 0.0

    # 7. Result-specific payout override
    if result == "win":
        # Sanity check: a winning payout must exceed the wager. If OCR
        # produced a bogus second dollar amount, recalculate from odds.
        if payout <= wager and odds and wager > 0:
            try:
                odds_val = int(float(odds))
                if odds_val < 0:
                    calc_profit = wager * 100.0 / abs(odds_val)
                else:
                    calc_profit = wager * odds_val / 100.0
                payout = round(wager + calc_profit, 2)
                logger.warning(
                    "FD settled card: OCR payout ($%.2f) <= wager ($%.2f), "
                    "recalculated from odds %s -> $%.2f (bet_id=%s)",
                    raw_amounts[1] if len(raw_amounts) >= 2 else 0.0,
                    wager, odds, payout, bet_id,
                )
            except (ValueError, TypeError):
                pass
        profit = round(payout - wager, 2)
    elif result == "void":
        payout = wager  # void = wager refunded; override raw $0.00 from OCR
        profit = 0.0
    elif result == "loss":
        payout = 0.0
        profit = round(-wager, 2)
    else:
        profit = 0.0

    line = f"{team} {spread}" if team and spread else ""

    # 6b. Resolve game date: prediction files take precedence over PLACED date
    if not game_date and team:
        game_date = _resolve_game_date(team)
    if not game_date and placed_date:
        game_date = placed_date
        logger.info(
            "FD settled card: using PLACED date fallback %s (bet_id=%s)",
            game_date, bet_id,
        )

    if not line or wager <= 0:
        logger.warning(
            "FD settled card parse incomplete: line=%r wager=%s bet_id=%s",
            line, wager, bet_id,
        )
        return {
            "_skipped": True, "_skip_reason": "incomplete", "_settled": True,
            "platform": "FanDuel",
            "bet_id": bet_id, "wager": wager, "result": result, "game": game,
        }

    if result == "pending":
        logger.info(
            "FD settled card: result=pending, skipping settlement flow "
            "(line=%r wager=%s bet_id=%s)",
            line, wager, bet_id,
        )
        return {
            "_skipped": True, "_skip_reason": "unsettled", "_settled": True,
            "platform": "FanDuel",
            "bet_id": bet_id, "wager": wager, "result": result, "game": game,
            "line": line,
        }

    # Mark BET ID as seen only after successful parse
    if bet_id:
        _SEEN_FD_BET_IDS.add(bet_id)

    return {
        "platform": "FanDuel",
        "game": game,
        "bet_type": "spread",
        "line": line,
        "odds": odds,
        "wager": wager,
        "date": game_date,
        "result": result,
        "payout": payout,
        "profit": profit,
        "bet_id": bet_id,
        "league": league,
        "_settled": True,
    }


def _parse_kalshi_blocks(text: str) -> list[dict]:
    """Parse Kalshi bet slip using structured approach.

    Kalshi OCR produces blocks like:
        NCAAMB
        Team1 at Team2: Spread
        Yes * Team1 wins by over X.X Points
        ...
        Cost
        $X.XX
    """
    # Extract spread details: "Yes/No . Team wins by over X.X Points"
    spread_re = re.compile(
        r"(Yes|No)\s*[•·.*]\s*(.+?)\s+wins by over\s+(\d+\.?\d*)\s+Points",
        re.IGNORECASE,
    )
    spread_details = spread_re.findall(text)
    if not spread_details:
        return []

    # Extract matchups: "Team1 at Team2: Spread"
    matchup_re = re.compile(r"(.+?)\s+at\s+(.+?):\s*Spread", re.IGNORECASE)
    matchups = matchup_re.findall(text)

    # Extract costs: "Cost\n$X.XX" or "Cost $X.XX"
    cost_re = re.compile(r"Cost\s+\$(\d+\.?\d*)", re.IGNORECASE)
    costs = cost_re.findall(text)

    bets = []
    for i, (side, team, spread) in enumerate(spread_details):
        team = team.strip()
        if i < len(matchups):
            game = f"{matchups[i][0].strip()} vs {matchups[i][1].strip()}"
        else:
            game = ""
        wager = float(costs[i]) if i < len(costs) else 0

        bets.append({
            "platform": "Kalshi",
            "game": game,
            "bet_type": "spread",
            "line": f"{team} -{spread} {side.upper()}",
            "odds": "n/a",
            "wager": wager,
        })

    return bets


def _detect_platform(text: str) -> str:
    """Detect betting platform from OCR text."""
    text_lower = text.lower()
    if "draftkings" in text_lower:
        return "DraftKings"
    elif "fanduel" in text_lower:
        return "FanDuel"
    elif "kalshi" in text_lower or "wins by over" in text_lower:
        return "Kalshi"
    elif "betmgm" in text_lower:
        return "BetMGM"
    elif "caesars" in text_lower:
        return "Caesars"
    # FanDuel heuristics -- FD screenshots often lack the "FanDuel" word
    elif "spread betting" in text_lower or "total wager" in text_lower:
        return "FanDuel"
    # DraftKings heuristics -- DK format uses "Wager:" prefix
    elif re.search(r"Wager:\s*\$?", text):
        return "DraftKings"
    else:
        return "Unknown"


def _parse_bet_slip_text(
    text: str, platform: str = None, raw_text: str = None,
    is_fd_settled: bool = False,
) -> list[dict]:
    """Parse raw OCR text from a bet slip into structured bet data."""
    # Use provided platform or try to detect from text
    if platform is None:
        platform = _detect_platform(text)

    # Use platform-specific parser when available
    if platform == "DraftKings":
        bets = _parse_dk_blocks(text)
        if bets:
            return bets

    if platform == "FanDuel":
        # Settled path: use dedicated settled parser BEFORE pending parser
        if is_fd_settled:
            bets = _parse_fd_settled_cards(raw_text or text)
            if bets:
                return bets

        bets = _parse_fd_blocks(text)
        if bets:
            return bets
        # Settled slip fallback (uses raw text to preserve score lines)
        partial = _parse_fd_settled_fallback(raw_text or text)
        if partial:
            return [partial]

    if platform == "Kalshi":
        bets = _parse_kalshi_blocks(text)
        if bets:
            return bets

    # --- Generic fallback parser ---

    # Find spread lines: "Team +/-X.X" or "Team +/-X"
    spread_pattern = re.compile(
        r"([A-Z][A-Za-z\s&'.]+?)\s+([+-]\d+\.?\d*)", re.MULTILINE
    )
    spreads = spread_pattern.findall(text)

    # Find American odds: -110, +150, etc.
    odds_pattern = re.compile(r"([+-]\d{3,})")
    odds_matches = odds_pattern.findall(text)

    # Find wager amounts -- require keyword prefix to avoid matching stray numbers
    wager_pattern = re.compile(
        r"(?:wager|risk|stake|bet)\s*:?\s*\$?(\d+\.?\d{0,2})\b", re.IGNORECASE
    )
    wager_matches = wager_pattern.findall(text)
    wagers = [float(w) for w in wager_matches if 0.25 <= float(w) <= 500]

    # Find "vs" or "@" matchups
    matchup_pattern = re.compile(
        r"([A-Z][A-Za-z\s&'.]+?)\s+(?:vs\.?|@|at)\s+([A-Z][A-Za-z\s&'.]+?)(?:\n|$)",
        re.IGNORECASE,
    )
    matchups = matchup_pattern.findall(text)

    bets = []
    if spreads:
        for i, (team, spread) in enumerate(spreads):
            team = team.strip()
            # Skip if it looks like a score or irrelevant number
            if len(team) < 2:
                continue

            bet = {
                "platform": platform,
                "game": f"{matchups[i][0].strip()} vs {matchups[i][1].strip()}" if i < len(matchups) else "",
                "bet_type": "spread",
                "line": f"{team} {spread}",
                "odds": odds_matches[i] if i < len(odds_matches) else "n/a",
                "wager": wagers[i] if i < len(wagers) else 0,
            }
            bets.append(bet)

    # If regex parsing found nothing, return the raw text so the user can see what OCR got
    if not bets:
        bets.append({
            "platform": platform,
            "game": "",
            "bet_type": "spread",
            "line": "",
            "odds": "n/a",
            "wager": 0,
            "_raw_ocr": text,
        })

    return bets


def parse_dk_share_url(url: str) -> dict | None:
    """
    Parse a DraftKings social share URL to extract bet result data.

    URL format: https://sportsbook.draftkings.com/social/post/{uuid}?slipAdd

    Uses the DK social API to fetch structured bet data (the share pages are
    JS-rendered SPAs, so plain HTTP fetching doesn't work).

    Returns a dict with bet details and result, or None if parsing fails.
    """
    # Extract the post UUID from the URL
    uuid_match = re.search(r'/social/post/([a-zA-Z0-9-]+)', url)
    if not uuid_match:
        logger.warning(f"Could not extract post key from DK share URL: {url}")
        return None

    post_key = uuid_match.group(1)

    try:
        resp = httpx.post(
            "https://api.draftkings.com/comments/feed/post/details.json",
            json={
                "postKey": post_key,
                "replyDepth": 1,
                "replyScrolling": {"limit": 10, "sort": "desc"},
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success", False):
            logger.warning(f"DK API returned failure for post {post_key}")
            return None

        # Bet data is base64-encoded JSON in postEntries
        post = data["post"]
        entries = post.get("postEntries", [])
        if not entries:
            logger.warning(f"No post entries in DK share: {post_key}")
            return None

        props = entries[0].get("metadataProperties", {}).get("properties", {})
        bet_b64 = props.get("value", "")
        if not bet_b64:
            logger.warning(f"No betJSON value in DK share: {post_key}")
            return None

        bet_data = json.loads(base64.b64decode(bet_b64))

        status = str(bet_data.get("status", "pending")).strip().lower()
        # Map DK statuses to our format (win/loss/void/pending).
        status_map = {
            "won": "win",
            "lost": "loss",
            "pushed": "void",
            "voided": "void",
            "cashout": "void",
            "open": "pending",
            "pending": "pending",
            "placed": "pending",
            "accepted": "pending",
            "unsettled": "pending",
            "active": "pending",
        }
        result = status_map.get(status, "pending")

        stake = float(bet_data.get("stake", 0))
        payout = float(bet_data.get("payout", 0))

        # Extract bet details from combinationOutcomes
        outcomes = bet_data.get("combinationOutcomes", [])
        if not outcomes:
            logger.warning(f"No outcomes in DK bet data: {post_key}")
            return None

        outcome = outcomes[0]
        line = outcome.get("outcomeLabel", "")
        odds = outcome.get("playedOddsAmerican", "n/a")
        bet_type = outcome.get("offerLabel", "spread").lower()

        # Build game string from events
        events = bet_data.get("events", [])
        game = events[0].get("name", "").replace(" @ ", " vs ") if events else ""

        # Calculate profit
        if result == "win":
            profit = payout - stake
        elif result == "loss":
            profit = -stake
        else:
            profit = 0

        return {
            "platform": "DraftKings",
            "game": game,
            "bet_type": bet_type,
            "line": line,
            "odds": odds,
            "wager": stake,
            "result": result,
            "payout": payout,
            "profit": round(profit, 2),
        }

    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching DK share API: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing DK share API response: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error parsing DK share URL: {e}")
        return None


def parse_kalshi_share_url(url: str) -> dict | None:
    """
    Parse a Kalshi shared trade URL to extract bet data.

    URL params carry the trade details:
        marketTicker    = KXNCAAMBSPREAD-26FEB14SMCPAC-SMC8
        direction       = yes
        cost_cents      = 159
        max_payout_cents = 300

    Calls the Kalshi public market API to resolve team name and spread
    from the market title (e.g. "Saint Mary's wins by over 8.5 Points").

    Returns a dict with bet details, or None if parsing fails.
    """
    parsed_url = urlparse(url)
    params = parse_qs(parsed_url.query)

    market_ticker = params.get("marketTicker", [None])[0]
    direction = params.get("direction", [None])[0]
    cost_cents = params.get("cost_cents", [None])[0]
    max_payout_cents = params.get("max_payout_cents", [None])[0]

    if not all([market_ticker, direction, cost_cents, max_payout_cents]):
        logger.warning(f"Missing required params in Kalshi share URL: {url}")
        return None

    cost = int(cost_cents) / 100
    max_payout = int(max_payout_cents) / 100
    side = direction.upper()

    team = ""
    spread = 0.0
    game = ""
    result = "pending"

    # Fetch market details from Kalshi API
    try:
        resp = httpx.get(
            f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}",
            timeout=10,
        )
        resp.raise_for_status()
        market = resp.json().get("market", {})

        # Title like "Saint Mary's wins by over 8.5 Points?"
        title = market.get("title", "")
        title_match = re.match(
            r"(.+?)\s+wins by over\s+(\d+\.?\d*)\s+Points\??", title, re.IGNORECASE
        )
        if title_match:
            team = title_match.group(1).strip()
            spread = float(title_match.group(2))

        # Fetch event for game matchup (e.g. "Saint Mary's at Pacific: Spread")
        event_ticker = market.get("event_ticker", "")
        if event_ticker:
            try:
                ev_resp = httpx.get(
                    f"https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}",
                    timeout=10,
                )
                ev_resp.raise_for_status()
                ev_title = ev_resp.json().get("event", {}).get("title", "")
                at_match = re.match(r"(.+?)\s+at\s+(.+?)(?::\s*Spread)?$", ev_title, re.IGNORECASE)
                if at_match:
                    game = f"{at_match.group(1).strip()} vs {at_match.group(2).strip()}"
            except Exception as e:
                logger.warning(f"Could not fetch event details for {event_ticker}: {e}")

        # Check settlement status
        status = market.get("status", "")
        if status in ("settled", "finalized"):
            market_result = market.get("result", "")
            if market_result == "yes":
                result = "win" if side == "YES" else "loss"
            elif market_result == "no":
                result = "win" if side == "NO" else "loss"

    except Exception as e:
        logger.warning(f"Kalshi API error for {market_ticker}, falling back to ticker: {e}")
        # Fallback: decode from ticker (KXNCAAMBSPREAD-26FEB14SMCPAC-SMC8)
        contract = market_ticker.rsplit("-", 1)[-1] if "-" in market_ticker else ""
        num_match = re.match(r"([A-Z]+)(\d+)", contract)
        if num_match:
            team = num_match.group(1)
            spread = float(num_match.group(2)) + 0.5

    if not team:
        logger.warning(f"Could not determine team from Kalshi ticker: {market_ticker}")
        return None

    line = f"{team} -{spread} {side}"

    if result == "win":
        payout = max_payout
        profit = round(max_payout - cost, 2)
    elif result == "loss":
        payout = 0.0
        profit = round(-cost, 2)
    else:
        # Preserve max payout so settlement can compute exact profit later.
        payout = max_payout
        profit = 0.0

    return {
        "platform": "Kalshi",
        "game": game,
        "bet_type": "spread",
        "line": line,
        "odds": "n/a",
        "wager": cost,
        "result": result,
        "payout": round(payout, 2),
        "profit": profit,
    }


async def _log_share_url_bet(update: Update, bet: dict):
    """Log or settle a bet parsed from a share URL (DraftKings or Kalshi).

    If the bet is settled, tries to update an existing pending bet first.
    Otherwise logs it as a new bet. Sends a reply message in all cases.
    """
    odds = bet.get("odds", "n/a")
    odds_str = f", {odds}" if odds and odds != "n/a" else ""

    if bet.get("result") in ("win", "loss", "void"):
        update_msg = update_bet_result(bet)
        if not update_msg:
            # Kalshi fallback: ticker-derived abbreviation may not match
            update_msg = _find_kalshi_pending_by_spread(bet)
        if update_msg:
            await update.message.reply_text(update_msg)
            return
        # No matching pending bet -- log it as a settled bet directly
        row = append_bet(bet)
        if row is None:
            await update.message.reply_text("Duplicate bet -- already logged.")
        else:
            profit_val = float(row.get('profit', 0) or 0)
            await update.message.reply_text(
                f"Logged ({row['result'].upper()}): {row['line']}{odds_str}, "
                f"${float(row['wager']):.2f} on {row['platform']} "
                f"(profit: {profit_val:+.2f}U)"
            )
    else:
        row = append_bet(bet)
        if row is None:
            await update.message.reply_text("Duplicate bet -- already logged.")
        else:
            await update.message.reply_text(
                f"Logged: {row['line']}{odds_str}, ${float(row['wager']):.2f} on {row['platform']}"
            )


def parse_shorthand(text: str) -> dict:
    """
    Parse shorthand text entry like:
        FD PROV +15.5 -110 1.25
        DK UConn -15.5 NO n/a 1.53

    Format: PLATFORM TEAM SPREAD [YES/NO] ODDS WAGER
    """
    parts = text.strip().split()
    if len(parts) < 4:
        return None

    # Platform alias mapping
    platform_map = {
        "FD": "FanDuel",
        "DK": "DraftKings",
        "K": "Kalshi",
        "KAL": "Kalshi",
        "FANDUEL": "FanDuel",
        "DRAFTKINGS": "DraftKings",
        "KALSHI": "Kalshi",
    }

    platform = platform_map.get(parts[0].upper(), parts[0])

    # Find the spread value (first token that looks like +/- number)
    spread_idx = None
    for i in range(1, len(parts)):
        if re.match(r"^[+-]\d+\.?\d*$", parts[i]):
            spread_idx = i
            break

    if spread_idx is None:
        return None

    team = " ".join(parts[1:spread_idx])
    spread = parts[spread_idx]

    remaining = parts[spread_idx + 1 :]

    # Check for YES/NO
    side = None
    if remaining and remaining[0].upper() in ("YES", "NO"):
        side = remaining[0].upper()
        remaining = remaining[1:]

    line = f"{team} {spread}"
    if side:
        line += f" {side}"

    # Next is odds, then wager
    odds = "n/a"
    wager = 0

    if remaining:
        odds = remaining[0]
        remaining = remaining[1:]
    if remaining:
        try:
            wager = float(remaining[0].lstrip("$"))
        except ValueError:
            logger.warning("Could not parse wager '%s' from shorthand input", remaining[0])

    return {
        "platform": platform,
        "game": "",  # Can't determine from shorthand
        "bet_type": "spread",
        "line": line,
        "odds": odds,
        "wager": wager,
    }


# --- Command handlers ---


MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [["/today", "/record"], ["/settle", "/pending"]],
    resize_keyboard=True,
)


@authorized_only
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "CBB Bet Logger\n\n"
        "Send a bet slip screenshot to log a bet.\n"
        "Or type shorthand: FD PROV +15.5 -110 1.25\n"
        "Or share a DraftKings/Kalshi link to log or settle bets.\n\n"
        "Commands:\n"
        "/pending - Show pending bets\n"
        "/settle - Settle pending bets\n"
        "/today - Today's model picks\n"
        "/record - W-L record and profit\n"
        "/delete N - Delete Nth pending bet",
        reply_markup=MAIN_KEYBOARD,
    )


@authorized_only
async def cmd_pending(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show pending bets."""
    if not os.path.exists(BETTING_HISTORY):
        await update.message.reply_text("No betting history found.")
        return

    df = pd.read_csv(BETTING_HISTORY)
    pending = df[df["result"] == "pending"]

    if len(pending) == 0:
        await update.message.reply_text("No pending bets.")
        return

    lines = [f"Pending bets: {len(pending)}\n"]
    for _, row in pending.iterrows():
        lines.append(f"  {row['date']} | {row['line']} | ${row['wager']:.2f} | {row['platform']}")

    await update.message.reply_text("\n".join(lines), reply_markup=MAIN_KEYBOARD)


@authorized_only
async def cmd_settle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Settle all pending bets and import Kalshi settlements."""
    if _settle_lock.locked():
        await update.message.reply_text("Settlement already in progress.")
        return

    async with _settle_lock:
        await update.message.reply_text("Settling bets...")

        # 1) Settle pending (FanDuel etc.) bets via score lookup
        summary = settle_pending_bets()

        msg_parts = []
        if summary["settled"]:
            msg_parts.append(f"Settled {summary['settled']}:")
            msg_parts.extend(summary["details"][:20])
        if summary["still_pending"]:
            msg_parts.append(f"Still pending: {summary['still_pending']}")

        # 2) Import settled Kalshi positions (uses persisted sync cursor)
        try:
            kalshi_result = settle_to_csv()
            logged = kalshi_result["logged"]
            kalshi_settled = kalshi_result["settled"]
            if kalshi_result["error"]:
                logger.warning("Kalshi settlement error: %s", kalshi_result["error"])
                msg_parts.append(f"Kalshi: {kalshi_result['error']}")
            elif logged or kalshi_settled:
                parts = []
                if logged:
                    total_profit = sum(float(r["profit"]) for r in logged)
                    wins = sum(1 for r in logged if r["result"] == "win")
                    losses = sum(1 for r in logged if r["result"] == "loss")
                    parts.append(f"+{len(logged)} new ({wins}W-{losses}L, {total_profit:+.2f}U)")
                if kalshi_settled:
                    parts.append(f"{kalshi_settled} pending settled")
                msg_parts.append(f"Kalshi: {', '.join(parts)}")
                for r in logged:
                    icon = {"win": "W", "loss": "L"}.get(r.get("result", ""), "?")
                    p = float(r.get("profit", 0))
                    msg_parts.append(f"  [{icon}] {r.get('line', '?')} ({r.get('game', '?')}) {p:+.2f}U")
                logger.info(
                    "Kalshi settled: %d new, %d pending updated, %d skipped",
                    len(logged), kalshi_settled, kalshi_result["skipped"],
                )
                for r in logged:
                    logger.info(
                        "  Kalshi new: %s | %s | %s | %+.2f",
                        r.get("game", "?"), r.get("line", "?"),
                        r.get("result", "?"), float(r.get("profit", 0)),
                    )
            else:
                logger.info("Kalshi settle: nothing new (skipped=%d)", kalshi_result["skipped"])
        except Exception:
            logger.exception("Kalshi settlement fetch failed")
            msg_parts.append("Kalshi: fetch failed")

        await update.message.reply_text("\n".join(msg_parts) if msg_parts else "Nothing new.", reply_markup=MAIN_KEYBOARD)


@authorized_only
async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show today's model predictions."""
    if not os.path.exists(DAILY_PREDICTIONS):
        await update.message.reply_text("No predictions file found. Run predict.py first.")
        return

    df = pd.read_csv(DAILY_PREDICTIONS)

    # Filter to value bets
    std_col = df["Std_Rating"] if "Std_Rating" in df.columns else pd.Series("PASS", index=df.index)
    rating_col = df["Rating"] if "Rating" in df.columns else pd.Series("PASS", index=df.index)
    value_bets = df[
        std_col.isin(VALUE_RATINGS)
        | rating_col.isin(VALUE_RATINGS)
    ]

    if len(value_bets) == 0:
        await update.message.reply_text(f"No value bets today ({len(df)} games total).")
        return

    lines = [f"Today's picks ({len(value_bets)} value bets):\n"]
    for _, row in value_bets.iterrows():
        conf = row.get("Conf", 0)
        pick = row.get("Pick", "")

        std_edge_str = row.get("Std_Edge_Pct", "")
        kalshi_edge_str = row.get("Edge_Pct", "")
        std_units = row.get("Std_Units", 0) or 0
        kalshi_units = row.get("Units", 0) or 0
        std_edge_val = _parse_edge_pct(std_edge_str)
        kalshi_edge_val = _parse_edge_pct(kalshi_edge_str)

        parts = [f"{pick}  {conf:.0%}"]
        if std_edge_val > 0:
            parts.append(f"DK {std_edge_str} {std_units:.1f}U")
        if kalshi_edge_val > 0:
            parts.append(f"Kalshi {kalshi_edge_str} {kalshi_units:.1f}U")
        if std_edge_val <= 0 and kalshi_edge_val <= 0:
            edge = std_edge_str or kalshi_edge_str
            if edge:
                parts.append(edge)

        lines.append(" | ".join(parts))

    # Telegram max message length is 4096 chars
    msg = "\n".join(lines)
    if len(msg) > 4000:
        msg = msg[:4000] + "\n..."

    await update.message.reply_text(msg, reply_markup=MAIN_KEYBOARD)


@authorized_only
async def cmd_record(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show W-L record and profit from betting_history.csv."""
    if not os.path.exists(BETTING_HISTORY):
        await update.message.reply_text("No betting history found.")
        return

    df = pd.read_csv(BETTING_HISTORY)
    settled = df[df["result"].isin(["win", "loss", "void"])]

    if len(settled) == 0:
        pending_count = len(df[df["result"] == "pending"])
        await update.message.reply_text(f"No settled bets yet. {pending_count} pending.")
        return

    wins = len(settled[settled["result"] == "win"])
    losses = len(settled[settled["result"] == "loss"])
    voids = len(settled[settled["result"] == "void"])
    total_profit = settled["profit"].astype(float).sum()
    total_wagered = settled["wager"].astype(float).sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    # Last 7 days
    eastern = pytz.timezone("US/Eastern")
    cutoff = (datetime.now(eastern) - timedelta(days=7)).strftime("%Y-%m-%d")
    recent = settled[settled["date"] >= cutoff]
    recent_wins = len(recent[recent["result"] == "win"])
    recent_losses = len(recent[recent["result"] == "loss"])
    recent_profit = recent["profit"].astype(float).sum() if len(recent) > 0 else 0

    pending_count = len(df[df["result"] == "pending"])

    lines = [
        "Record:\n",
        f"  All-time: {wins}W-{losses}L" + (f"-{voids}P" if voids else ""),
        f"  Profit: {total_profit:+.2f}U ({roi:+.1f}% ROI)",
        f"  Wagered: {total_wagered:.2f}U\n",
        f"  Last 7d: {recent_wins}W-{recent_losses}L, {recent_profit:+.2f}U",
        f"  Pending: {pending_count}",
    ]

    await update.message.reply_text("\n".join(lines), reply_markup=MAIN_KEYBOARD)


@authorized_only
async def cmd_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Delete the Nth pending bet. Usage: /delete N"""
    if not context.args:
        # Show numbered list of pending bets
        if not os.path.exists(BETTING_HISTORY):
            await update.message.reply_text("No betting history found.")
            return
        df = pd.read_csv(BETTING_HISTORY)
        pending = df[df["result"] == "pending"]
        if len(pending) == 0:
            await update.message.reply_text("No pending bets to delete.")
            return
        lines = ["Pending bets:\n"]
        for i, (_, row) in enumerate(pending.iterrows(), 1):
            lines.append(f"  {i}. {row['date']} | {row['line']} | ${row['wager']:.2f} | {row['platform']}")
        lines.append("\nUsage: /delete N")
        await update.message.reply_text("\n".join(lines))
        return

    try:
        n = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Usage: /delete N (where N is the pending bet number)")
        return

    df = pd.read_csv(BETTING_HISTORY)
    pending_indices = df.index[df["result"] == "pending"].tolist()

    if n < 1 or n > len(pending_indices):
        await update.message.reply_text(f"Invalid number. There are {len(pending_indices)} pending bets.")
        return

    idx = pending_indices[n - 1]
    deleted_row = df.loc[idx]
    df = df.drop(idx)
    df.to_csv(BETTING_HISTORY, index=False)

    await update.message.reply_text(
        f"Deleted pending bet #{n}: {deleted_row['line']} ${deleted_row['wager']:.2f} ({deleted_row['platform']})"
    )


# --- Message handlers ---


@authorized_only
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle bet slip screenshot via macOS OCR."""
    await update.message.reply_text("Parsing bet slip...")

    # Download the photo (highest resolution)
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    buf = BytesIO()
    await file.download_to_memory(buf)
    image_bytes = buf.getvalue()

    try:
        bets = parse_bet_screenshot(image_bytes)
    except (IOError, OSError) as e:
        logger.error(f"File I/O error during OCR: {e}")
        await update.message.reply_text(
            "Could not process image file. Please try a different screenshot."
        )
        return
    except Exception as e:
        # Log full exception for debugging, but don't expose internals to user
        logger.exception(f"Unexpected error parsing screenshot: {e}")
        await update.message.reply_text(
            "Could not parse bet slip. Please try again or use manual entry:\n"
            "FD TeamName +5.5 -110 1.25"
        )
        return

    # Handle partial bets from settled-slip fallback
    partial_bets = [b for b in bets if b.get("_partial")]
    if partial_bets:
        for bet in partial_bets:
            match = _find_matching_pending_bet(bet)
            if match:
                settle_data = {
                    "line": match["line"],
                    "wager": match["wager"],
                    "platform": match["platform"],
                    "result": bet["result"],
                    "payout": bet.get("payout", ""),
                    "profit": bet.get("profit", ""),
                }
                msg = update_bet_result(settle_data)
                if msg:
                    await update.message.reply_text(msg)
                else:
                    await update.message.reply_text("Found matching bet but could not settle.")
            else:
                teams_str = ", ".join(bet.get("_teams", ["Unknown"]))
                await update.message.reply_text(
                    f"Settled FD slip detected:\n"
                    f"  Teams: {teams_str}\n"
                    f"  Wager: ${bet['wager']:.2f}, Payout: ${bet['payout']:.2f}\n"
                    f"  Result: {bet['result'].upper()}\n\n"
                    f"Could not read the spread from OCR.\n"
                    f"Reply with: TEAM SPREAD (e.g. UTSA +12.5)"
                )
                context.user_data["partial_settled_bet"] = bet
        return

    # Handle settled bets from the new settled card parser
    settled_all = [b for b in bets if b.get("_settled")]
    non_settled_bets = [b for b in bets if not b.get("_settled") and not b.get("_partial")]
    if settled_all:
        actual_settled = [b for b in settled_all if not b.get("_skipped")]
        skipped = [b for b in settled_all if b.get("_skipped")]
        skipped_incomplete = [s for s in skipped if s.get("_skip_reason") == "incomplete"]
        skipped_dedup = [s for s in skipped if s.get("_skip_reason") == "dedup"]
        skipped_unsettled = [s for s in skipped if s.get("_skip_reason") == "unsettled"]

        msgs = []
        for bet in actual_settled:
            try:
                bet_copy = {k: v for k, v in bet.items() if not k.startswith("_")}
                if not bet_copy.get("line") or not bet_copy.get("wager"):
                    logger.warning("Settled bet missing line or wager: %s", bet_copy)
                    msgs.append(
                        f"Could not parse settled bet fully"
                        f" (bet_id={bet_copy.get('bet_id', '?')})."
                        f" Try manual settlement."
                    )
                    continue
                # Try to settle a matching pending bet first
                update_msg = update_bet_result(bet_copy)
                if update_msg:
                    msgs.append(update_msg)
                else:
                    # No pending match -- append as settled (dedup catches existing)
                    row = append_bet(bet_copy)
                    if row is None:
                        msgs.append(f"Duplicate: {bet.get('line', '?')} -- already logged.")
                    else:
                        profit_val = float(row.get("profit", 0) or 0)
                        msgs.append(
                            f"Logged ({row['result'].upper()}): {row['line']}, "
                            f"${float(row['wager']):.2f} on {row['platform']} "
                            f"(profit: {profit_val:+.2f}U)"
                        )
            except (OSError, csv.Error, ValueError) as e:
                logger.exception("I/O error logging settled bet %s: %s", bet.get("bet_id", "?"), e)
                msgs.append(f"Error writing {bet.get('line', '?')} to CSV. Try again.")
            except Exception as e:
                logger.exception("Unexpected error logging settled bet %s: %s", bet.get("bet_id", "?"), e)
                msgs.append(f"Unexpected error processing {bet.get('line', '?')}. Check bot logs.")

        # Per-card skip feedback
        if not actual_settled and not skipped_incomplete and not skipped_unsettled and skipped_dedup:
            # All cards were deduped, no new bets
            msgs.append(f"{len(skipped_dedup)} bet(s) already processed from a previous screenshot.")
        elif not actual_settled and not skipped:
            # Nothing parsed at all
            msgs.append(
                "No bets could be parsed. Cards may be partially visible"
                " -- try scrolling to show complete cards and resend."
            )
        else:
            # Show missed incomplete cards
            for s in skipped_incomplete:
                parts = []
                if s.get("bet_id"):
                    parts.append(f"BET ID {s['bet_id']}")
                if s.get("wager") and s["wager"] > 0:
                    parts.append(f"${s['wager']:.2f}")
                if s.get("result") and s["result"] != "pending":
                    parts.append(s["result"].upper())
                if parts:
                    msgs.append(f"Missed: {', '.join(parts)}")
            if skipped_incomplete:
                msgs.append("Scroll to show full cards and resend for missed bets.")
            # Show unsettled/open cards that were skipped
            if skipped_unsettled:
                n = len(skipped_unsettled)
                lines = [s.get("line", "?") for s in skipped_unsettled]
                msgs.append(
                    f"Skipped {n} open/unsettled card(s): {', '.join(lines)}. "
                    "Send again after they settle."
                )

        if msgs:
            await update.message.reply_text("\n".join(msgs))
        if not non_settled_bets:
            return
        bets = non_settled_bets

    # Filter to valid bets
    valid_bets = []
    for bet in bets:
        if bet.get("_raw_ocr") and (not bet.get("line") or not bet.get("wager")):
            await update.message.reply_text(
                f"Could not auto-parse. OCR text:\n\n{bet['_raw_ocr']}\n\n"
                "Try manual entry: FD TeamName +5.5 -110 1.25"
            )
            continue
        if not bet.get("line") or not bet.get("wager"):
            await update.message.reply_text(
                f"Missing required fields in parsed bet: {json.dumps(bet, indent=2)}"
            )
            continue
        valid_bets.append(bet)

    if not valid_bets:
        if not any(b.get("_raw_ocr") for b in bets):
            await update.message.reply_text("No valid bets found in the screenshot.")
        return

    # Single bet: log immediately
    if len(valid_bets) == 1:
        row = append_bet(valid_bets[0])
        if row is None:
            await update.message.reply_text("Duplicate bet -- already logged.")
        else:
            await update.message.reply_text(
                f"Logged: {row['line']}, {row['odds']}, ${float(row['wager']):.2f} on {row['platform']}"
            )
        return

    # Multiple bets: show numbered list and wait for selection
    lines = [f"Found {len(valid_bets)} bets:\n"]
    for i, bet in enumerate(valid_bets, 1):
        lines.append(f"  {i}. {bet['line']}  {bet['odds']}  ${bet['wager']:.2f}  ({bet['platform']})")
    lines.append("\nReply with numbers to log (e.g. '1 3') or 'all'.")

    context.user_data["pending_bets"] = valid_bets
    await update.message.reply_text("\n".join(lines))


@authorized_only
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages (shorthand bet entry, pending bet selection, or share URLs)."""
    text = update.message.text.strip()
    user = update.effective_user
    logger.info(f"Text from {user.id} ({user.username}): {text}")

    # Ignore if it starts with / (command that wasn't recognized)
    if text.startswith("/"):
        await update.message.reply_text("Unknown command. Try /start for help.")
        return

    # Handle partial settled bet completion (user providing missing spread)
    partial_bet = context.user_data.get("partial_settled_bet")
    if partial_bet:
        spread_match = re.match(r"^(.+?)\s+([+-]\d+\.?\d*)\s*$", text.strip())
        if spread_match:
            team = spread_match.group(1).strip()
            spread = spread_match.group(2)
            partial_bet["line"] = f"{team} {spread}"
            partial_bet.pop("_partial", None)
            partial_bet.pop("_teams", None)

            msg = update_bet_result(partial_bet)
            if msg:
                await update.message.reply_text(msg)
            else:
                row = append_bet(partial_bet)
                if row is None:
                    await update.message.reply_text("Duplicate bet -- already logged.")
                else:
                    profit_val = float(row.get("profit", 0) or 0)
                    await update.message.reply_text(
                        f"Logged ({row['result'].upper()}): {row['line']}, "
                        f"${float(row['wager']):.2f} on {row['platform']} "
                        f"(profit: {profit_val:+.2f}U)"
                    )
            context.user_data.pop("partial_settled_bet", None)
            return

    # Check for share URLs (DraftKings or Kalshi)
    share_url_parsers = [
        (r'https://sportsbook\.draftkings\.com/social/post/[a-zA-Z0-9-]+', "DraftKings", parse_dk_share_url),
        (r'https://kalshi\.com/markets/[^\s]+', "Kalshi", parse_kalshi_share_url),
    ]
    for pattern, platform_name, parse_fn in share_url_parsers:
        url_match = re.search(pattern, text)
        if url_match:
            await update.message.reply_text(f"Parsing {platform_name} share link...")
            bet = parse_fn(url_match.group(0))
            if bet is None:
                await update.message.reply_text(
                    f"Could not parse {platform_name} share link. Try screenshot instead."
                )
                return
            await _log_share_url_bet(update, bet)
            return

    # Handle pending multi-bet selection from a screenshot
    pending_bets = context.user_data.get("pending_bets")
    if pending_bets:
        if text.lower() == "all":
            indices = list(range(len(pending_bets)))
        else:
            try:
                indices = [int(x) - 1 for x in text.split() if x.isdigit()]
            except ValueError:
                indices = []

        if indices:
            logged = []
            dupes = 0
            for idx in indices:
                if 0 <= idx < len(pending_bets):
                    row = append_bet(pending_bets[idx])
                    if row is None:
                        dupes += 1
                    else:
                        logged.append(
                            f"Logged: {row['line']}, {row['odds']}, ${float(row['wager']):.2f} on {row['platform']}"
                        )
            msgs = []
            if logged:
                msgs.extend(logged)
            if dupes:
                msgs.append(f"{dupes} duplicate(s) skipped.")
            context.user_data.pop("pending_bets", None)
            await update.message.reply_text("\n".join(msgs) if msgs else "No bets logged.")
            return

    bet = parse_shorthand(text)
    if bet is None:
        await update.message.reply_text(
            "Could not parse. Use format:\n"
            "PLATFORM TEAM SPREAD ODDS WAGER\n"
            "Example: FD Providence +15.5 -110 1.25"
        )
        return

    row = append_bet(bet)
    if row is None:
        await update.message.reply_text("Duplicate bet -- already logged.")
        return
    await update.message.reply_text(
        f"Logged: {row['line']}, {row['odds']}, ${float(row['wager']):.2f} on {row['platform']}"
    )


LOCK_FILE = os.path.join(BASE_DIR, ".telegram_bot.lock")


def _acquire_instance_lock():
    """Acquire an exclusive lock to prevent multiple bot instances.

    Returns the open file handle (must stay open for the lock to hold).
    Exits the process if another instance is already running.
    """
    try:
        lock_fh = open(LOCK_FILE, "w")
    except OSError as e:
        logger.error("Cannot create lock file %s: %s", LOCK_FILE, e)
        sys.exit(1)

    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            logger.error("Another bot instance is already running.")
        else:
            logger.error("Could not acquire lock on %s: %s", LOCK_FILE, e)
        lock_fh.close()
        sys.exit(1)

    try:
        lock_fh.write(str(os.getpid()))
        lock_fh.flush()
    except OSError as e:
        logger.warning("Could not write PID to lock file (non-fatal): %s", e)

    return lock_fh


def _register_scheduled_jobs(job_queue):
    """Register scheduled jobs on the bot's job queue."""
    eastern = pytz.timezone("US/Eastern")
    job_queue.run_daily(
        _reminder_check_unlogged,
        time=datetime.strptime("06:00", "%H:%M").time().replace(tzinfo=eastern),
        job_kwargs={"misfire_grace_time": None},
    )
    logger.info("Scheduled daily unlogged-bet reminder at 6:00 AM ET")


def main():
    """Start the Telegram bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set in .env")
        print("Create a bot via @BotFather on Telegram and add the token to .env")
        return

    lock_fh = _acquire_instance_lock()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("pending", cmd_pending))
    app.add_handler(CommandHandler("settle", cmd_settle))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("record", cmd_record))
    app.add_handler(CommandHandler("delete", cmd_delete))

    # Messages
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Scheduled jobs
    if app.job_queue is not None:
        _register_scheduled_jobs(app.job_queue)
    else:
        logger.warning(
            "JobQueue not available -- install python-telegram-bot[job-queue] "
            "for scheduled reminders"
        )

    print("Bot started. Send /start in Telegram to begin.")
    try:
        app.run_polling()
    finally:
        try:
            lock_fh.close()
        except OSError as e:
            logger.warning("Error closing lock file handle: %s", e)
        try:
            os.unlink(LOCK_FILE)
        except OSError as e:
            logger.warning("Could not remove lock file %s: %s", LOCK_FILE, e)


if __name__ == "__main__":
    main()
