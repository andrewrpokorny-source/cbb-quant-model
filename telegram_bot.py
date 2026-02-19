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
import json
import errno
import fcntl
import base64
import tempfile
import logging
import logging.handlers
import functools
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import urlparse, parse_qs

import httpx
import pytz
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from ocrmac import ocrmac

from settle_bets import settle_pending_bets

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BETTING_HISTORY = os.path.join(BASE_DIR, "betting_history.csv")
DAILY_PREDICTIONS = os.path.join(BASE_DIR, "daily_predictions.csv")
PERF_FILE = os.path.join(BASE_DIR, "performance_log.csv")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# User authorization: comma-separated list of allowed Telegram user IDs
_allowed_users_str = os.getenv("TELEGRAM_ALLOWED_USERS", "")
ALLOWED_USER_IDS = set(int(uid.strip()) for uid in _allowed_users_str.split(",") if uid.strip())


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
CSV_HEADERS = ["date", "platform", "game", "bet_type", "line", "odds", "wager", "result", "payout", "profit"]

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


def ensure_csv_exists():
    """Create betting_history.csv with headers if it doesn't exist."""
    if not os.path.exists(BETTING_HISTORY):
        with open(BETTING_HISTORY, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


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


def update_bet_result(bet: dict) -> str | None:
    """
    Update a pending bet with its result, matching by line and wager.

    Returns a status message, or None if no matching pending bet found.
    Uses a single file handle with lock held throughout to avoid TOCTOU races.
    """
    if not os.path.exists(BETTING_HISTORY):
        return None

    new_key = (_normalize_line(bet["line"]), str(bet["wager"]), bet["platform"].upper())

    with open(BETTING_HISTORY, "r+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            reader = list(csv.DictReader(f))

            # Find matching pending bet
            matched_idx = None
            for i, row in enumerate(reader):
                if row.get("result") != "pending":
                    continue
                existing_key = (
                    _normalize_line(row.get("line", "")),
                    str(row.get("wager", "")),
                    row.get("platform", "").upper(),
                )
                if existing_key == new_key:
                    matched_idx = i
                    break

            if matched_idx is None:
                return None

            # Update the matched row
            reader[matched_idx]["result"] = bet["result"]
            reader[matched_idx]["payout"] = bet.get("payout", "")
            reader[matched_idx]["profit"] = bet.get("profit", "")

            # Write back while still holding the lock
            f.seek(0)
            f.truncate()
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(reader)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    result = bet["result"]
    profit = bet.get("profit", 0)
    return f"Updated: {bet['line']} -> {result.upper()} ({profit:+.2f}U)"


def _find_matching_pending_bet(partial_bet: dict) -> dict | None:
    """Match a partial settled bet against pending bets in history.

    Matches by platform + wager, then narrows by team name if needed.
    Returns the matched row as a dict, or None.
    """
    if not os.path.exists(BETTING_HISTORY):
        return None

    df = pd.read_csv(BETTING_HISTORY)
    pending = df[df["result"] == "pending"]
    if len(pending) == 0:
        return None

    platform = partial_bet["platform"].upper()
    wager = float(partial_bet["wager"])

    matches = pending[
        (pending["platform"].str.upper() == platform)
        & (abs(pending["wager"].astype(float) - wager) < 0.01)
    ]

    if len(matches) == 0:
        return None

    # Verify team name against every candidate (even single matches)
    teams = partial_bet.get("_teams", [])
    for _, row in matches.iterrows():
        line_upper = str(row.get("line", "")).upper()
        game_upper = str(row.get("game", "")).upper()
        for team in teams:
            if team.upper() in line_upper or team.upper() in game_upper:
                return row.to_dict()

    # Single match with no team info -- accept it as best guess
    if len(matches) == 1 and not teams:
        return matches.iloc[0].to_dict()

    logger.info(
        f"Could not narrow pending bet match: {len(matches)} candidates, "
        f"teams={teams}"
    )
    return None


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
    }

    # Use file locking to prevent race conditions
    with open(BETTING_HISTORY, "r+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Check for duplicates while holding the lock
            new_key = (_normalize_line(row["line"]), str(row["wager"]), row["platform"].upper())
            reader = csv.DictReader(f)
            for existing in reader:
                existing_key = (
                    _normalize_line(existing.get("line", "")),
                    str(existing.get("wager", "")),
                    existing.get("platform", "").upper(),
                )
                if existing_key == new_key:
                    logger.info(f"Duplicate bet skipped: {row['line']} {row['wager']} {row['platform']}")
                    return None

            # Seek to end and append while still holding the lock
            f.seek(0, 2)
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return row


def parse_bet_screenshot(image_bytes: bytes) -> list[dict]:
    """
    Use macOS native OCR (ocrmac) to extract text from a bet slip screenshot,
    then parse the text into structured bet data.
    Returns a list of parsed bet dicts.
    """
    # Write bytes to a temp file for ocrmac
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        results = ocrmac.OCR(tmp_path, recognition_level="accurate").recognize()
    finally:
        os.unlink(tmp_path)

    # Combine OCR text lines (confidence threshold 0.5)
    lines = [text for text, confidence, bbox in results if confidence > 0.5]
    logger.info(f"OCR raw lines:\n{chr(10).join(lines)}")

    # Detect platform from RAW text (before cleaning removes identifiers)
    raw_text_for_detection = "\n".join(lines)
    platform = _detect_platform(raw_text_for_detection)
    logger.info(f"Detected platform: {platform}")

    # Clean junk lines before parsing
    cleaned = _clean_ocr_text(lines)
    cleaned_text = "\n".join(cleaned)
    logger.info(f"OCR cleaned text:\n{cleaned_text}")

    raw_text = "\n".join(lines)
    return _parse_bet_slip_text(cleaned_text, platform=platform, raw_text=raw_text)


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
    if payout > wager:
        result = "win"
    elif payout == 0 or "returned" in raw_text.lower():
        result = "void" if "returned" in raw_text.lower() else "loss"
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
    spread_line_re = re.compile(
        r"^([A-Za-z][A-Za-z &'.\-]+?)\s+([+-]\d+\.?\d*)\s*$", re.MULTILINE
    )
    matches = list(spread_line_re.finditer(text))
    if not matches:
        return []

    bets = []
    for i, match in enumerate(matches):
        team = match.group(1).strip()
        spread = match.group(2)

        # Block of text between this spread line and the next
        block_start = match.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[block_start:block_end]

        # Matchup: "Team1 @ Team2"
        matchup_match = re.search(
            r"([A-Za-z][A-Za-z &'.]+?)\s+@\s+([A-Za-z][A-Za-z &'.]+)",
            block,
        )
        if matchup_match:
            game = f"{matchup_match.group(1).strip()} vs {matchup_match.group(2).strip()}"
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
        })

    return bets


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


def _parse_bet_slip_text(text: str, platform: str = None, raw_text: str = None) -> list[dict]:
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

        status = bet_data.get("status", "pending")
        # Map DK statuses to our format (win/loss/void)
        status_map = {"won": "win", "lost": "loss", "pushed": "void",
                      "voided": "void", "cashout": "void"}
        result = status_map.get(status, status)

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
        payout = 0.0
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
            wager = float(remaining[0])
        except ValueError:
            pass

    return {
        "platform": platform,
        "game": "",  # Can't determine from shorthand
        "bet_type": "spread",
        "line": line,
        "odds": odds,
        "wager": wager,
    }


# --- Command handlers ---


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
        "/delete N - Delete Nth pending bet"
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

    await update.message.reply_text("\n".join(lines))


@authorized_only
async def cmd_settle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Settle all pending bets."""
    await update.message.reply_text("Settling pending bets...")

    summary = settle_pending_bets()

    msg_parts = [
        f"Settled: {summary['settled']}",
        f"Still pending: {summary['still_pending']}",
    ]
    if summary["details"]:
        msg_parts.append("")
        msg_parts.extend(summary["details"][:20])  # Cap at 20 lines

    await update.message.reply_text("\n".join(msg_parts))


@authorized_only
async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show today's model predictions."""
    if not os.path.exists(DAILY_PREDICTIONS):
        await update.message.reply_text("No predictions file found. Run predict.py first.")
        return

    df = pd.read_csv(DAILY_PREDICTIONS)

    # Filter to value bets (STRONG only)
    value_bets = df[
        (df.get("Std_Rating", pd.Series(dtype=str)) == "STRONG")
        | (df.get("Rating", pd.Series(dtype=str)) == "STRONG")
    ]

    if len(value_bets) == 0:
        await update.message.reply_text(f"No value bets today ({len(df)} games total).")
        return

    lines = [f"Today's picks ({len(value_bets)} value bets):\n"]
    for _, row in value_bets.iterrows():
        units = row.get("Std_Units", row.get("Units", 0))
        conf = row.get("Conf", 0)
        edge = row.get("Std_Edge_Pct", row.get("Edge_Pct", ""))
        pick = row.get("Pick", "")

        lines.append(f"{pick}  {conf:.0%} | {edge} | {units:.1f}U")

    # Telegram max message length is 4096 chars
    msg = "\n".join(lines)
    if len(msg) > 4000:
        msg = msg[:4000] + "\n..."

    await update.message.reply_text(msg)


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

    await update.message.reply_text("\n".join(lines))


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
