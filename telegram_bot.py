"""
Telegram bot for logging bets and settling them.

Screenshot a bet slip -> share to this bot -> auto-parsed and logged.
Commands: /settle, /pending, /today, /record

Usage:
    python telegram_bot.py
"""

import os
import re
import csv
import json
import tempfile
import logging
from datetime import datetime, timedelta
from io import BytesIO

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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# CSV headers for betting_history.csv
CSV_HEADERS = ["date", "platform", "game", "bet_type", "line", "odds", "wager", "result", "payout", "profit"]

# Short team abbreviations that are valid (not junk OCR text)
VALID_SHORT_TEAMS = {"TCU", "USC", "LSU", "SMU", "UCF", "UNC", "UAB", "FIU", "BYU", "UIC"}

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
    re.compile(r"^\$\d+\.\d+\s*\+?\s*$"),  # balance display like "$12.10 +"
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


def append_bet(bet: dict):
    """Append a bet row to betting_history.csv. Returns None if duplicate."""
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
        "result": "pending",
        "payout": "",
        "profit": "",
    }

    # Duplicate detection: check for matching (normalized line, wager, platform)
    if os.path.exists(BETTING_HISTORY):
        new_key = (_normalize_line(row["line"]), str(row["wager"]), row["platform"].upper())
        with open(BETTING_HISTORY, "r", newline="") as f:
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

    with open(BETTING_HISTORY, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)

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

    # Clean junk lines before parsing
    cleaned = _clean_ocr_text(lines)
    raw_text = "\n".join(cleaned)
    logger.info(f"OCR cleaned text:\n{raw_text}")

    return _parse_bet_slip_text(raw_text)


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
        wager_match = re.search(r"Wager:\s*\$?(\d+\.?\d*)", block)
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


def _parse_bet_slip_text(text: str) -> list[dict]:
    """Parse raw OCR text from a bet slip into structured bet data."""
    # Detect platform
    text_lower = text.lower()
    if "draftkings" in text_lower:
        platform = "DraftKings"
    elif "fanduel" in text_lower:
        platform = "FanDuel"
    elif "kalshi" in text_lower or "wins by over" in text_lower:
        platform = "Kalshi"
    elif "betmgm" in text_lower:
        platform = "BetMGM"
    elif "caesars" in text_lower:
        platform = "Caesars"
    # FanDuel heuristics -- FD screenshots often lack the "FanDuel" word
    elif "spread betting" in text_lower or "total wager" in text_lower:
        platform = "FanDuel"
    # DraftKings heuristics -- DK format uses "Wager:" prefix
    elif re.search(r"Wager:\s*\$", text):
        platform = "DraftKings"
    else:
        platform = "Unknown"

    # Use platform-specific parser when available
    if platform == "DraftKings":
        bets = _parse_dk_blocks(text)
        if bets:
            return bets

    if platform == "FanDuel":
        bets = _parse_fd_blocks(text)
        if bets:
            return bets

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


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "CBB Bet Logger\n\n"
        "Send a bet slip screenshot to log a bet.\n"
        "Or type shorthand: FD PROV +15.5 -110 1.25\n\n"
        "Commands:\n"
        "/pending - Show pending bets\n"
        "/settle - Settle pending bets\n"
        "/today - Today's model picks\n"
        "/record - W-L record and profit\n"
        "/delete N - Delete Nth pending bet"
    )


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


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show today's model predictions."""
    if not os.path.exists(DAILY_PREDICTIONS):
        await update.message.reply_text("No predictions file found. Run predict.py first.")
        return

    df = pd.read_csv(DAILY_PREDICTIONS)

    # Filter to value bets (STRONG or GOOD)
    value_bets = df[
        (df.get("Std_Rating", pd.Series(dtype=str)).isin(["STRONG", "GOOD"]))
        | (df.get("Rating", pd.Series(dtype=str)).isin(["STRONG", "GOOD"]))
    ]

    if len(value_bets) == 0:
        await update.message.reply_text(f"No value bets today ({len(df)} games total).")
        return

    lines = [f"Today's picks ({len(value_bets)} value bets):\n"]
    for _, row in value_bets.iterrows():
        rating = row.get("Std_Rating", row.get("Rating", ""))
        units = row.get("Std_Units", row.get("Units", 0))
        conf = row.get("Conf", 0)
        edge = row.get("Std_Edge_Pct", row.get("Edge_Pct", ""))
        pick = row.get("Pick", "")

        tag = "[S]" if rating == "STRONG" else "[G]"
        lines.append(f"{tag} {pick}  {conf:.0%} | {edge} | {units:.1f}U")

    # Telegram max message length is 4096 chars
    msg = "\n".join(lines)
    if len(msg) > 4000:
        msg = msg[:4000] + "\n..."

    await update.message.reply_text(msg)


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
    except Exception as e:
        logger.error(f"Error parsing screenshot: {e}")
        await update.message.reply_text(f"Could not parse bet slip: {e}")
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


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages (shorthand bet entry or pending bet selection)."""
    text = update.message.text.strip()

    # Ignore if it starts with / (command that wasn't recognized)
    if text.startswith("/"):
        await update.message.reply_text("Unknown command. Try /start for help.")
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


def main():
    """Start the Telegram bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set in .env")
        print("Create a bot via @BotFather on Telegram and add the token to .env")
        return

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
    app.run_polling()


if __name__ == "__main__":
    main()
