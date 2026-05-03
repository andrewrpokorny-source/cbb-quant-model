import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import os
import altair as alt
import html as html_mod
import predict
import backtest
import mlb.predict as mlb_predict
import io
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pytz
import csv
import re
import requests
from betting import format_line_shopping_text, VALUE_RATINGS, RATING_RANK
from league_config import get_league_artifact_paths, get_league_settings, get_scoreboard_base_url, normalize_league
from prediction_io import load_predictions_csv
from dashboard_helpers import filter_recent_kalshi, position_matches_game, utc_date_to_eastern

# --- PATH CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BET_HIST_FILE = os.path.join(BASE_DIR, "data/betting_history.csv")

LEAGUES = ["mlb", "mens", "womens"]

st.set_page_config(page_title="Quant Edge", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --green-900: #0a1f12;
    --green-800: #0f2b1c;
    --green-700: #1a4d2e;
    --green-600: #2a6b42;
    --green-500: #3d8a5a;
    --green-100: #e6f0ea;
    --purple-800: #2d1b4e;
    --purple-700: #4a2d7a;
    --purple-600: #6b42a6;
    --purple-100: #ede6f5;
    --gold-700: #7a5c10;
    --gold-600: #8a6d1b;
    --gold-100: #faf3e0;
    --neutral-900: #1a1e1a;
    --neutral-700: #3a3e3a;
    --neutral-500: #6b7c6b;
    --neutral-400: #8a9a8a;
    --neutral-200: #e0dfd8;
    --neutral-100: #eeeee8;
    --neutral-50: #f7f6f2;
    --live: #b5342a;
    --surface: #ffffff;
    --bg: #faf9f5;
    --font-display: 'Newsreader', Georgia, serif;
    --font-body: 'Plus Jakarta Sans', system-ui, sans-serif;
    --font-mono: 'IBM Plex Mono', 'JetBrains Mono', monospace;
}

/* Base styles */
.stApp {
    background-color: var(--bg);
}

.main .block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f2f1ec;
    width: 240px !important;
    border-right: 1px solid #e0dfd8;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]),
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    font-family: var(--font-body);
    color: var(--neutral-700) !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--neutral-900) !important;
    font-family: var(--font-body) !important;
}

/* Sidebar widget overrides for light background */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: var(--neutral-500) !important;
}

section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
section[data-testid="stSidebar"] .stTextInput input {
    background: white;
    color: var(--neutral-900);
    border-color: var(--neutral-200);
}

section[data-testid="stSidebar"] [data-testid="stExpander"] {
    border-color: var(--neutral-200);
    background: white;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: var(--neutral-700) !important;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: var(--green-700);
    color: white;
    border: none;
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.82rem;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    transition: all 0.15s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--green-600);
}

.sidebar-brand {
    font-family: var(--font-body);
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--green-800);
    margin-bottom: 0.25rem;
    letter-spacing: -0.01em;
    line-height: 1.1;
}

.sidebar-sub {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--neutral-400);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.5rem;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid var(--neutral-200);
    margin: 1rem 0;
}

/* Typography overrides */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: var(--font-body) !important;
    color: var(--green-900) !important;
    letter-spacing: -0.02em;
}

p, span, div, .stMarkdown {
    font-family: var(--font-body);
}

/* League column headers */
.league-header {
    font-family: var(--font-body);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    padding-bottom: 0.5rem;
    margin-bottom: 0.75rem;
    border-bottom: 3px solid;
}

.league-header.mens {
    color: var(--green-800);
    border-color: var(--green-700);
}

.league-header.womens {
    color: var(--purple-800);
    border-color: var(--purple-700);
}

/* Section headers */
.section-title, .league-header {
    scroll-margin-top: 4rem;
}
.section-title {
    font-family: var(--font-body);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--neutral-500);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 1.5rem 0 0.5rem 0;
    padding-top: 0.5rem;
    border-top: 2px solid var(--neutral-200);
}

/* Value bet cards */
.bet-card {
    background: var(--surface);
    border: 1px solid var(--neutral-200);
    border-radius: 10px;
    padding: 1rem 1.1rem 0.9rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.bet-card:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.07);
    transform: translateY(-1px);
}

.bet-card.strong {
    border-left: 4px solid var(--green-700);
}

.bet-card.strong.womens-card {
    border-left-color: var(--purple-700);
}

.bet-badge {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 4px;
}

.bet-badge.strong {
    background: var(--green-700);
    color: #ffffff;
}

.womens-card .bet-badge.strong {
    background: var(--purple-700);
}

.bet-card.good {
    border-left: 4px solid var(--gold-600);
}

.bet-badge.good {
    background: var(--gold-600);
    color: #ffffff;
}

.bet-card.marginal {
    border-left: 4px solid #7a5c2e;
}

.bet-badge.marginal {
    background: #7a5c2e;
    color: #ffffff;
}

.bet-card.pass {
    border-left: 4px solid var(--neutral-400);
}

.bet-badge.pass {
    background: var(--neutral-400);
    color: #ffffff;
}

.placed-badge {
    font-family: var(--font-mono);
    font-size: 0.55rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 4px;
    background: #1a6fb5;
    color: #ffffff;
    display: inline-block;
    margin-left: 8px;
}

.bet-pick {
    font-family: var(--font-body);
    font-size: 1rem;
    font-weight: 700;
    color: var(--green-900);
    margin: 2px 0;
    letter-spacing: -0.01em;
}

.bet-matchup {
    font-family: var(--font-body);
    font-size: 0.75rem;
    color: var(--neutral-500);
}

.bet-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--neutral-100);
}

.stat-item {
    display: flex;
    flex-direction: column;
    gap: 1px;
}

.stat-label {
    font-family: var(--font-mono);
    font-size: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--neutral-400);
}

.stat-value {
    font-family: var(--font-mono);
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--green-900);
}

.strong .stat-value.positive { color: var(--green-600); }
.womens-card.strong .stat-value.positive { color: var(--purple-600); }
.good .stat-value.positive { color: var(--gold-700); }
.marginal .stat-value.positive { color: #7a5c2e; }
.pass .stat-value.positive { color: var(--neutral-500); }
.stat-value.fee-value { color: var(--neutral-500); }

/* Kalshi badge */
.kalshi-row {
    background: var(--neutral-50);
    border: 1px solid var(--neutral-200);
    border-radius: 6px;
    padding: 6px 10px;
    margin-top: 8px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--green-900);
}

.kalshi-label {
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--neutral-500);
    margin-right: 6px;
}

.kalshi-link {
    color: var(--green-600);
    text-decoration: none;
    font-size: 0.72rem;
    font-weight: 500;
}
.kalshi-link:hover {
    text-decoration: underline;
    color: var(--green-700);
}

/* Polymarket badge */
.poly-row {
    background: var(--neutral-50);
    border: 1px solid #c7d2fe;
    border-radius: 6px;
    padding: 6px 10px;
    margin-top: 4px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: #312e81;
}
.poly-link {
    color: #6366f1;
    text-decoration: none;
    font-size: 0.72rem;
    font-weight: 500;
}
.poly-link:hover {
    text-decoration: underline;
    color: #4f46e5;
}

/* Summary KPI row */
.kpi-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.25rem;
}

.kpi-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--neutral-200);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}

.kpi-value {
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--green-800);
    line-height: 1.1;
}

.kpi-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--neutral-400);
    margin-top: 4px;
}

/* Bet header with time */
.bet-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}

.bet-time {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--neutral-400);
}

/* Refresh button (main area) */
.main .stButton > button {
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.82rem;
    background: var(--green-700);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.45rem 1.2rem;
    transition: all 0.15s ease;
}

.main .stButton > button:hover {
    background: var(--green-600);
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(26,77,46,0.25);
}

/* Expander styling */
.streamlit-expanderHeader {
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--green-900);
    background: var(--neutral-50);
    border-radius: 8px;
}

/* Code blocks for line shopping */
.stCodeBlock {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    background: var(--neutral-50) !important;
    border: 1px solid var(--neutral-200) !important;
    border-radius: 8px !important;
}

/* Metrics */
.stMetric {
    background: var(--surface);
    padding: 0.75rem;
    border-radius: 10px;
    border: 1px solid var(--neutral-200);
}

.stMetric label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid var(--neutral-200);
    margin: 1rem 0;
}

/* Dataframe */
.stDataFrame {
    font-family: var(--font-body);
}

.stDataFrame [data-testid="stDataFrameResizable"] {
    border: 1px solid var(--neutral-200);
    border-radius: 10px;
    overflow: hidden;
}

/* Live position cards */
.live-card {
    background: var(--surface);
    border: 1px solid var(--neutral-200);
    border-left: 4px solid var(--live);
    border-radius: 10px;
    padding: 1rem 1.1rem 0.9rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.live-card:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.07);
    transform: translateY(-1px);
}

.live-card.womens-card {
    border-left-color: var(--purple-700);
}

.live-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.live-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: var(--live);
    border-radius: 50%;
    margin-right: 5px;
    animation: pulse-dot 1.5s ease-in-out infinite;
}

.womens-card .live-dot {
    background: var(--purple-700);
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.live-badge {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--live);
    display: inline-flex;
    align-items: center;
}

.womens-card .live-badge {
    color: var(--purple-700);
}

.live-clock {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--live);
}

.womens-card .live-clock {
    color: var(--purple-700);
}

.live-score-row {
    display: flex;
    justify-content: center;
    align-items: baseline;
    gap: 0.6rem;
    margin: 8px 0;
}

.live-team {
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--green-900);
    flex: 1;
}

.live-team.away { text-align: right; }
.live-team.home { text-align: left; }

.live-score {
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--green-900);
    letter-spacing: -0.02em;
    flex-shrink: 0;
}

.live-bet-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--neutral-100);
}

.live-bet-stats .stat-item {
    display: flex;
    flex-direction: column;
    gap: 1px;
}

.live-bet-stats .stat-label {
    font-family: var(--font-mono);
    font-size: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--neutral-400);
}

.live-bet-stats .stat-value {
    font-family: var(--font-mono);
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--green-900);
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# HELPERS
# ==========================================

_KALSHI_SERIES_SLUGS = {
    "KXNCAAMBSPREAD": "mens-college-basketball-spread",
    "KXNCAAMBGAME": "mens-college-basketball-mens-game",
    "KXNCAAWBSPREAD": "womens-college-basketball-spread",
    "KXNCAAWBGAME": "college-basketball-womens-game",
    "KXMLBGAME": "baseball-game",
    "KXMLBSPREAD": "baseball-spread",
    "KXMLBTOTAL": "baseball-total",
}


def kalshi_event_url(ticker) -> str:
    if not ticker or not isinstance(ticker, str):
        return ""
    parts = ticker.rsplit("-", 1)
    event_ticker = parts[0] if len(parts) > 1 else ticker
    series = event_ticker.split("-", 1)[0]
    slug = _KALSHI_SERIES_SLUGS.get(series.upper(), "")
    if not slug:
        return f"https://kalshi.com/markets/{series.lower()}"
    return f"https://kalshi.com/markets/{series.lower()}/{slug}/{event_ticker.lower()}"


def polymarket_event_url(slug_or_id) -> str:
    """Build a Polymarket event URL from an event slug or token ID."""
    if not slug_or_id or not isinstance(slug_or_id, str):
        return ""
    return f"https://polymarket.com/event/{slug_or_id}"


def _esc(val) -> str:
    """HTML-escape a value for safe embedding in st.markdown."""
    return html_mod.escape(str(val)) if val is not None else ""


def _filter_not_started(df):
    """Drop rows whose game time has already passed (in-progress games).

    Handles two Date/Time formats:
      CBB: "04/04 08:49 PM"  (Eastern, no year)
      MLB: "2026-04-04 17:10" (UTC, ISO with year)

    Rows with unparseable Date/Time values are kept (never silently hidden).
    """
    if df.empty or 'Date/Time' not in df.columns:
        return df
    eastern = pytz.timezone('US/Eastern')
    utc = pytz.utc
    now = datetime.now(eastern)
    now_naive = now.replace(tzinfo=None)
    year = now.year
    dt_strings = df["Date/Time"].astype(str)

    # Parse CBB format: "04/04 08:49 PM" (Eastern, needs year prefix)
    cbb_parsed = pd.to_datetime(
        str(year) + "/" + dt_strings,
        format="%Y/%m/%d %I:%M %p",
        errors="coerce",
    )
    # Year-rollover: Dec viewing Jan games -- bump year (pre-localization, naive compare)
    needs_bump = cbb_parsed.notna() & (cbb_parsed < (now_naive - timedelta(days=60)))
    if needs_bump.any():
        cbb_parsed = cbb_parsed.where(
            ~needs_bump,
            pd.to_datetime(
                str(year + 1) + "/" + dt_strings,
                format="%Y/%m/%d %I:%M %p",
                errors="coerce",
            ),
        )

    # Parse MLB/ISO format for rows CBB didn't match: "2026-04-04 17:10" (UTC)
    iso_parsed = pd.to_datetime(dt_strings, format="%Y-%m-%d %H:%M", errors="coerce")
    # Only use ISO for rows where CBB parse failed
    iso_parsed = iso_parsed.where(cbb_parsed.isna())

    # Localize each format to its source timezone, then combine as tz-aware
    cbb_aware = cbb_parsed.dt.tz_localize(eastern, ambiguous="NaT", nonexistent="NaT")
    iso_aware = iso_parsed.dt.tz_localize(utc, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(eastern)

    # Combine: prefer CBB where available, fill with ISO
    parsed = cbb_aware.combine_first(iso_aware)

    n_failed = parsed.isna().sum()
    if n_failed > 0:
        bad = df.loc[parsed.isna(), "Date/Time"].unique()
        print(f"      WARNING: _filter_not_started: {n_failed} row(s) with "
              f"unparseable Date/Time (kept): {list(bad[:5])}")
    return df[parsed.isna() | (parsed > now)].copy()


def _format_odds_display(x):
    """Format American odds for display, preserving +/- sign."""
    try:
        if pd.isna(x) or str(x).strip().lower() in ('', 'nan'):
            return ''
        return f"{int(float(x)):+d}"
    except (ValueError, TypeError):
        return ''


def _parse_edge(series):
    """Parse edge percentage strings like '+5.2%' into floats."""
    cleaned = series.fillna('').astype(str).str.rstrip('%').str.lstrip('+')
    return pd.to_numeric(cleaned, errors='coerce').fillna(0)


# ==========================================
# LIVE KALSHI POSITIONS
# ==========================================

POSITION_PREFIXES = (
    "KXNCAAMBGAME", "KXNCAAWBGAME",
    "KXNCAAMBSPREAD", "KXNCAAWBSPREAD",
    "KXMLB",
)


@st.cache_data(ttl=120)
def _fetch_kalshi_positions():
    """Fetch unsettled Kalshi positions, filtered to CBB markets.

    Returns:
        List of position dicts. Each contains dollar-denominated string
        fields: market_exposure_dollars, fees_paid_dollars, position_fp.
        Returns [] on failure.
    """
    try:
        from kalshi.client import KalshiClient
        client = KalshiClient()
        if not client.api_key:
            return []
        positions = client.get_positions(settlement_status="unsettled")
        return [p for p in positions if any(p.get("ticker", "").startswith(pfx) for pfx in POSITION_PREFIXES)]
    except Exception as e:
        print(f"      Failed to fetch Kalshi positions: {e}")
        return []


def _event_ticker(ticker) -> str:
    """Extract event ticker (drop market suffix) for game-level matching."""
    if not ticker or not isinstance(ticker, str):
        return ""
    parts = ticker.rsplit("-", 1)
    return parts[0] if len(parts) > 1 else ticker


def _build_position_lookup(positions: list[dict]) -> dict:
    """Build event_ticker -> list of (side, contracts) for 'action on game' display."""
    lookup: dict[str, list[dict]] = {}
    for pos in positions:
        ticker = pos.get("ticker", "")
        et = _event_ticker(ticker)
        if not et:
            continue
        position_fp = float(pos.get("position_fp", 0) or 0)
        if position_fp == 0:
            continue
        side = "YES" if position_fp > 0 else "NO"
        contracts = int(abs(position_fp))
        cost = float(pos.get("market_exposure_dollars", 0) or 0)
        fee = float(pos.get("fees_paid_dollars", 0) or 0)
        basis = round(cost + fee, 2)
        lookup.setdefault(et, []).append({"side": side, "contracts": contracts, "basis": basis})
    return lookup


def _placed_badge_html(kalshi_ticker: str, position_lookup: dict) -> str:
    """Return HTML for 'PLACED' badge if user has a position on this game, else ''."""
    if not kalshi_ticker or not position_lookup:
        return ""
    et = _event_ticker(kalshi_ticker)
    positions = position_lookup.get(et, [])
    if not positions:
        return ""
    parts = []
    for p in positions:
        parts.append(f"{p['contracts']}x {p['side']} @ ${p['basis']:.2f}")
    label = ", ".join(parts)
    return f'<span class="placed-badge">PLACED: {_esc(label)}</span>'


def _format_position_col(ticker: str, position_lookup: dict) -> str:
    """Return plain-text position summary for dataframe columns (e.g. '5x YES')."""
    if not ticker or not position_lookup:
        return ""
    et = _event_ticker(ticker)
    positions = position_lookup.get(et, [])
    if not positions:
        return ""
    parts = [f"{p['contracts']}x {p['side']} @ ${p['basis']:.2f}" for p in positions]
    return ", ".join(parts)


@st.cache_data(ttl=60)
def _fetch_live_espn_games():
    """Fetch in-progress ESPN games, keyed by team abbreviation.

    Returns dict[str, list[dict]] -- multiple games can share the same
    abbreviation across leagues (e.g. MLB 'CIN' and CBB 'CIN').
    """
    games_by_abbr: dict[str, list[dict]] = {}
    for lg in LEAGUES:
        try:
            url = get_scoreboard_base_url(lg)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"      ESPN scoreboard fetch failed for {lg}: {e}")
            continue
        for event in data.get("events", []):
            status = event.get("status", {})
            state = status.get("type", {}).get("state", "")
            if state != "in":
                continue
            competitions = event.get("competitions", [])
            if not competitions:
                continue
            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[0])
            home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[1])

            clock = status.get("displayClock", "")
            period = status.get("period", 0)
            desc = status.get("type", {}).get("description", "")

            if lg == "mlb":
                # MLB: period = inning, clock = outs or top/bot indicator
                inning_half = status.get("type", {}).get("shortDetail", "")
                clock_display = inning_half if inning_half else f"Inn {period}"
            elif "halftime" in desc.lower():
                clock_display = "HALF"
            elif period > 2:
                clock_display = f"{clock} OT" if clock else "OT"
            else:
                half_label = f"{period}H" if period else ""
                clock_display = f"{clock} {half_label}".strip()

            # Extract game date from ESPN event (UTC ISO -> Eastern date)
            event_date_raw = comp.get("date") or event.get("date", "")
            game_date = utc_date_to_eastern(event_date_raw)

            game_info = {
                "away_abbr": away.get("team", {}).get("abbreviation", ""),
                "away_name": away.get("team", {}).get("shortDisplayName", ""),
                "away_score": away.get("score", "0"),
                "home_abbr": home.get("team", {}).get("abbreviation", ""),
                "home_name": home.get("team", {}).get("shortDisplayName", ""),
                "home_score": home.get("score", "0"),
                "clock": clock_display,
                "league": lg,
                "game_date": game_date,
            }
            if game_info["away_abbr"]:
                games_by_abbr.setdefault(game_info["away_abbr"], []).append(game_info)
            if game_info["home_abbr"]:
                games_by_abbr.setdefault(game_info["home_abbr"], []).append(game_info)
    return games_by_abbr


def _build_live_positions(positions, live_games) -> list[dict]:
    """Match Kalshi positions to live ESPN games."""
    from kalshi.client import KalshiClient
    client = KalshiClient()
    results = []
    for pos in positions:
        ticker = pos.get("ticker", "")
        # Ticker suffix after last '-' contains the YES team abbreviation.
        # Game tickers use the bare abbreviation (e.g. "MIZZ"), while spread
        # tickers append a spread number (e.g. "SMC8"), so extract only
        # the leading letters.
        parts = ticker.rsplit("-", 1)
        if len(parts) < 2:
            continue
        tail = parts[1].upper()
        abbr_match = re.match(r"([A-Z]+)", tail)
        if not abbr_match:
            continue
        yes_abbr = abbr_match.group(1)

        # Determine league from ticker prefix to avoid cross-league mismatches
        ticker_upper = ticker.upper()
        if "KXMLB" in ticker_upper:
            ticker_league = "mlb"
        elif "AAWB" in ticker_upper:
            ticker_league = "womens"
        else:
            ticker_league = "mens"

        # Match position to live game -- try all candidates for this abbreviation
        candidates = live_games.get(yes_abbr, [])
        game = None
        for candidate in candidates:
            if position_matches_game(ticker, candidate, ticker_league):
                game = candidate
                break
        if not game:
            continue

        # position_fp > 0 means YES contracts held, < 0 means NO; 0 means no active position
        position_fp = float(pos.get("position_fp", 0) or 0)
        if position_fp > 0:
            side = "YES"
        elif position_fp < 0:
            side = "NO"
        else:
            continue

        # Resolve full team name from the game data
        if yes_abbr == game["home_abbr"]:
            yes_team_name = game["home_name"]
        elif yes_abbr == game["away_abbr"]:
            yes_team_name = game["away_name"]
        else:
            yes_team_name = yes_abbr

        contracts = int(abs(position_fp))
        cost = float(pos.get("market_exposure_dollars", 0) or 0)
        fee = float(pos.get("fees_paid_dollars", 0) or 0)
        net_cost = cost + fee

        # Determine market type (game vs spread) from ticker
        market_type = "spread" if "SPREAD" in ticker.upper() else "game"

        # Fetch current market bid price (what we'd get if we sold now)
        current_price = None
        try:
            market = client.get_market(ticker)
            if side == "YES":
                bid = market.get("yes_bid_dollars")
            else:
                bid = market.get("no_bid_dollars")
            if bid is not None:
                current_price = float(bid)
        except (requests.RequestException, ValueError, TypeError) as e:
            print(f"      Failed to fetch market price for {ticker}: {e}")

        # Unrealized P&L: what the position is worth now vs what was paid
        pnl = None
        if current_price is not None:
            pnl = round(current_price * contracts - net_cost, 2)

        results.append({
            "ticker": ticker,
            "side": side,
            "side_team": yes_team_name,
            "contracts": contracts,
            "net_cost": net_cost,
            "game": game,
            "market_type": market_type,
            "current_price": current_price,
            "pnl": pnl,
        })
    return results


# ==========================================
# LOAD ALL LEAGUES (parallel)
# ==========================================
league_data = {}


def _run_predictions(lg, spread_overrides=None):
    """Run prediction pipeline for a single league. Thread-safe."""
    if lg == "mlb":
        mlb_predict.generate_predictions(lg)
    else:
        predict.main(
            spread_overrides=spread_overrides or {},
            league=lg,
        )


# Determine which leagues need a prediction run
_leagues_to_run = []
_league_artifacts = {}
# Read spread overrides on the main thread (st.session_state is thread-local)
_spread_overrides = {}
for lg in LEAGUES:
    paths = get_league_artifact_paths(BASE_DIR, lg)
    loaded_key = f"predictions_loaded_{lg}"
    pred_file = paths["predictions_file"]
    has_artifacts = os.path.exists(paths["model_file"]) and os.path.exists(paths["data_file"])
    _league_artifacts[lg] = {"paths": paths, "has_artifacts": has_artifacts}
    _spread_overrides[lg] = st.session_state.get(f"spread_overrides_{lg}", {})

    if has_artifacts:
        if os.path.exists(pred_file):
            eastern = pytz.timezone('US/Eastern')
            file_date = datetime.fromtimestamp(os.path.getmtime(pred_file)).date()
            today_date = datetime.now(eastern).date()
            if file_date < today_date:
                st.session_state[loaded_key] = False
        if not st.session_state.get(loaded_key, False):
            _leagues_to_run.append(lg)

# Run predictions in parallel (MLB is independent; mens/womens serialize via _RUNTIME_LOCK)
if _leagues_to_run:
    labels = [get_league_settings(lg)["label"] for lg in _leagues_to_run]
    with st.spinner(f"Loading {', '.join(labels)}..."):
        _errors = {}
        with ThreadPoolExecutor(max_workers=len(_leagues_to_run)) as executor:
            futures = {
                executor.submit(_run_predictions, lg, _spread_overrides[lg]): lg
                for lg in _leagues_to_run
            }
            for future in as_completed(futures):
                lg = futures[future]
                pred_file = _league_artifacts[lg]["paths"]["predictions_file"]
                try:
                    future.result()
                    if os.path.exists(pred_file):
                        st.session_state[f"predictions_loaded_{lg}"] = True
                except Exception as e:
                    _errors[lg] = e
        for lg, e in _errors.items():
            st.error(f"Failed to load {get_league_settings(lg)['label']} predictions: {e}")

# Build league_data from prediction CSVs
for lg in LEAGUES:
    settings = get_league_settings(lg)
    paths = _league_artifacts[lg]["paths"]
    has_artifacts = _league_artifacts[lg]["has_artifacts"]
    pred_file = paths["predictions_file"]

    try:
        df, predictions_status = load_predictions_csv(pred_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
        st.warning(f"Could not read {settings['label']} predictions: {e}")
        df = pd.DataFrame()
        predictions_status = "error"

    # Out-of-season / empty predictions are surfaced inside each league's own
    # section (e.g. "No matching picks") rather than as a banner above the MLB
    # content. Keeps the active in-season league at the top.

    if "Bet_Type" in df.columns:
        spread_df = df[df["Bet_Type"] != "game"].copy()
        game_df = df[df["Bet_Type"] == "game"].copy()
    elif not df.empty:
        spread_df = df.copy()
        game_df = pd.DataFrame(columns=df.columns)
    else:
        spread_df = pd.DataFrame()
        game_df = pd.DataFrame()

    spread_df = _filter_not_started(spread_df)
    game_df = _filter_not_started(game_df)

    league_data[lg] = {
        "settings": settings,
        "paths": paths,
        "df": df,
        "spread_df": spread_df,
        "game_df": game_df,
        "predictions_ls": predict.get_latest_predictions(lg) if lg != "mlb" else None,
        "games_needing_spreads": predict.get_games_needing_spreads(lg) if lg != "mlb" else {},
    }


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Quant Edge</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">CBB + MLB | Kalshi Markets</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
<style>
.nav-link, .nav-link:visited, .nav-link:link {
    display: block;
    padding: 5px 10px;
    margin: 1px 0;
    color: #4a5a4a !important;
    text-decoration: none !important;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    border-radius: 5px;
    transition: background 0.15s, color 0.15s;
}
.nav-link:hover {
    background: rgba(0,0,0,0.05);
    color: #1a4d2e !important;
    text-decoration: none !important;
}
.nav-section-label {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #aab5aa;
    padding: 0 10px 4px;
}
</style>
<div class="nav-section-label">Navigate</div>
<a href="#mlb-moneyline" class="nav-link">MLB Moneyline</a>
<a href="#cbb-spread-bets" class="nav-link">CBB Spread Bets</a>
<a href="#cbb-game-bets" class="nav-link">CBB Game Bets</a>
<a href="#full-slates" class="nav-link">Full Slates</a>
<a href="#performance" class="nav-link">Performance</a>
<a href="#betting-history" class="nav-link">Betting History</a>
""", unsafe_allow_html=True)

    # Missing spreads per league
    for lg in LEAGUES:
        games = league_data[lg]["games_needing_spreads"]
        if games:
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            n = len(games)
            label = league_data[lg]["settings"]["label"]
            st.caption(f"{label}: {n} missing spread{'s' if n != 1 else ''}")
            with st.expander(f"Enter spreads ({label})", expanded=False):
                with st.form(f"spread_overrides_form_{lg}"):
                    override_inputs = {}
                    for game in games:
                        matchup = game['matchup']
                        st.markdown(f"**{matchup}**")
                        override_inputs[matchup] = st.text_input(
                            matchup,
                            placeholder="-7.5 or +3",
                            key=f"spread_{lg}_{game['id']}",
                            label_visibility="collapsed",
                        )
                    submitted = st.form_submit_button("Apply & Re-run")
                    if submitted:
                        overrides_key = f"spread_overrides_{lg}"
                        overrides = st.session_state.get(overrides_key, {})
                        for matchup, val in override_inputs.items():
                            val = val.strip()
                            if val:
                                try:
                                    overrides[matchup] = float(val)
                                except ValueError:
                                    st.error(f"Invalid: {val}")
                        st.session_state[overrides_key] = overrides
                        st.session_state[f"predictions_loaded_{lg}"] = False
                        st.rerun()


# ==========================================
# SPREAD BETS: Men's | Women's side by side
# ==========================================


def _render_spread_bets(col, lg, position_lookup=None):
    """Render spread value bets for one league."""
    d = league_data[lg]
    spread_df = d["spread_df"]
    predictions_ls = d["predictions_ls"]
    label = d["settings"]["label"]
    card_extra = "womens-card" if lg == "womens" else ""

    with col:
        st.markdown(
            f'<div class="league-header {lg}">{label}</div>',
            unsafe_allow_html=True,
        )

        spread_value_bets = pd.DataFrame(columns=spread_df.columns) if not spread_df.empty else pd.DataFrame()
        if not spread_df.empty and 'Std_Rating' in spread_df.columns:
            spread_rating = spread_df['Rating'] if 'Rating' in spread_df.columns else pd.Series('PASS', index=spread_df.index)
            poly_spread_rating = spread_df['Poly_Rating'] if 'Poly_Rating' in spread_df.columns else pd.Series('PASS', index=spread_df.index)
            spread_value_bets = spread_df[
                (spread_df['Std_Rating'].isin(VALUE_RATINGS)) |
                (spread_rating.isin(VALUE_RATINGS)) |
                (poly_spread_rating.isin(VALUE_RATINGS))
            ].copy()

        if len(spread_value_bets) > 0:
            poly_edge = _parse_edge(spread_value_bets['Poly_Edge_Pct']) if 'Poly_Edge_Pct' in spread_value_bets.columns else pd.Series(0.0, index=spread_value_bets.index)
            spread_value_bets['_best_edge'] = np.maximum(
                np.maximum(
                    _parse_edge(spread_value_bets['Std_Edge_Pct']),
                    _parse_edge(spread_value_bets['Edge_Pct']),
                ),
                poly_edge,
            )
            spread_value_bets = (
                spread_value_bets
                .sort_values('_best_edge', ascending=False)
                .drop(columns=['_best_edge'])
            )

            for _, row in spread_value_bets.iterrows():
                std_rating = row.get('Std_Rating', 'PASS')
                kalshi_rating = row.get('Rating', 'PASS') if pd.notna(row.get('Rating')) else 'PASS'
                conf = row['Conf']
                game_time = row.get('Date/Time', '')

                poly_rating = row.get('Poly_Rating', 'PASS') if pd.notna(row.get('Poly_Rating')) else 'PASS'
                std_rank_val = RATING_RANK.get(std_rating, 0)
                kalshi_rank_val = RATING_RANK.get(kalshi_rating, 0)
                poly_rank_val = RATING_RANK.get(poly_rating, 0)
                best_rank = max(std_rank_val, kalshi_rank_val, poly_rank_val)
                if kalshi_rank_val == best_rank:
                    best_rating = kalshi_rating
                elif poly_rank_val == best_rank:
                    best_rating = poly_rating
                else:
                    best_rating = std_rating
                badge_css = best_rating.lower()

                if kalshi_rank_val == best_rank and kalshi_rank_val > 0:
                    display_edge = row.get('Edge_Pct', 'N/A')
                    display_units = row.get('Units', 0) or 0
                    edge_source = "Kalshi Edge"
                elif poly_rank_val == best_rank and poly_rank_val > 0:
                    display_edge = row.get('Poly_Edge_Pct', 'N/A')
                    display_units = row.get('Poly_Units', 0) or 0
                    edge_source = "Poly Edge"
                else:
                    display_edge = row.get('Std_Edge_Pct', 'N/A')
                    display_units = row.get('Std_Units', 0) or 0
                    edge_source = "Edge"

                breakeven = row.get('Breakeven_Spread', None)
                breakeven_str = f"{breakeven:+.1f}" if breakeven and pd.notna(breakeven) else "---"

                odds_str = _format_odds_display(row.get('Std_Odds', '')) or "---"

                kalshi_side = row.get('Kalshi_Side')
                kalshi_price = row.get('Kalshi_Price')
                has_kalshi = pd.notna(kalshi_side) and kalshi_side

                kalshi_html = ""
                kalshi_ticker = row.get('Kalshi_Ticker', '')
                if has_kalshi:
                    kalshi_edge = _esc(row.get('Edge_Pct', 'N/A'))
                    kalshi_fee = row.get('Kalshi_Fee')
                    fee_str = f" + {_esc(kalshi_fee)}&#162; fee" if pd.notna(kalshi_fee) and kalshi_fee else ""
                    cbb_pick = _esc(row.get('Pick', ''))
                    kalshi_text = f"Kalshi: Buy {_esc(kalshi_side)} @ {_esc(kalshi_price)}&#162;{fee_str} &middot; {kalshi_edge}"
                    if kalshi_ticker:
                        kalshi_url = _esc(kalshi_event_url(kalshi_ticker))
                        kalshi_html = f'<div class="kalshi-row"><a href="{kalshi_url}" target="_blank" class="kalshi-link">{kalshi_text}</a></div>'
                    else:
                        kalshi_html = f'<div class="kalshi-row"><span class="kalshi-label">{kalshi_text}</span></div>'

                # Polymarket spread display
                poly_side = row.get('Poly_Side')
                poly_price = row.get('Poly_Price')
                has_poly = pd.notna(poly_side) and poly_side if isinstance(poly_side, str) else False

                poly_html = ""
                if has_poly:
                    poly_edge = _esc(row.get('Poly_Edge_Pct', 'N/A'))
                    poly_fee = row.get('Poly_Fee')
                    poly_fee_str = f" + {_esc(poly_fee)}&#162; fee" if pd.notna(poly_fee) and poly_fee else ""
                    poly_ticker = row.get('Poly_Ticker', '')
                    poly_text = f"Poly: Buy {_esc(poly_side)} @ {_esc(poly_price)}&#162;{poly_fee_str} &middot; {poly_edge}"
                    if poly_ticker:
                        poly_url = _esc(polymarket_event_url(poly_ticker))
                        poly_html = f'<div class="poly-row"><a href="{poly_url}" target="_blank" class="poly-link">{poly_text}</a></div>'
                    else:
                        poly_html = f'<div class="poly-row"><span class="kalshi-label">{poly_text}</span></div>'

                placed_html = _placed_badge_html(kalshi_ticker, position_lookup or {})

                st.markdown(f'''
                <div class="bet-card {badge_css} {card_extra}">
                    <div class="bet-header">
                        <div><span class="bet-badge {badge_css}">{_esc(best_rating.title())}</span>{placed_html}</div>
                        <div class="bet-time">{_esc(game_time)}</div>
                    </div>
                    <div class="bet-pick">{_esc(row['Pick'])}</div>
                    <div class="bet-matchup">{_esc(row['Matchup'])}</div>
                    <div class="bet-stats">
                        <div class="stat-item">
                            <span class="stat-label">Model</span>
                            <span class="stat-value">{conf:.1%}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">{_esc(edge_source)}</span>
                            <span class="stat-value positive">{_esc(display_edge)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Odds</span>
                            <span class="stat-value">{_esc(odds_str)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Units</span>
                            <span class="stat-value">{display_units:.1f}U</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Breakeven</span>
                            <span class="stat-value">{_esc(breakeven_str)}</span>
                        </div>
                    </div>
                    {kalshi_html}
                    {poly_html}
                </div>
                ''', unsafe_allow_html=True)

                line_shopping_data = None
                if predictions_ls is not None:
                    match = predictions_ls[
                        predictions_ls['Matchup'] == row['Matchup']
                    ]
                    if len(match) > 0 and 'Line_Shopping_Data' in match.columns:
                        line_shopping_data = match.iloc[0].get('Line_Shopping_Data')

                if line_shopping_data is not None:
                    with st.expander("Line Shopping", expanded=False):
                        st.code(format_line_shopping_text(line_shopping_data), language=None)
        else:
            st.caption("No spread value bets on this slate.")


def _render_game_bets(col, lg, position_lookup=None):
    """Render Kalshi ML game value bets for one league."""
    d = league_data[lg]
    game_df = d["game_df"]
    label = d["settings"]["label"]
    card_extra = "womens-card" if lg == "womens" else ""

    with col:
        st.markdown(
            f'<div class="league-header {lg}">{label}</div>',
            unsafe_allow_html=True,
        )

        if game_df.empty:
            st.caption("No ML game markets on this slate.")
            return

        kalshi_game_val = game_df['Rating'].isin(VALUE_RATINGS) if 'Rating' in game_df.columns else pd.Series(False, index=game_df.index)
        poly_game_val = game_df['Poly_Rating'].isin(VALUE_RATINGS) if 'Poly_Rating' in game_df.columns else pd.Series(False, index=game_df.index)
        game_value = game_df[kalshi_game_val | poly_game_val].copy()
        if game_value.empty:
            st.caption("No ML game markets on this slate.")
            return

        game_value['_edge_sort'] = _parse_edge(game_value['Edge_Pct']) if 'Edge_Pct' in game_value.columns else 0
        game_value = game_value.sort_values('_edge_sort', ascending=False).drop(columns=['_edge_sort'])
        for _, row in game_value.iterrows():
            rating = row.get("Rating", "PASS")
            badge_css = str(rating).lower()
            game_kalshi_ticker = row.get('Kalshi_Ticker', '')
            game_fee = row.get('Kalshi_Fee')
            game_fee_val = float(game_fee) if pd.notna(game_fee) and game_fee else 0.0
            game_price_val = float(row.get('Kalshi_Price', 0) or 0)
            net_cost = game_price_val + game_fee_val
            game_fee_str = f" + {game_fee_val:.1f}&#162; fee" if game_fee_val else ""
            game_pick = _esc(row.get('Pick', ''))
            game_side = _esc(row.get('Kalshi_Side', ''))
            game_kalshi_text = f"Kalshi: Buy {game_side} @ {_esc(row.get('Kalshi_Price', ''))}&#162;{game_fee_str}"
            game_kalshi_url = _esc(kalshi_event_url(game_kalshi_ticker))
            if game_kalshi_url:
                game_kalshi_html = f'<a href="{game_kalshi_url}" target="_blank" class="kalshi-link">{game_kalshi_text}</a>'
            else:
                game_kalshi_html = f'<span class="kalshi-label">{game_kalshi_text}</span>'
            # Polymarket game display
            game_poly_side = row.get('Poly_Side')
            game_has_poly = pd.notna(game_poly_side) and game_poly_side if isinstance(game_poly_side, str) else False
            game_poly_html = ""
            if game_has_poly:
                game_poly_edge = _esc(row.get('Poly_Edge_Pct', 'N/A'))
                game_poly_fee = row.get('Poly_Fee')
                game_poly_fee_str = f" + {float(game_poly_fee):.1f}&#162; fee" if pd.notna(game_poly_fee) and game_poly_fee else ""
                game_poly_ticker = row.get('Poly_Ticker', '')
                game_poly_text = f"Poly: Buy {_esc(game_poly_side)} @ {_esc(row.get('Poly_Price', ''))}&#162;{game_poly_fee_str} &middot; {game_poly_edge}"
                if game_poly_ticker:
                    game_poly_url = _esc(polymarket_event_url(game_poly_ticker))
                    game_poly_html = f'<div class="poly-row"><a href="{game_poly_url}" target="_blank" class="poly-link">{game_poly_text}</a></div>'
                else:
                    game_poly_html = f'<div class="poly-row"><span class="kalshi-label">{game_poly_text}</span></div>'

            placed_html = _placed_badge_html(game_kalshi_ticker, position_lookup or {})
            st.markdown(f'''
            <div class="bet-card {badge_css} {card_extra}">
                <div class="bet-header">
                    <div><span class="bet-badge {badge_css}">{_esc(str(rating).title())}</span>{placed_html}</div>
                    <div class="bet-time">{_esc(row.get('Date/Time', ''))}</div>
                </div>
                <div class="bet-pick">{_esc(row.get('Pick', ''))}</div>
                <div class="bet-matchup">{_esc(row.get('Matchup', ''))}</div>
                <div class="bet-stats">
                    <div class="stat-item">
                        <span class="stat-label">Model</span>
                        <span class="stat-value">{row.get('Conf', 0):.1%}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Kalshi Edge</span>
                        <span class="stat-value positive">{_esc(row.get('Edge_Pct', ''))}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Units</span>
                        <span class="stat-value">{float(row.get('Units', 0) or 0):.1f}U</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Fee</span>
                        <span class="stat-value fee-value">{game_fee_val:.1f}&#162;</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Net Cost</span>
                        <span class="stat-value fee-value">{net_cost:.1f}&#162;</span>
                    </div>
                </div>
                <div class="kalshi-row">{game_kalshi_html}</div>
                {game_poly_html}
            </div>
            ''', unsafe_allow_html=True)


# ==========================================
# LIVE KALSHI POSITIONS (in-progress games)
# ==========================================
try:
    _kalshi_positions = _fetch_kalshi_positions()
    _live_games = _fetch_live_espn_games() if _kalshi_positions else {}
    _live_positions = _build_live_positions(_kalshi_positions, _live_games) if _kalshi_positions else []
except Exception as e:
    print(f"      Live positions section error: {e}")
    _live_positions = []

# Auto-refresh: keep running even after live positions disappear so that
# recently-settled results are picked up without a manual browser refresh.
if "live_auto_refresh" not in st.session_state:
    st.session_state.live_auto_refresh = True

if st.session_state.live_auto_refresh:
    st_autorefresh(interval=60_000, key="live_autorefresh")

if _live_positions:
    st.markdown('<div class="section-title">Live Positions</div>', unsafe_allow_html=True)

    # Refresh controls
    ctrl_cols = st.columns([1, 1, 6])
    with ctrl_cols[0]:
        if st.button("Refresh now", key="live_refresh_btn"):
            _fetch_kalshi_positions.clear()
            _fetch_live_espn_games.clear()
            st.rerun()
    with ctrl_cols[1]:
        auto_on = st.toggle("Auto-refresh", value=st.session_state.live_auto_refresh, key="live_auto_toggle")
        st.session_state.live_auto_refresh = auto_on

    live_cols = st.columns(min(len(_live_positions), 3))
    for i, lp in enumerate(_live_positions):
        g = lp["game"]
        col = live_cols[i % len(live_cols)]
        league_label = "W" if g["league"] == "womens" else "M"
        type_label = "SPR" if lp["market_type"] == "spread" else "ML"
        card_extra = "womens-card" if g["league"] == "womens" else ""

        # P&L display with color
        pnl = lp.get("pnl")
        if pnl is not None:
            pnl_color = "var(--green-600)" if pnl >= 0 else "var(--live)"
            pnl_html = f'<span style="color:{pnl_color};font-weight:700">{pnl:+.2f}</span>'
        else:
            pnl_html = '<span style="color:var(--neutral-400)">--</span>'

        mkt_html = f'${lp["current_price"]:.2f}' if lp.get("current_price") is not None else '--'

        size_label = f'{lp["contracts"]}x ' if lp["contracts"] > 1 else ''

        kalshi_url = kalshi_event_url(lp["ticker"])
        kalshi_link = f'<a href="{_esc(kalshi_url)}" target="_blank" style="color:var(--neutral-400);font-size:0.65rem;text-decoration:none">view</a>' if kalshi_url else ''

        with col:
            st.markdown(f'''
            <div class="live-card {card_extra}">
                <div class="live-header">
                    <span class="live-badge"><span class="live-dot"></span>LIVE {_esc(league_label)} {_esc(type_label)}</span>
                    <span class="live-clock">{_esc(g["clock"])} {kalshi_link}</span>
                </div>
                <div class="live-score-row">
                    <span class="live-team away">{_esc(g["away_name"])}</span>
                    <span class="live-score">{_esc(g["away_score"])} &ndash; {_esc(g["home_score"])}</span>
                    <span class="live-team home">{_esc(g["home_name"])}</span>
                </div>
                <div class="live-bet-stats">
                    <div class="stat-item">
                        <span class="stat-label">Position</span>
                        <span class="stat-value">{_esc(size_label)}{_esc(lp["side"])} {_esc(lp["side_team"])}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Cost</span>
                        <span class="stat-value">${lp["net_cost"]:.2f}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Mkt</span>
                        <span class="stat-value">{mkt_html}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">P&amp;L</span>
                        <span class="stat-value">{pnl_html}</span>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

# Recent Kalshi results (last 7 days)
_recent_kalshi = []
if os.path.exists(BET_HIST_FILE):
    try:
        with open(BET_HIST_FILE, "r", newline="") as _f:
            _all_bets = list(csv.DictReader(_f))
        _recent_kalshi = filter_recent_kalshi(_all_bets)
    except (OSError, csv.Error, UnicodeDecodeError, ValueError) as e:
        print(f"      Failed to read recent Kalshi results: {e}")

if _recent_kalshi:
    st.markdown('<div class="section-title">Recent Kalshi Results</div>', unsafe_allow_html=True)
    _rows_html = ""
    for _r in _recent_kalshi:
        _res = _r.get("result", "").strip().lower()
        _profit = float(_r.get("profit", 0) or 0)
        if _res == "win":
            _res_style = "color:var(--green-600);font-weight:700"
            _res_label = "W"
        elif _res == "loss":
            _res_style = "color:var(--live);font-weight:700"
            _res_label = "L"
        else:
            _res_style = "color:var(--neutral-500)"
            _res_label = "V"
        _pnl_color = "var(--green-600)" if _profit >= 0 else "var(--live)"
        _rows_html += f'''
        <tr style="border-bottom:1px solid var(--neutral-100)">
            <td style="padding:6px 8px;color:var(--neutral-500)">{_esc(_r.get("date", "")[5:])}</td>
            <td style="padding:6px 8px">{_esc(_r.get("game", ""))}</td>
            <td style="padding:6px 8px">{_esc(_r.get("line", ""))}</td>
            <td style="padding:6px 8px;text-align:center"><span style="{_res_style}">{_res_label}</span></td>
            <td style="padding:6px 8px;text-align:right;color:{_pnl_color};font-weight:600">{_profit:+.2f}</td>
        </tr>'''
    st.markdown(f'''
    <table style="width:100%;font-family:var(--font-mono);font-size:0.78rem;border-collapse:collapse;margin-bottom:0.5rem">
        <thead>
            <tr style="border-bottom:1px solid var(--neutral-200);color:var(--neutral-400);font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em">
                <th style="text-align:left;padding:4px 8px">Date</th>
                <th style="text-align:left;padding:4px 8px">Game</th>
                <th style="text-align:left;padding:4px 8px">Line</th>
                <th style="text-align:center;padding:4px 8px">Result</th>
                <th style="text-align:right;padding:4px 8px">P&L</th>
            </tr>
        </thead>
        <tbody>{_rows_html}
        </tbody>
    </table>
    ''', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

_position_lookup = _build_position_lookup(_fetch_kalshi_positions())

st.markdown('<div class="section-title" id="mlb-moneyline">MLB -- Moneyline Picks</div>', unsafe_allow_html=True)
_render_game_bets(st.container(), "mlb", _position_lookup)

st.markdown('<div class="section-title" id="cbb-spread-bets">CBB -- Spread Bets</div>', unsafe_allow_html=True)
col_spread_m, col_spread_w = st.columns(2)
_render_spread_bets(col_spread_m, "mens", _position_lookup)
_render_spread_bets(col_spread_w, "womens", _position_lookup)

st.markdown('<div class="section-title" id="cbb-game-bets">CBB -- Kalshi Game Bets (ML)</div>', unsafe_allow_html=True)
col_game_m, col_game_w = st.columns(2)
_render_game_bets(col_game_m, "mens", _position_lookup)
_render_game_bets(col_game_w, "womens", _position_lookup)


# ==========================================
# KPI ROW
# ==========================================
total_value = 0
total_units = 0.0
total_kalshi = 0
record_str = "--"
profit_str = "--"
roi_str = "--"

for lg in LEAGUES:
    d = league_data[lg]
    df = d["df"]
    if df.empty or 'Std_Rating' not in df.columns:
        continue
    rating_col = df['Rating'] if 'Rating' in df.columns else pd.Series('PASS', index=df.index)
    vb = df[(df['Std_Rating'].isin(VALUE_RATINGS)) | (rating_col.isin(VALUE_RATINGS))]
    total_value += len(vb)
    if len(vb) > 0:
        total_units += np.maximum(vb['Std_Units'].fillna(0), vb['Units'].fillna(0)).sum()
    gdf = d["game_df"]
    if not gdf.empty and 'Rating' in gdf.columns:
        total_kalshi += len(gdf[gdf['Rating'].isin(VALUE_RATINGS)])

if os.path.exists(BET_HIST_FILE):
    try:
        _bh = pd.read_csv(BET_HIST_FILE)
        _settled = _bh[_bh["result"].isin(["win", "loss", "void"])]
        if len(_settled) > 0:
            _wins = len(_settled[_settled["result"] == "win"])
            _losses = len(_settled[_settled["result"] == "loss"])
            _profit = pd.to_numeric(_settled["profit"], errors="coerce").fillna(0).sum()
            _wagered = pd.to_numeric(_settled["wager"], errors="coerce").fillna(0).sum()
            _roi = (_profit / _wagered * 100) if _wagered > 0 else 0
            record_str = f"{_wins}W-{_losses}L"
            profit_str = f"{_profit:+.1f}U"
            roi_str = f"{_roi:+.1f}%"
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
        st.warning(f"Could not read betting history: {e}")

st.markdown(f'''
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-value">{total_value}</div>
        <div class="kpi-label">Value Bets</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{total_units:.1f}U</div>
        <div class="kpi-label">To Deploy</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{total_kalshi}</div>
        <div class="kpi-label">ML Bets</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{record_str}</div>
        <div class="kpi-label">Record</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{profit_str}</div>
        <div class="kpi-label">Profit</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{roi_str}</div>
        <div class="kpi-label">ROI</div>
    </div>
</div>
''', unsafe_allow_html=True)


# ==========================================
# BOTTOM: Record + Performance
# ==========================================
st.markdown("<hr>", unsafe_allow_html=True)

col_record, col_perf = st.columns(2)

with col_record:
    st.markdown('<div class="league-header mens" id="betting-history">Betting Record</div>', unsafe_allow_html=True)

    if os.path.exists(BET_HIST_FILE):
        try:
            bet_hist = pd.read_csv(BET_HIST_FILE)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            st.warning(f"Could not read betting history: {e}")
            bet_hist = None

        if bet_hist is not None:
            pending_bets = bet_hist[bet_hist["result"] == "pending"]
            pending_count = len(pending_bets)

            if pending_count > 0:
                st.caption(f"{pending_count} pending bet{'s' if pending_count != 1 else ''}")
                pending_display = pending_bets[["date", "platform", "line", "odds", "wager"]].copy()
                pending_display["wager"] = pd.to_numeric(pending_display["wager"], errors="coerce").apply(lambda x: f"${x:.2f}" if pd.notna(x) else "--")
                st.dataframe(pending_display, use_container_width=True, hide_index=True, height=180)

            settled = bet_hist[bet_hist["result"].isin(["win", "loss", "void"])]
            if len(settled) > 0:
                wins = len(settled[settled["result"] == "win"])
                losses = len(settled[settled["result"] == "loss"])
                total_profit = pd.to_numeric(settled["profit"], errors="coerce").fillna(0).sum()
                total_wagered = pd.to_numeric(settled["wager"], errors="coerce").fillna(0).sum()
                roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Record", f"{wins}W-{losses}L")
                rc2.metric("Profit", f"{total_profit:+.2f}U")
                rc3.metric("ROI", f"{roi:+.1f}%")

                recent = settled.tail(10).iloc[::-1].copy()
                recent["Result"] = recent["result"].apply(
                    lambda x: {"win": "W", "loss": "L", "void": "P"}.get(x, x)
                )
                recent["P/L"] = pd.to_numeric(recent["profit"], errors="coerce").apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "--")
                display_cols = ["date", "platform", "line", "odds", "wager", "Result", "P/L"]
                st.dataframe(recent[display_cols], use_container_width=True, hide_index=True, height=280)
            elif pending_count == 0:
                st.caption("No betting history yet.")
    else:
        st.caption("No betting history file found.")

with col_perf:
    st.markdown('<div class="league-header mens" id="performance">Model Performance</div>', unsafe_allow_html=True)

    def get_metrics(df_subset):
        if len(df_subset) == 0:
            return 0, 0.0, 0.0
        df_subset = df_subset.copy()
        df_subset['units'] = df_subset['pick_correct'].apply(lambda x: 1.0 if x else -1.1)
        cnt = len(df_subset)
        wins = df_subset['pick_correct'].sum()
        rate = wins / cnt
        profit = df_subset['units'].sum()
        return cnt, rate, profit

    perf_tabs = st.tabs([league_data[lg]["settings"]["label"] for lg in LEAGUES])

    for pi, lg in enumerate(LEAGUES):
        with perf_tabs[pi]:
            perf_file = league_data[lg]["paths"]["performance_file"]
            if os.path.exists(perf_file):
                try:
                    hist = pd.read_csv(perf_file)
                except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
                    st.warning(f"Could not read performance data: {e}")
                    hist = None
                if hist is None:
                    continue
                hist['date'] = pd.to_datetime(hist['date'])

                try:
                    today = pd.Timestamp.now(tz='US/Eastern').normalize()
                except (pytz.exceptions.UnknownTimeZoneError, TypeError):
                    today = pd.Timestamp.now().normalize() - timedelta(hours=5)

                yesterday = today - timedelta(days=1)
                start_7 = today - timedelta(days=7)

                df_yesterday = hist[hist['date'].dt.date == yesterday.date()]
                df_7 = hist[hist['date'].dt.date >= start_7.date()]
                df_30 = hist

                pc1, pc2, pc3 = st.columns(3)
                cnt_y, rate_y, prof_y = get_metrics(df_yesterday)
                pc1.metric("Yesterday", f"{prof_y:+.1f}U", f"{cnt_y} bets | {rate_y:.0%}")

                cnt_7, rate_7, prof_7 = get_metrics(df_7)
                pc2.metric("7 Days", f"{prof_7:+.1f}U", f"{cnt_7} bets | {rate_7:.0%}")

                cnt_30, rate_30, prof_30 = get_metrics(df_30)
                pc3.metric("All Time", f"{prof_30:+.1f}U", f"{cnt_30} bets | {rate_30:.0%}")

                hist['units'] = hist['pick_correct'].apply(lambda x: 1.0 if x else -1.1)
                hist['cumulative_units'] = hist['units'].cumsum()

                line_color = '#1a4d2e' if lg == 'mens' else '#4a2d7a'
                chart = alt.Chart(hist).mark_area(
                    line={'color': line_color},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[
                            alt.GradientStop(color=f'{line_color}1a', offset=0),
                            alt.GradientStop(color=f'{line_color}4d', offset=1)
                        ],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    x=alt.X('date:T', title=None, axis=alt.Axis(format='%b %d', labelAngle=0)),
                    y=alt.Y('cumulative_units:Q', title='Units'),
                    tooltip=[
                        alt.Tooltip('date:T', title='Date', format='%b %d'),
                        alt.Tooltip('cumulative_units:Q', title='Total Units', format='.1f')
                    ]
                ).properties(height=220).configure_axis(
                    grid=True,
                    gridColor='#e5e5e0',
                    domainColor='#e5e5e0'
                ).configure_view(
                    strokeWidth=0
                )
                st.altair_chart(chart, use_container_width=True)

                with st.expander("View Bet History"):
                    hist['Result'] = hist['pick_correct'].apply(lambda x: "W" if x else "L")
                    hist['Date_Str'] = hist['date'].dt.strftime("%b %d")
                    hist['Spread'] = hist['picked_spread'].apply(lambda x: round(x * 2) / 2)
                    hist['Pick'] = hist['picked_team'] + " " + hist['Spread'].astype(str)

                    df_display = hist.sort_values('date', ascending=False)
                    df_display = df_display[['Date_Str', 'Pick', 'Result', 'conf']].rename(
                        columns={'Date_Str': 'Date', 'conf': 'Conf'}
                    )
                    df_display['Conf'] = df_display['Conf'].apply(lambda x: f"{x:.0%}")

                    st.dataframe(df_display, use_container_width=True, hide_index=True)

            else:
                st.caption("No performance data yet.")
                if st.button("Run Backtest", key=f"backtest_{lg}"):
                    with st.spinner("Training models..."):
                        f = io.StringIO()
                        try:
                            with redirect_stdout(f):
                                backtest.run_backtest(league=lg)
                            if os.path.exists(perf_file):
                                st.success("Done!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")


# ==========================================
# BOTTOM: Spread Slates (full width, tabs)
# ==========================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title" id="full-slates">Full Slates</div>', unsafe_allow_html=True)


def _render_shadow_grader_panel():
    """Show cumulative MLB shadow vs production summary if the ledger exists."""
    from mlb.shadow_grader import (
        DEFAULT_LEDGER,
        aggregate_report,
        format_report,
    )

    if not os.path.exists(DEFAULT_LEDGER):
        return

    try:
        ledger = pd.read_csv(DEFAULT_LEDGER)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError):
        return

    report = aggregate_report(ledger)
    if report.get("n_total", 0) == 0:
        return

    with st.expander(
        f"Shadow vs Production -- {report.get('n_graded', 0)} graded "
        f"of {report['n_total']} ledger rows",
        expanded=False,
    ):
        st.code(format_report(report), language="text")


def _render_mlb_slate(game_df, lg):
    """Render the MLB full slate with moneyline picks and Kalshi links."""
    _render_shadow_grader_panel()

    show_filter = st.selectbox(
        "Show",
        ["All Games", "Value Bets Only"],
        label_visibility="collapsed",
        key=f"slate_filter_{lg}",
    )

    std_col = game_df["Std_Rating"] if "Std_Rating" in game_df.columns else pd.Series("PASS", index=game_df.index)
    kalshi_col = game_df["Rating"] if "Rating" in game_df.columns else pd.Series("PASS", index=game_df.index)
    is_value = std_col.isin(VALUE_RATINGS) | kalshi_col.isin(VALUE_RATINGS)

    if show_filter == "Value Bets Only":
        display_df = game_df[is_value].copy()
    else:
        display_df = game_df.copy()

    if display_df.empty:
        st.caption("No matching picks.")
        return

    # Production-model display columns
    if "Conf" in display_df.columns:
        display_df["Prod Conf"] = display_df["Conf"].apply(lambda x: f"{x:.1%}")

    # Shadow-model display columns (delineated with the "Shadow" prefix)
    if "MarketV2_Conf" in display_df.columns:
        display_df["Shadow Conf"] = display_df["MarketV2_Conf"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else ""
        )
    if "MarketV2_Market_NoVig_Home" in display_df.columns:
        display_df["Mkt NoVig Home"] = display_df["MarketV2_Market_NoVig_Home"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else ""
        )
    if "MarketV2_Edge_vs_Market" in display_df.columns:
        display_df["Shadow Edge"] = display_df["MarketV2_Edge_vs_Market"].apply(
            lambda x: f"{x:+.1%}" if pd.notna(x) else ""
        )
    if "MarketV2_Agrees_With_Production" in display_df.columns:
        display_df["Same?"] = display_df["MarketV2_Agrees_With_Production"].apply(
            lambda x: "" if x == "" or pd.isna(x) else ("yes" if bool(x) else "no")
        )
    if "MarketV2_Status" in display_df.columns:
        display_df["Shadow Status"] = display_df["MarketV2_Status"]

    # Rename for clarity in the table
    rename_map = {
        "Pick": "Prod Pick",
        "Std_Odds": "Prod Odds",
        "MarketV2_Pick": "Shadow Pick",
    }
    if "Std_Edge_Pct" in display_df.columns:
        rename_map["Std_Edge_Pct"] = "DK Edge"
    if "Std_Rating" in display_df.columns:
        rename_map["Std_Rating"] = "DK Rating"
    if "Std_Units" in display_df.columns:
        rename_map["Std_Units"] = "DK Units"
    if "Edge_Pct" in display_df.columns:
        rename_map["Edge_Pct"] = "Kalshi Edge"
    if "Rating" in display_df.columns:
        rename_map["Rating"] = "Kalshi Rating"
    if "Units" in display_df.columns:
        rename_map["Units"] = "Kalshi Units"
    display_df = display_df.rename(columns=rename_map)

    # Add position column from Kalshi portfolio
    if "Kalshi_Ticker" in display_df.columns and _position_lookup:
        display_df["Position"] = display_df["Kalshi_Ticker"].apply(
            lambda t: _format_position_col(t, _position_lookup) if pd.notna(t) and t else ""
        )
    else:
        display_df["Position"] = ""

    # Layout: production block, then shadow block, then market book / Kalshi
    show_cols = ["Date/Time", "Matchup", "Home_SP", "Away_SP",
                 "Prod Pick", "Prod Conf", "Prod Odds",
                 "Shadow Pick", "Shadow Conf", "Mkt NoVig Home",
                 "Shadow Edge", "Same?", "Shadow Status",
                 "DK Edge", "DK Rating", "DK Units",
                 "Kalshi Edge", "Kalshi Rating", "Kalshi Units",
                 "Kalshi_Side", "Kalshi_Price", "Position"]
    show_cols = [c for c in show_cols if c in display_df.columns]

    # Sort by best edge across both sources
    std_edge = _parse_edge(display_df["DK Edge"]) if "DK Edge" in display_df.columns else pd.Series(0.0, index=display_df.index)
    k_edge = _parse_edge(display_df["Kalshi Edge"]) if "Kalshi Edge" in display_df.columns else pd.Series(0.0, index=display_df.index)
    display_df["_edge_sort"] = pd.concat([std_edge, k_edge], axis=1).max(axis=1)
    display_df = display_df.sort_values("_edge_sort", ascending=False).drop(columns=["_edge_sort"])

    st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)


slate_tabs = st.tabs([league_data[lg]["settings"]["label"] for lg in LEAGUES])

for i, lg in enumerate(LEAGUES):
    with slate_tabs[i]:
        spread_df = league_data[lg]["spread_df"]
        game_df = league_data[lg]["game_df"]

        # For MLB, all predictions are game type (moneyline)
        if spread_df.empty and not game_df.empty:
            _render_mlb_slate(game_df, lg)
            continue

        if spread_df.empty:
            st.caption("No picks on this slate.")
            continue

        show_filter = st.selectbox(
            "Show",
            ["All Games", "Value Bets Only"],
            label_visibility="collapsed",
            key=f"slate_filter_{lg}",
        )

        if 'Std_Rating' in spread_df.columns:
            kalshi_rating = spread_df['Rating'] if 'Rating' in spread_df.columns else pd.Series('PASS', index=spread_df.index)
            is_value = (
                (spread_df['Std_Rating'].isin(VALUE_RATINGS)) |
                (kalshi_rating.isin(VALUE_RATINGS))
            )
            if show_filter == "Value Bets Only":
                display_df = spread_df[is_value].copy()
            else:
                display_df = spread_df.copy()
        else:
            display_df = spread_df.copy()

        if display_df.empty:
            st.caption("No matching picks.")
        else:
            if 'Conf' in display_df.columns:
                display_df['Confidence'] = display_df['Conf'].apply(lambda x: f"{x:.1%}")

            std_edge = _parse_edge(display_df['Std_Edge_Pct']) if 'Std_Edge_Pct' in display_df.columns else pd.Series(0.0, index=display_df.index)
            k_edge = _parse_edge(display_df['Edge_Pct']) if 'Edge_Pct' in display_df.columns else pd.Series(0.0, index=display_df.index)
            std_units = display_df['Std_Units'].fillna(0) if 'Std_Units' in display_df.columns else pd.Series(0.0, index=display_df.index)
            k_units = display_df['Units'].fillna(0) if 'Units' in display_df.columns else pd.Series(0.0, index=display_df.index)

            kalshi_better = k_edge > std_edge
            display_df['Best_Edge'] = k_edge.where(kalshi_better, std_edge).apply(lambda x: f"+{x:.1f}%" if x > 0 else "")
            display_df['Best_Units'] = k_units.where(kalshi_better, std_units)

            # Format odds column for display
            if 'Std_Odds' in display_df.columns:
                display_df['Odds'] = display_df['Std_Odds'].apply(_format_odds_display)
            else:
                display_df['Odds'] = ''

            table_cols = ['Date/Time', 'Pick', 'Confidence', 'Odds', 'Best_Edge', 'Best_Units']
            valid_cols = [c for c in table_cols if c in display_df.columns]
            table_df = display_df[valid_cols].rename(columns={
                'Date/Time': 'Time', 'Best_Edge': 'Edge', 'Best_Units': 'Units'
            })

            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                height=400,
                column_config={
                    "Time": st.column_config.TextColumn("Time", width="small"),
                    "Pick": st.column_config.TextColumn("Pick", width="large"),
                    "Confidence": st.column_config.TextColumn("Conf", width="small"),
                    "Odds": st.column_config.TextColumn("Odds", width="small"),
                    "Edge": st.column_config.TextColumn("Edge", width="small"),
                    "Units": st.column_config.NumberColumn("Units", format="%.1f", width="small"),
                }
            )

        # Kalshi game table
        if not game_df.empty:
            st.markdown('<div class="section-title">Kalshi Games</div>', unsafe_allow_html=True)
            table_game = game_df.copy()
            table_game["Confidence"] = table_game["Conf"].apply(lambda x: f"{x:.1%}") if "Conf" in table_game.columns else ""
            if "Kalshi_Ticker" in table_game.columns:
                table_game["Link"] = table_game["Kalshi_Ticker"].apply(
                    lambda t: kalshi_event_url(t) if pd.notna(t) and t else None
                )
                if _position_lookup:
                    table_game["Position"] = table_game["Kalshi_Ticker"].apply(
                        lambda t: _format_position_col(t, _position_lookup) if pd.notna(t) and t else ""
                    )
                else:
                    table_game["Position"] = ""
            game_cols = ["Date/Time", "Pick", "Confidence", "Kalshi_Price", "Kalshi_Fee", "Edge_Pct", "Units", "Rating", "Position", "Link"]
            game_cols = [c for c in game_cols if c in table_game.columns]
            table_game = table_game[game_cols].rename(columns={
                "Date/Time": "Time", "Kalshi_Price": "Price", "Kalshi_Fee": "Fee", "Edge_Pct": "Edge",
            })
            st.dataframe(
                table_game,
                use_container_width=True,
                hide_index=True,
                height=260,
                column_config={
                    "Time": st.column_config.TextColumn("Time", width="small"),
                    "Pick": st.column_config.TextColumn("Pick", width="large"),
                    "Confidence": st.column_config.TextColumn("Conf", width="small"),
                    "Price": st.column_config.NumberColumn("Price", width="small"),
                    "Fee": st.column_config.NumberColumn("Fee", width="small"),
                    "Edge": st.column_config.TextColumn("Edge", width="small"),
                    "Units": st.column_config.NumberColumn("Units", format="%.1f", width="small"),
                    "Rating": st.column_config.TextColumn("Rating", width="small"),
                    "Position": st.column_config.TextColumn("Position", width="small"),
                    "Link": st.column_config.LinkColumn("Kalshi", width="small", display_text="Trade"),
                }
            )


st.caption(f"Men's: {os.path.basename(league_data['mens']['paths']['model_file'])} | Women's: {os.path.basename(league_data['womens']['paths']['model_file'])}")
