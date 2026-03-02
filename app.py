import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
import predict
import backtest
import settle_bets
import io
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import pytz
from betting import format_line_shopping_text, VALUE_RATINGS, RATING_RANK
from league_config import get_league_artifact_paths, get_league_settings, normalize_league

# --- PATH CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BET_HIST_FILE = os.path.join(BASE_DIR, "betting_history.csv")

st.set_page_config(page_title="CBB Quant Edge", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --green-900: #0a1f12;
    --green-800: #0f2b1c;
    --green-700: #1a4d2e;
    --green-600: #2a6b42;
    --green-500: #3d8a5a;
    --green-100: #e6f0ea;
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
    background: var(--green-900);
    width: 240px !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    font-family: var(--font-body);
    color: rgba(255,255,255,0.8) !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: var(--font-body) !important;
}

section[data-testid="stSidebar"] .stSelectbox label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.4) !important;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: var(--green-700);
    color: white;
    border: 1px solid rgba(255,255,255,0.1);
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.82rem;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    transition: all 0.15s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--green-600);
    border-color: rgba(255,255,255,0.2);
}

.sidebar-brand {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 600;
    font-style: italic;
    color: #ffffff;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.sidebar-sub {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.5rem;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
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

/* Section headers */
.section-title {
    font-family: var(--font-body);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--green-800);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 0.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--green-700);
    display: inline-block;
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
.good .stat-value.positive { color: var(--gold-700); }
.marginal .stat-value.positive { color: #7a5c2e; }
.pass .stat-value.positive { color: var(--neutral-500); }

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

/* Missing spreads */
.missing-spreads-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--gold-100);
    border: 1px solid #ebe0c0;
    border-radius: 10px;
    padding: 8px 14px;
    margin: 0.5rem 0;
}

.missing-spreads-banner .ms-count {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 600;
    background: var(--gold-600);
    color: #ffffff;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.03em;
}

.missing-spreads-banner .ms-text {
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--gold-700);
}

.spread-game-row {
    background: var(--surface);
    border: 1px solid var(--neutral-200);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
}

.spread-game-teams {
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--green-900);
}

.spread-game-time {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--neutral-400);
    margin-top: 2px;
}

/* Record panel */
.record-stat {
    background: var(--surface);
    border: 1px solid var(--neutral-200);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}

.record-stat .rs-value {
    font-family: var(--font-mono);
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--green-800);
}

.record-stat .rs-label {
    font-family: var(--font-mono);
    font-size: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--neutral-400);
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# --- LEAGUE STATE ---
if 'active_league' not in st.session_state:
    st.session_state.active_league = "mens"

active_league = normalize_league(st.session_state.active_league)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">CBB Quant Edge</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Spread + Kalshi Markets</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    selected_league = st.selectbox(
        "League",
        ["mens", "womens"],
        format_func=lambda x: "Men's CBB" if x == "mens" else "Women's CBB",
        index=0 if active_league == "mens" else 1,
    )
    if selected_league != active_league:
        st.session_state.active_league = selected_league
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    if st.button("Refresh Predictions"):
        with st.spinner("Running model..."):
            predict.main(
                spread_overrides=st.session_state.get(f"spread_overrides_{active_league}", {}),
                league=active_league,
            )
            st.session_state[f"predictions_loaded_{active_league}"] = True
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    if st.button("Settle Bets"):
        with st.spinner("Settling..."):
            summary = settle_bets.settle_pending_bets(league=active_league)
        st.success(f"{summary['settled']} settled, {summary['still_pending']} pending")
        if summary["details"]:
            for d in summary["details"]:
                st.caption(d)
        st.rerun()

# --- LEAGUE PATHS ---
league_settings = get_league_settings(active_league)
league_paths = get_league_artifact_paths(BASE_DIR, active_league)
PRED_FILE = league_paths["predictions_file"]
PERF_FILE = league_paths["performance_file"]
DATA_FILE = league_paths["data_file"]
MODEL_FILE = league_paths["model_file"]
spread_overrides_key = f"spread_overrides_{active_league}"
predictions_loaded_key = f"predictions_loaded_{active_league}"


# --- AUTO-REFRESH PREDICTIONS ON STARTUP ---
if not st.session_state.get(predictions_loaded_key, False):
    with st.spinner("Loading predictions..."):
        predict.main(
            spread_overrides=st.session_state.get(spread_overrides_key, {}),
            league=active_league,
        )
    st.session_state[predictions_loaded_key] = True

predictions_with_line_shopping = predict.get_latest_predictions(active_league)
games_needing_spreads = predict.get_games_needing_spreads(active_league)


# --- KALSHI URL HELPER ---
_KALSHI_SERIES_SLUGS = {
    "KXNCAAMBSPREAD": "mens-college-basketball-spread",
    "KXNCAAMBGAME": "mens-college-basketball-mens-game",
    "KXNCAAWBSPREAD": "womens-college-basketball-spread",
    "KXNCAAWBGAME": "college-basketball-womens-game",
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


def _parse_edge(series):
    """Parse edge percentage strings like '+5.2%' into floats."""
    cleaned = series.fillna('').astype(str).str.rstrip('%').str.lstrip('+')
    return pd.to_numeric(cleaned, errors='coerce').fillna(0)


# ==========================================
# MAIN CONTENT
# ==========================================
if os.path.exists(PRED_FILE):
    df = pd.read_csv(PRED_FILE)
    if "Bet_Type" in df.columns:
        spread_df = df[df["Bet_Type"] != "game"].copy()
        game_df = df[df["Bet_Type"] == "game"].copy()
    else:
        spread_df = df.copy()
        game_df = pd.DataFrame(columns=df.columns)

    # --- KPI SUMMARY ROW ---
    if 'Std_Rating' in df.columns:
        rating_col = df['Rating'] if 'Rating' in df.columns else pd.Series('PASS', index=df.index)
        value_bets = df[
            (df['Std_Rating'].isin(VALUE_RATINGS)) |
            (rating_col.isin(VALUE_RATINGS))
        ].copy()

        num_value = len(value_bets)
        if num_value > 0:
            total_units = np.maximum(
                value_bets['Std_Units'].fillna(0),
                value_bets['Units'].fillna(0),
            ).sum()
        else:
            total_units = 0

        # Compute record stats for KPI bar
        record_str = "--"
        profit_str = "--"
        roi_str = "--"
        if os.path.exists(BET_HIST_FILE):
            _bh = pd.read_csv(BET_HIST_FILE)
            _settled = _bh[_bh["result"].isin(["win", "loss", "void"])]
            if len(_settled) > 0:
                _wins = len(_settled[_settled["result"] == "win"])
                _losses = len(_settled[_settled["result"] == "loss"])
                _profit = _settled["profit"].astype(float).sum()
                _wagered = _settled["wager"].astype(float).sum()
                _roi = (_profit / _wagered * 100) if _wagered > 0 else 0
                record_str = f"{_wins}W-{_losses}L"
                profit_str = f"{_profit:+.1f}U"
                roi_str = f"{_roi:+.1f}%"

        st.markdown(f'''
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-value">{num_value}</div>
                <div class="kpi-label">Value Bets</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{total_units:.1f}U</div>
                <div class="kpi-label">To Deploy</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{len(game_df)}</div>
                <div class="kpi-label">Kalshi Games</div>
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

    # --- MISSING SPREADS (compact banner) ---
    if games_needing_spreads:
        n_missing = len(games_needing_spreads)
        st.markdown(f'''
        <div class="missing-spreads-banner">
            <span class="ms-count">{n_missing}</span>
            <span class="ms-text">game{"s" if n_missing != 1 else ""} missing ESPN spread</span>
        </div>
        ''', unsafe_allow_html=True)

        with st.expander("Enter missing spreads", expanded=False):
            with st.form("spread_overrides_form"):
                override_inputs = {}
                form_cols = st.columns(3)
                for i, game in enumerate(games_needing_spreads):
                    matchup = game['matchup']
                    with form_cols[i % 3]:
                        st.markdown(f'''
                        <div class="spread-game-row">
                            <div class="spread-game-teams">{matchup}</div>
                            <div class="spread-game-time">{game.get("time_str", "")}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        override_inputs[matchup] = st.text_input(
                            matchup,
                            placeholder="-7.5 or +3",
                            key=f"spread_{active_league}_{game['id']}",
                            label_visibility="collapsed",
                        )
                submitted = st.form_submit_button("Apply & Re-run")
                if submitted:
                    overrides = st.session_state.get(spread_overrides_key, {})
                    for matchup, val in override_inputs.items():
                        val = val.strip()
                        if val:
                            try:
                                overrides[matchup] = float(val)
                            except ValueError:
                                st.error(f"Invalid spread for {matchup}: {val}")
                    st.session_state[spread_overrides_key] = overrides
                    st.session_state[predictions_loaded_key] = False
                    st.rerun()

    # ==========================================
    # TWO-COLUMN LAYOUT: Value Bets | Slate Table
    # ==========================================
    col_bets, col_slate = st.columns([3, 4])

    # --- LEFT: VALUE BETS ---
    with col_bets:
        # -- Spread Value Bets --
        spread_value_bets = pd.DataFrame(columns=spread_df.columns)
        if not spread_df.empty and 'Std_Rating' in spread_df.columns:
            spread_rating = spread_df['Rating'] if 'Rating' in spread_df.columns else pd.Series('PASS', index=spread_df.index)
            spread_value_bets = spread_df[
                (spread_df['Std_Rating'].isin(VALUE_RATINGS)) |
                (spread_rating.isin(VALUE_RATINGS))
            ].copy()

        if len(spread_value_bets) > 0:
            st.markdown('<div class="section-title">Spread Value Bets</div>', unsafe_allow_html=True)

            std_rank = spread_value_bets['Std_Rating'].fillna('PASS').map(RATING_RANK).fillna(0)
            kalshi_rank = spread_value_bets['Rating'].fillna('PASS').map(RATING_RANK).fillna(0)
            spread_value_bets['_best_rank'] = np.maximum(std_rank, kalshi_rank)
            spread_value_bets['_best_edge'] = np.maximum(
                _parse_edge(spread_value_bets['Std_Edge_Pct']),
                _parse_edge(spread_value_bets['Edge_Pct']),
            )
            spread_value_bets = (
                spread_value_bets
                .sort_values(['_best_rank', '_best_edge'], ascending=[False, False])
                .drop(columns=['_best_rank', '_best_edge'])
            )

            for idx, row in spread_value_bets.iterrows():
                std_rating = row.get('Std_Rating', 'PASS')
                kalshi_rating = row.get('Rating', 'PASS') if pd.notna(row.get('Rating')) else 'PASS'
                conf = row['Conf']
                game_time = row.get('Date/Time', '')

                std_rank_val = RATING_RANK.get(std_rating, 0)
                kalshi_rank_val = RATING_RANK.get(kalshi_rating, 0)
                kalshi_is_primary = kalshi_rank_val > std_rank_val
                best_rating = kalshi_rating if kalshi_is_primary else std_rating
                badge_css = best_rating.lower()

                if kalshi_is_primary:
                    display_edge = row.get('Edge_Pct', 'N/A')
                    display_units = row.get('Units', 0) or 0
                    edge_source = "Kalshi Edge"
                else:
                    display_edge = row.get('Std_Edge_Pct', 'N/A')
                    display_units = row.get('Std_Units', 0) or 0
                    edge_source = "Edge"

                breakeven = row.get('Breakeven_Spread', None)
                breakeven_str = f"{breakeven:+.1f}" if breakeven and pd.notna(breakeven) else "---"

                kalshi_side = row.get('Kalshi_Side')
                kalshi_price = row.get('Kalshi_Price')
                has_kalshi = pd.notna(kalshi_side) and kalshi_side

                kalshi_html = ""
                if has_kalshi:
                    kalshi_edge = row.get('Edge_Pct', 'N/A')
                    kalshi_ticker = row.get('Kalshi_Ticker', '')
                    kalshi_text = f"Kalshi {kalshi_side} @ {kalshi_price}c . {kalshi_edge}"
                    if kalshi_ticker:
                        kalshi_html = f'<div class="kalshi-row"><a href="{kalshi_event_url(kalshi_ticker)}" target="_blank" class="kalshi-link">{kalshi_text}</a></div>'
                    else:
                        kalshi_html = f'<div class="kalshi-row"><span class="kalshi-label">{kalshi_text}</span></div>'

                st.markdown(f'''
                <div class="bet-card {badge_css}">
                    <div class="bet-header">
                        <div class="bet-badge {badge_css}">{best_rating.title()}</div>
                        <div class="bet-time">{game_time}</div>
                    </div>
                    <div class="bet-pick">{row['Pick']}</div>
                    <div class="bet-matchup">{row['Matchup']}</div>
                    <div class="bet-stats">
                        <div class="stat-item">
                            <span class="stat-label">Model</span>
                            <span class="stat-value">{conf:.1%}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">{edge_source}</span>
                            <span class="stat-value positive">{display_edge}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Units</span>
                            <span class="stat-value">{display_units:.1f}U</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Breakeven</span>
                            <span class="stat-value">{breakeven_str}</span>
                        </div>
                    </div>
                    {kalshi_html}
                </div>
                ''', unsafe_allow_html=True)

                line_shopping_data = None
                if predictions_with_line_shopping is not None:
                    match = predictions_with_line_shopping[
                        predictions_with_line_shopping['Matchup'] == row['Matchup']
                    ]
                    if len(match) > 0 and 'Line_Shopping_Data' in match.columns:
                        line_shopping_data = match.iloc[0].get('Line_Shopping_Data')

                if line_shopping_data is not None:
                    with st.expander("Line Shopping", expanded=False):
                        st.code(format_line_shopping_text(line_shopping_data), language=None)
        else:
            st.markdown('<div class="section-title">Spread Value Bets</div>', unsafe_allow_html=True)
            st.caption("No spread value bets on this slate.")

        # -- Kalshi Game Value Bets --
        st.markdown('<div class="section-title">Kalshi Game Markets</div>', unsafe_allow_html=True)

        if game_df.empty:
            st.caption("No Kalshi game markets on this slate.")
        else:
            game_value = game_df[game_df['Rating'].isin(VALUE_RATINGS)].copy() if 'Rating' in game_df.columns else pd.DataFrame()
            if not game_value.empty:
                for _, row in game_value.sort_values(by='Conf', ascending=False).iterrows():
                    rating = row.get("Rating", "PASS")
                    badge_css = str(rating).lower()
                    game_kalshi_ticker = row.get('Kalshi_Ticker', '')
                    game_kalshi_text = f"Kalshi {row.get('Kalshi_Side', '')} @ {row.get('Kalshi_Price', '')}c"
                    game_kalshi_url = kalshi_event_url(game_kalshi_ticker)
                    if game_kalshi_url:
                        game_kalshi_html = f'<a href="{game_kalshi_url}" target="_blank" class="kalshi-link">{game_kalshi_text}</a>'
                    else:
                        game_kalshi_html = f'<span class="kalshi-label">{game_kalshi_text}</span>'
                    st.markdown(f'''
                    <div class="bet-card {badge_css}">
                        <div class="bet-header">
                            <div class="bet-badge {badge_css}">{str(rating).title()}</div>
                            <div class="bet-time">{row.get('Date/Time', '')}</div>
                        </div>
                        <div class="bet-pick">{row.get('Pick', '')}</div>
                        <div class="bet-matchup">{row.get('Matchup', '')}</div>
                        <div class="bet-stats">
                            <div class="stat-item">
                                <span class="stat-label">Model</span>
                                <span class="stat-value">{row.get('Conf', 0):.1%}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Kalshi Edge</span>
                                <span class="stat-value positive">{row.get('Edge_Pct', '')}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Units</span>
                                <span class="stat-value">{float(row.get('Units', 0) or 0):.1f}U</span>
                            </div>
                        </div>
                        <div class="kalshi-row">{game_kalshi_html}</div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.caption("No Kalshi value plays today.")

    # --- RIGHT: FULL SLATE TABLES ---
    with col_slate:
        # -- Spread Slate --
        st.markdown('<div class="section-title">Spread Slate</div>', unsafe_allow_html=True)

        show_filter = st.selectbox(
            "Show",
            ["All Games", "Value Bets Only"],
            label_visibility="collapsed"
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
            st.caption("No spread picks on this slate.")
        else:
            if 'Conf' in display_df.columns:
                display_df['Confidence'] = display_df['Conf'].apply(lambda x: f"{x:.1%}")

            std_edge = _parse_edge(display_df['Std_Edge_Pct']) if 'Std_Edge_Pct' in display_df.columns else pd.Series(0.0, index=display_df.index)
            kalshi_edge = _parse_edge(display_df['Edge_Pct']) if 'Edge_Pct' in display_df.columns else pd.Series(0.0, index=display_df.index)
            std_units = display_df['Std_Units'].fillna(0) if 'Std_Units' in display_df.columns else pd.Series(0.0, index=display_df.index)
            kalshi_units = display_df['Units'].fillna(0) if 'Units' in display_df.columns else pd.Series(0.0, index=display_df.index)

            kalshi_is_better = kalshi_edge > std_edge
            best_edge = kalshi_edge.where(kalshi_is_better, std_edge)
            best_units = kalshi_units.where(kalshi_is_better, std_units)

            display_df['Best_Edge'] = best_edge.apply(lambda x: f"+{x:.1f}%" if x > 0 else "")
            display_df['Best_Units'] = best_units

            table_cols = ['Date/Time', 'Pick', 'Confidence', 'Best_Edge', 'Best_Units']
            valid_cols = [c for c in table_cols if c in display_df.columns]

            table_df = display_df[valid_cols].copy()
            table_df = table_df.rename(columns={
                'Date/Time': 'Time',
                'Best_Edge': 'Edge',
                'Best_Units': 'Units'
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
                    "Edge": st.column_config.TextColumn("Edge", width="small"),
                    "Units": st.column_config.NumberColumn("Units", format="%.1f", width="small"),
                }
            )

        # -- Kalshi Game Table --
        if not game_df.empty:
            st.markdown('<div class="section-title">Kalshi Games Table</div>', unsafe_allow_html=True)

            table_game = game_df.copy()
            table_game["Confidence"] = table_game["Conf"].apply(lambda x: f"{x:.1%}") if "Conf" in table_game.columns else ""
            if "Kalshi_Ticker" in table_game.columns:
                table_game["Link"] = table_game["Kalshi_Ticker"].apply(
                    lambda t: kalshi_event_url(t) if pd.notna(t) and t else None
                )
            game_cols = ["Date/Time", "Pick", "Confidence", "Kalshi_Price", "Edge_Pct", "Units", "Rating", "Link"]
            game_cols = [c for c in game_cols if c in table_game.columns]
            table_game = table_game[game_cols].rename(columns={
                "Date/Time": "Time",
                "Kalshi_Price": "Price",
                "Edge_Pct": "Edge",
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
                    "Edge": st.column_config.TextColumn("Edge", width="small"),
                    "Units": st.column_config.NumberColumn("Units", format="%.1f", width="small"),
                    "Rating": st.column_config.TextColumn("Rating", width="small"),
                    "Link": st.column_config.LinkColumn("Kalshi", width="small", display_text="Trade"),
                }
            )

    # ==========================================
    # BOTTOM ROW: Record + Performance side by side
    # ==========================================
    st.markdown("<hr>", unsafe_allow_html=True)

    col_record, col_perf = st.columns(2)

    # --- RECORD ---
    with col_record:
        st.markdown('<div class="section-title">Betting Record</div>', unsafe_allow_html=True)

        if os.path.exists(BET_HIST_FILE):
            bet_hist = pd.read_csv(BET_HIST_FILE)
            pending_bets = bet_hist[bet_hist["result"] == "pending"]
            pending_count = len(pending_bets)

            if pending_count > 0:
                st.caption(f"{pending_count} pending bet{'s' if pending_count != 1 else ''}")
                pending_display = pending_bets[["date", "platform", "line", "odds", "wager"]].copy()
                pending_display["wager"] = pending_display["wager"].apply(lambda x: f"${x:.2f}")
                st.dataframe(pending_display, use_container_width=True, hide_index=True, height=180)

            settled = bet_hist[bet_hist["result"].isin(["win", "loss", "void"])]
            if len(settled) > 0:
                wins = len(settled[settled["result"] == "win"])
                losses = len(settled[settled["result"] == "loss"])
                total_profit = settled["profit"].astype(float).sum()
                total_wagered = settled["wager"].astype(float).sum()
                roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Record", f"{wins}W-{losses}L")
                rc2.metric("Profit", f"{total_profit:+.2f}U")
                rc3.metric("ROI", f"{roi:+.1f}%")

                recent = settled.tail(10).iloc[::-1].copy()
                recent["Result"] = recent["result"].apply(
                    lambda x: {"win": "W", "loss": "L", "void": "P"}.get(x, x)
                )
                recent["P/L"] = recent["profit"].apply(lambda x: f"{float(x):+.2f}")
                display_cols = ["date", "platform", "line", "Result", "P/L"]
                st.dataframe(recent[display_cols], use_container_width=True, hide_index=True, height=280)
            elif pending_count == 0:
                st.caption("No betting history yet.")
        else:
            st.caption("No betting history file found.")

    # --- PERFORMANCE ---
    with col_perf:
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

        if os.path.exists(PERF_FILE):
            hist = pd.read_csv(PERF_FILE)
            hist['date'] = pd.to_datetime(hist['date'])

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

            chart = alt.Chart(hist).mark_area(
                line={'color': '#1a4d2e'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[
                        alt.GradientStop(color='rgba(26, 77, 46, 0.1)', offset=0),
                        alt.GradientStop(color='rgba(26, 77, 46, 0.3)', offset=1)
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
            if st.button("Run Backtest"):
                with st.spinner("Training models..."):
                    f = io.StringIO()
                    try:
                        with redirect_stdout(f):
                            backtest.run_backtest(league=active_league)
                        if os.path.exists(PERF_FILE):
                            st.success("Done!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.caption(f"{league_settings['label']} | {os.path.basename(MODEL_FILE)} | {os.path.basename(DATA_FILE)}")

else:
    st.warning("No predictions found.")
    if st.button("Run Prediction Engine"):
        with st.spinner("Calculating..."):
            predict.main(
                spread_overrides=st.session_state.get(spread_overrides_key, {}),
                league=active_league,
            )
            st.session_state[predictions_loaded_key] = True
        st.rerun()
