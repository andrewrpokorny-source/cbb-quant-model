import streamlit as st
import pandas as pd
import os
import altair as alt
import predict
import backtest
import settle_bets
import io
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import pytz
from betting import format_line_shopping_text

# --- PATH CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_FILE = os.path.join(BASE_DIR, "daily_predictions.csv")
PERF_FILE = os.path.join(BASE_DIR, "performance_log.csv")
BET_HIST_FILE = os.path.join(BASE_DIR, "betting_history.csv")
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")

st.set_page_config(page_title="CBB Quant Edge", layout="centered")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base styles */
.stApp {
    background-color: #faf9f6;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Typography overrides */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'DM Sans', sans-serif !important;
    color: #1a2e1a !important;
    letter-spacing: -0.02em;
}

p, span, div, .stMarkdown {
    font-family: 'DM Sans', sans-serif;
}

/* Header styling */
.main-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a2e1a;
    margin-bottom: 0;
    letter-spacing: -0.03em;
}

.sub-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #6b7c6b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* Section headers */
.section-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a2e1a;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #1a4d2e;
    display: inline-block;
}

/* Value bet cards */
.bet-card {
    background: #ffffff;
    border: 1px solid #e5e5e0;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}

.bet-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.bet-card.strong {
    border-left: 4px solid #1a4d2e;
}

.bet-card.good {
    border-left: 4px solid #c9a227;
}

.bet-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 8px;
    border-radius: 3px;
    display: inline-block;
    margin-bottom: 8px;
}

.bet-badge.strong {
    background: #1a4d2e;
    color: #ffffff;
}

.bet-badge.good {
    background: #c9a227;
    color: #1a2e1a;
}

.bet-pick {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #1a2e1a;
    margin: 4px 0;
}

.bet-matchup {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #6b7c6b;
}

.bet-stats {
    display: flex;
    gap: 1.5rem;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #f0f0eb;
}

.stat-item {
    display: flex;
    flex-direction: column;
}

.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8a9a8a;
}

.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: #1a2e1a;
}

.stat-value.positive {
    color: #1a4d2e;
}

/* Kalshi badge */
.kalshi-row {
    background: #f5f7f5;
    border-radius: 6px;
    padding: 10px 12px;
    margin-top: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #1a2e1a;
}

.kalshi-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b7c6b;
    margin-right: 8px;
}

/* Summary bar */
.summary-bar {
    display: flex;
    justify-content: flex-start;
    gap: 2rem;
    background: #ffffff;
    border: 1px solid #e5e5e0;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1.5rem 0;
}

.summary-item {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.summary-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a4d2e;
}

.summary-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b7c6b;
    margin-top: 2px;
}

.summary-divider {
    width: 1px;
    background: #e5e5e0;
}

/* Bet header with time */
.bet-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.bet-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #8a9a8a;
}

/* Games table styling */
.games-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #6b7c6b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Refresh button */
.stButton > button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    background: #1a4d2e;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.25rem;
    transition: background 0.2s ease;
}

.stButton > button:hover {
    background: #2a5d3e;
    color: white;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
    color: #1a2e1a;
    background: #f8f8f5;
    border-radius: 6px;
}

/* Code blocks for line shopping */
.stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    background: #f8f8f5 !important;
    border: 1px solid #e5e5e0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #e5e5e0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #6b7c6b;
    padding: 0.75rem 1.5rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}

.stTabs [aria-selected="true"] {
    color: #1a4d2e;
    border-bottom: 2px solid #1a4d2e;
}

/* Metrics */
.stMetric {
    background: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e5e5e0;
}

.stMetric label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #e5e5e0;
    margin: 1.5rem 0;
}

/* Dataframe */
.stDataFrame {
    font-family: 'DM Sans', sans-serif;
}

.stDataFrame [data-testid="stDataFrameResizable"] {
    border: 1px solid #e5e5e0;
    border-radius: 8px;
    overflow: hidden;
}

/* Missing spreads */
.missing-spreads-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #fffbeb;
    border: 1px solid #f0e6c0;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 1rem 0;
}

.missing-spreads-banner .ms-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    background: #c9a227;
    color: #1a2e1a;
    padding: 2px 7px;
    border-radius: 3px;
    letter-spacing: 0.03em;
}

.missing-spreads-banner .ms-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #6b5e1a;
}

.spread-game-row {
    background: #ffffff;
    border: 1px solid #e5e5e0;
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 8px;
}

.spread-game-teams {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
    color: #1a2e1a;
}

.spread-game-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #8a9a8a;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# --- AUTO-REFRESH PREDICTIONS ON STARTUP ---
def should_refresh_predictions():
    """Check if predictions need refreshing (stale or missing)"""
    if not os.path.exists(PRED_FILE):
        return True
    try:
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern).date()
        file_mtime = datetime.fromtimestamp(os.path.getmtime(PRED_FILE))
        file_date = file_mtime.date()
        return file_date < today
    except (OSError, ValueError, TypeError):
        return True

# Run predictions once on startup (always run to get line shopping data)
if 'predictions_loaded' not in st.session_state:
    with st.spinner("Loading predictions..."):
        predict.main(spread_overrides=st.session_state.get('spread_overrides', {}))
    st.session_state.predictions_loaded = True

# Get predictions with line shopping data
predictions_with_line_shopping = predict.get_latest_predictions()

# Check for games needing manual spreads (used later)
games_needing_spreads = predict.get_games_needing_spreads()

# --- HEADER ---
st.markdown('<div class="main-header">CBB Quant Edge</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Gradient Boosting Model · Spread Analysis</div>', unsafe_allow_html=True)

# ==========================================
# MAIN CONTENT - No tabs, prioritized layout
# ==========================================
if os.path.exists(PRED_FILE):
    df = pd.read_csv(PRED_FILE)

    # --- SUMMARY BAR ---
    if 'Std_Rating' in df.columns:
        value_bets = df[
            (df['Std_Rating'].isin(['STRONG', 'GOOD'])) |
            (df['Rating'].isin(['STRONG', 'GOOD']))
        ].copy()

        num_value = len(value_bets)
        if num_value > 0:
            # Use best units per bet (max of Kalshi vs standard book)
            std_u = value_bets['Std_Units'].fillna(0)
            kalshi_u = value_bets['Units'].fillna(0)
            total_units = pd.concat([std_u, kalshi_u], axis=1).max(axis=1).sum()
        else:
            total_units = 0
        num_strong = len(value_bets[
            (value_bets['Std_Rating'] == 'STRONG') |
            (value_bets['Rating'] == 'STRONG')
        ]) if num_value > 0 else 0

        st.markdown(f'''
        <div class="summary-bar">
            <div class="summary-item">
                <span class="summary-value">{num_value}</span>
                <span class="summary-label">Value Bets</span>
            </div>
            <div class="summary-divider"></div>
            <div class="summary-item">
                <span class="summary-value">{total_units:.1f}U</span>
                <span class="summary-label">To Deploy</span>
            </div>
            <div class="summary-divider"></div>
            <div class="summary-item">
                <span class="summary-value">{len(df)}</span>
                <span class="summary-label">Games Today</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # --- VALUE BETS SECTION ---
        if num_value > 0:
            st.markdown('<div class="section-title">Value Bets</div>', unsafe_allow_html=True)

            # Use columns for side-by-side cards when 2+ bets
            if num_value >= 2:
                cols = st.columns(2)
            else:
                cols = [st.container()]

            RATING_RANK = {'STRONG': 3, 'GOOD': 2, 'MARGINAL': 1, 'PASS': 0}

            for i, (idx, row) in enumerate(value_bets.iterrows()):
                col = cols[i % 2] if num_value >= 2 else cols[0]

                with col:
                    std_rating = row.get('Std_Rating', 'MARGINAL')
                    kalshi_rating = row.get('Rating', 'PASS') if pd.notna(row.get('Rating')) else 'PASS'
                    conf = row['Conf']
                    game_time = row.get('Date/Time', '')

                    # Use whichever source gave the better rating
                    std_rank = RATING_RANK.get(std_rating, 0)
                    kalshi_rank = RATING_RANK.get(kalshi_rating, 0)
                    kalshi_is_primary = kalshi_rank > std_rank

                    best_rating = kalshi_rating if kalshi_is_primary else std_rating
                    is_strong = best_rating == 'STRONG'
                    card_class = "strong" if is_strong else "good"
                    badge_class = "strong" if is_strong else "good"
                    badge_text = "Strong" if is_strong else "Good"

                    # Pick edge/units from the source that triggered the rating
                    if kalshi_is_primary:
                        display_edge = row.get('Edge_Pct', 'N/A')
                        display_units = row.get('Units', 0) or 0
                        edge_source = "Kalshi Edge"
                    else:
                        display_edge = row.get('Std_Edge_Pct', 'N/A')
                        display_units = row.get('Std_Units', 0) or 0
                        edge_source = "Edge"

                    # Breakeven spread
                    breakeven = row.get('Breakeven_Spread', None)
                    breakeven_str = f"{breakeven:+.1f}" if breakeven and pd.notna(breakeven) else "—"

                    # Kalshi info
                    kalshi_side = row.get('Kalshi_Side')
                    kalshi_price = row.get('Kalshi_Price')
                    has_kalshi = pd.notna(kalshi_side) and kalshi_side

                    kalshi_html = ""
                    if has_kalshi:
                        kalshi_edge = row.get('Edge_Pct', 'N/A')
                        kalshi_html = f'<div class="kalshi-row"><span class="kalshi-label">Kalshi</span> {kalshi_side} @ {kalshi_price}¢ · {kalshi_edge}</div>'

                    st.markdown(f'''
                    <div class="bet-card {card_class}">
                        <div class="bet-header">
                            <div class="bet-badge {badge_class}">{badge_text}</div>
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

                    # Line Shopping - show inline, not hidden
                    line_shopping_data = None
                    if predictions_with_line_shopping is not None:
                        match = predictions_with_line_shopping[
                            predictions_with_line_shopping['Matchup'] == row['Matchup']
                        ]
                        if len(match) > 0 and 'Line_Shopping_Data' in match.columns:
                            line_shopping_data = match.iloc[0].get('Line_Shopping_Data')

                    if line_shopping_data is not None:
                        with st.expander("Line Shopping", expanded=True):
                            st.code(format_line_shopping_text(line_shopping_data), language=None)

            st.markdown("<hr>", unsafe_allow_html=True)

    # --- ALL PICKS SECTION ---
    st.markdown(f'<div class="section-title">Full Slate</div>', unsafe_allow_html=True)

    # Filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        show_filter = st.selectbox(
            "Show",
            ["All Games", "Value Bets Only", "Strong Only"],
            label_visibility="collapsed"
        )

    # Apply filter
    if 'Std_Rating' in df.columns:
        if show_filter == "Value Bets Only":
            display_df = df[df['Std_Rating'].isin(['STRONG', 'GOOD'])].copy()
        elif show_filter == "Strong Only":
            display_df = df[df['Std_Rating'] == 'STRONG'].copy()
        else:
            display_df = df.copy()
    else:
        display_df = df.copy()

    if 'Conf' in display_df.columns:
        display_df['Confidence'] = display_df['Conf'].apply(lambda x: f"{x:.1%}")

    table_cols = ['Date/Time', 'Pick', 'Confidence', 'Std_Edge_Pct', 'Std_Units']
    valid_cols = [c for c in table_cols if c in display_df.columns]

    # Rename for display
    table_df = display_df[valid_cols].copy()
    table_df = table_df.rename(columns={
        'Date/Time': 'Time',
        'Std_Edge_Pct': 'Edge',
        'Std_Units': 'Units'
    })

    st.dataframe(
        table_df,
        width='stretch',
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

    # --- MISSING SPREADS (compact, below predictions) ---
    if games_needing_spreads:
        n_missing = len(games_needing_spreads)
        st.markdown(f'''
        <div class="missing-spreads-banner">
            <span class="ms-count">{n_missing}</span>
            <span class="ms-text">game{"s" if n_missing != 1 else ""} missing ESPN spread — enter manually to get predictions</span>
        </div>
        ''', unsafe_allow_html=True)

        with st.expander("Enter missing spreads", expanded=False):
            with st.form("spread_overrides_form"):
                override_inputs = {}
                cols = st.columns(2)
                for i, game in enumerate(games_needing_spreads):
                    matchup = game['matchup']
                    with cols[i % 2]:
                        st.markdown(f'''
                        <div class="spread-game-row">
                            <div class="spread-game-teams">{matchup}</div>
                            <div class="spread-game-time">{game.get("time_str", "")}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        override_inputs[matchup] = st.text_input(
                            matchup,
                            placeholder="-7.5 or +3",
                            key=f"spread_{game['id']}",
                            label_visibility="collapsed",
                        )
                submitted = st.form_submit_button("Apply & Re-run")
                if submitted:
                    overrides = st.session_state.get('spread_overrides', {})
                    for matchup, val in override_inputs.items():
                        val = val.strip()
                        if val:
                            try:
                                overrides[matchup] = float(val)
                            except ValueError:
                                st.error(f"Invalid spread for {matchup}: {val}")
                    st.session_state.spread_overrides = overrides
                    st.session_state.predictions_loaded = False
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Refresh"):
            with st.spinner("Running model..."):
                predict.OUTPUT_FILE = PRED_FILE
                predict.main(spread_overrides=st.session_state.get('spread_overrides', {}))
            st.rerun()

else:
    st.warning("No predictions found.")
    if st.button("Run Prediction Engine"):
        with st.spinner("Calculating..."):
            predict.OUTPUT_FILE = PRED_FILE
            predict.main(spread_overrides=st.session_state.get('spread_overrides', {}))
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# --- BETTING LOG SECTION ---
if os.path.exists(BET_HIST_FILE):
    bet_hist = pd.read_csv(BET_HIST_FILE)
    pending_bets = bet_hist[bet_hist["result"] == "pending"]
    pending_count = len(pending_bets)

    with st.expander(f"Betting Log ({pending_count} pending)"):
        if pending_count > 0:
            st.markdown(f'<div class="section-title">Pending Bets ({pending_count})</div>', unsafe_allow_html=True)
            pending_display = pending_bets[["date", "platform", "line", "odds", "wager"]].copy()
            pending_display["wager"] = pending_display["wager"].apply(lambda x: f"${x:.2f}")
            st.dataframe(pending_display, width='stretch', hide_index=True)

            if st.button("Settle Bets"):
                with st.spinner("Settling pending bets via ESPN scores..."):
                    summary = settle_bets.settle_pending_bets()
                st.success(f"Settled {summary['settled']} bets. {summary['still_pending']} still pending.")
                if summary["details"]:
                    with st.expander("Settlement Details"):
                        for d in summary["details"]:
                            st.text(d)
                st.rerun()

        # Recent history
        settled = bet_hist[bet_hist["result"].isin(["win", "loss", "void"])]
        if len(settled) > 0:
            st.markdown('<div class="section-title">Recent Bets</div>', unsafe_allow_html=True)
            recent = settled.tail(15).iloc[::-1].copy()
            recent["Result"] = recent["result"].apply(
                lambda x: {"win": "W", "loss": "L", "void": "P"}.get(x, x)
            )
            recent["P/L"] = recent["profit"].apply(lambda x: f"{float(x):+.2f}")
            display_cols = ["date", "platform", "line", "odds", "wager", "Result", "P/L"]
            st.dataframe(recent[display_cols], width='stretch', hide_index=True)

            # Summary stats
            wins = len(settled[settled["result"] == "win"])
            losses = len(settled[settled["result"] == "loss"])
            total_profit = settled["profit"].astype(float).sum()
            total_wagered = settled["wager"].astype(float).sum()
            roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
            st.caption(
                f"Record: {wins}W-{losses}L | "
                f"Profit: {total_profit:+.2f}U | "
                f"ROI: {roi:+.1f}%"
            )

st.markdown("<hr>", unsafe_allow_html=True)

# --- PERFORMANCE SECTION (collapsible) ---
with st.expander("Performance History"):
    if os.path.exists(PERF_FILE):
        hist = pd.read_csv(PERF_FILE)
        hist['date'] = pd.to_datetime(hist['date'])

        def get_metrics(df_subset):
            if len(df_subset) == 0: return 0, 0.0, 0.0
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

        st.markdown('<div class="section-title">Performance</div>', unsafe_allow_html=True)
        st.caption(f"Stats as of {yesterday.strftime('%B %d, %Y')}")

        c1, c2, c3 = st.columns(3)

        cnt_y, rate_y, prof_y = get_metrics(df_yesterday)
        c1.metric("Yesterday", f"{prof_y:+.1f}U", f"{cnt_y} bets · {rate_y:.0%} win")

        cnt_7, rate_7, prof_7 = get_metrics(df_7)
        c2.metric("Last 7 Days", f"{prof_7:+.1f}U", f"{cnt_7} bets · {rate_7:.0%} win")

        cnt_30, rate_30, prof_30 = get_metrics(df_30)
        c3.metric("All Time", f"{prof_30:+.1f}U", f"{cnt_30} bets · {rate_30:.0%} win")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Profit Trend</div>', unsafe_allow_html=True)
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
        ).properties(height=280).configure_axis(
            grid=True,
            gridColor='#e5e5e0',
            domainColor='#e5e5e0'
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(chart, width='stretch')

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

            st.dataframe(df_display, width='stretch', hide_index=True)

    else:
        st.info("No performance data yet.")
        if st.button("Run Backtest"):
            with st.spinner("Training models..."):
                f = io.StringIO()
                try:
                    with redirect_stdout(f):
                        backtest.DATA_FILE = DATA_FILE
                        backtest.OUTPUT_FILE = PERF_FILE
                        backtest.run_backtest()
                    if os.path.exists(PERF_FILE):
                        st.success("Done!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

with st.expander("System"):
    st.caption(f"Data: {DATA_FILE}")
    if os.path.exists(DATA_FILE):
        st.caption(f"Size: {os.path.getsize(DATA_FILE) / 1024:.0f} KB")
