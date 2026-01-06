import streamlit as st
import pandas as pd
import os
import altair as alt
import predict
import backtest
import io
from contextlib import redirect_stdout
from datetime import datetime, timedelta

# --- PATH CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_FILE = os.path.join(BASE_DIR, "daily_predictions.csv")
PERF_FILE = os.path.join(BASE_DIR, "performance_log.csv")
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")

st.set_page_config(page_title="CBB Quant Edge", page_icon="🏀", layout="centered")

st.title("🏀 CBB Quant Edge")
st.caption("v2.9 | Strategy: Efficiency Differentials | Mode: Spread Specialist")
st.divider()

tab1, tab2 = st.tabs(["📅 Daily Picks", "📈 Performance Dashboard"])

# ==========================================
# TAB 1: DAILY PICKS
# ==========================================
with tab1:
    if os.path.exists(PRED_FILE):
        df = pd.read_csv(PRED_FILE)
        st.subheader(f"Today's Slate ({len(df)} Games)")
        
        if 'Conf' in df.columns:
            df['Confidence'] = df['Conf'].apply(lambda x: f"{x:.1%}")
        
        # Display Logic
        display_cols = ['Date/Time', 'Matchup', 'Pick', 'Confidence', 'Rest']
        if 'Raw Odds' in df.columns: display_cols.append('Raw Odds') # Debug col if exists
        
        valid_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(df[valid_cols].style.map(lambda x: "font-weight: bold", subset=['Pick']), use_container_width=True, hide_index=True)
        
        if st.button("Refresh Predictions"):
            with st.spinner("Running Engine..."):
                predict.OUTPUT_FILE = PRED_FILE
                predict.main()
            st.rerun()
    else:
        st.warning("⚠️ No predictions found.")
        if st.button("Run Prediction Engine"):
            with st.spinner("Calculating..."):
                predict.OUTPUT_FILE = PRED_FILE
                predict.main()
            st.rerun()

# ==========================================
# TAB 2: PERFORMANCE DASHBOARD
# ==========================================
with tab2:
    if os.path.exists(PERF_FILE):
        hist = pd.read_csv(PERF_FILE)
        hist['date'] = pd.to_datetime(hist['date'])
        
        # --- CALCULATE METRICS HELPER ---
        def get_metrics(df_subset):
            if len(df_subset) == 0: return 0, 0.0, 0.0
            
            # Recalculate units for this subset
            df_subset = df_subset.copy()
            df_subset['units'] = df_subset['pick_correct'].apply(lambda x: 1.0 if x else -1.1)
            
            cnt = len(df_subset)
            wins = df_subset['pick_correct'].sum()
            rate = wins / cnt
            profit = df_subset['units'].sum()
            return cnt, rate, profit

        # --- DATE FILTERS ---
        # Normalize to midnight for accurate comparisons
        today = pd.Timestamp.now().normalize()
        yesterday = today - timedelta(days=1)
        
        # 1. Yesterday
        df_yesterday = hist[hist['date'].dt.date == yesterday.date()]
        
        # 2. Last 7 Days
        df_7 = hist[hist['date'] >= (today - timedelta(days=7))]
        
        # 3. Last 30 Days (Full Log)
        df_30 = hist # Backtest is capped at 4 weeks usually
        
        # --- DISPLAY SCORECARDS ---
        st.subheader("📊 Performance Snapshots")
        
        # Row 1: The Three Timeframes
        c1, c2, c3 = st.columns(3)
        
        # Yesterday
        cnt_y, rate_y, prof_y = get_metrics(df_yesterday)
        c1.markdown("### Yesterday")
        c1.metric("Bets", cnt_y)
        c1.metric("Profit", f"{prof_y:+.2f} U", delta_color="normal")
        c1.metric("Win Rate", f"{rate_y:.1%}")
        
        # Last 7 Days
        cnt_7, rate_7, prof_7 = get_metrics(df_7)
        c2.markdown("### Last 7 Days")
        c2.metric("Bets", cnt_7)
        c2.metric("Profit", f"{prof_7:+.2f} U", delta_color="normal")
        c2.metric("Win Rate", f"{rate_7:.1%}")

        # Last 30 Days
        cnt_30, rate_30, prof_30 = get_metrics(df_30)
        c3.markdown("### Last 30 Days")
        c3.metric("Bets", cnt_30)
        c3.metric("Profit", f"{prof_30:+.2f} U", delta_color="normal")
        c3.metric("Win Rate", f"{rate_30:.1%}")
        
        st.divider()

        # --- CHARTS & LOGS (CONTEXT) ---
        st.subheader("💰 30-Day Profit Trend")
        
        # Recalculate cumulative sum for the chart
        hist['units'] = hist['pick_correct'].apply(lambda x: 1.0 if x else -1.1)
        hist['cumulative_units'] = hist['units'].cumsum()
        
        chart = alt.Chart(hist).mark_line(color='#4CAF50').encode(
            x=alt.X('date', title='Date'), 
            y=alt.Y('cumulative_units', title='Total Units Won'), 
            tooltip=['date', 'cumulative_units']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        
        with st.expander("📜 View Full Bet History"):
            hist['Result'] = hist['pick_correct'].apply(lambda x: "✅ WIN" if x else "❌ LOSS")
            hist['Date'] = hist['date'].dt.strftime("%m/%d")
            hist['Spread'] = hist['picked_spread'].apply(lambda x: round(x * 2) / 2)
            hist['Pick'] = hist['picked_team'] + " " + hist['Spread'].astype(str)
            
            st.dataframe(hist[['Date', 'Pick', 'Result']].sort_values('Date', ascending=False), use_container_width=True, hide_index=True)
        
    else:
        st.info("⚠️ Performance log missing.")
        if st.button("Run Backtest Simulation"):
            log_placeholder = st.empty()
            with st.spinner("Training models (approx 60s)..."):
                f = io.StringIO()
                try:
                    with redirect_stdout(f):
                        backtest.DATA_FILE = DATA_FILE
                        backtest.OUTPUT_FILE = PERF_FILE
                        backtest.run_backtest()
                    output = f.getvalue()
                    log_placeholder.code(output)
                    if os.path.exists(PERF_FILE):
                        st.success("Done!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Backtest Crash: {e}")

# --- DIAGNOSTICS ---
with st.expander("🛠 System Check"):
    st.write(f"**Current Directory:** `{os.getcwd()}`")
    st.write(f"**Data File:** `{DATA_FILE}`")
    st.write(f"**Performance File:** `{PERF_FILE}`")
    if os.path.exists(DATA_FILE):
        st.caption(f"Data Size: {os.path.getsize(DATA_FILE) / 1024:.1f} KB")