import streamlit as st
import pandas as pd
import os
import altair as alt
import predict
import backtest  # <--- IMPORT DIRECTLY
import io
from contextlib import redirect_stdout

# --- PATH CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_FILE = os.path.join(BASE_DIR, "daily_predictions.csv")
PERF_FILE = os.path.join(BASE_DIR, "performance_log.csv")
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")

st.set_page_config(page_title="CBB Quant Edge", page_icon="🏀", layout="centered")

st.title("🏀 CBB Quant Edge")
st.caption("v2.7 | Strategy: Efficiency Differentials | Mode: Spread Specialist")
st.divider()

tab1, tab2 = st.tabs(["📅 Daily Picks", "📈 Performance (Last 4 Weeks)"])

# ==========================================
# TAB 1: DAILY PICKS
# ==========================================
with tab1:
    if os.path.exists(PRED_FILE):
        df = pd.read_csv(PRED_FILE)
        st.subheader(f"Today's Slate ({len(df)} Games)")
        
        if 'Conf' in df.columns:
            df['Confidence'] = df['Conf'].apply(lambda x: f"{x:.1%}")
        
        display_cols = ['Date/Time', 'Matchup', 'Pick', 'Confidence', 'Rest']
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
# TAB 2: PERFORMANCE
# ==========================================
with tab2:
    if os.path.exists(PERF_FILE):
        hist = pd.read_csv(PERF_FILE)
        hist['date'] = pd.to_datetime(hist['date'])
        
        total_bets = len(hist)
        wins = hist['pick_correct'].sum()
        win_rate = wins / total_bets if total_bets > 0 else 0.0
        
        hist['units'] = hist['pick_correct'].apply(lambda x: 1.0 if x else -1.1)
        hist['cumulative_units'] = hist['units'].cumsum()
        total_profit = hist['units'].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Bets", total_bets)
        col2.metric("Win Rate", f"{win_rate:.1%}", delta=f"{win_rate - 0.5238:.1%} vs Vegas")
        col3.metric("Profit (Units)", f"{total_profit:+.2f}")
        
        chart = alt.Chart(hist).mark_line(color='#4CAF50').encode(
            x=alt.X('date', title='Date'), y=alt.Y('cumulative_units', title='Units'), tooltip=['date', 'cumulative_units']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        
        st.subheader("Graded Bet Log")
        hist['Result'] = hist['pick_correct'].apply(lambda x: "✅ WIN" if x else "❌ LOSS")
        hist['Date'] = hist['date'].dt.strftime("%m/%d")
        hist['Spread'] = hist['picked_spread'].apply(lambda x: round(x * 2) / 2)
        hist['Pick'] = hist['picked_team'] + " " + hist['Spread'].astype(str)
        
        st.dataframe(hist[['Date', 'Pick', 'Result']].sort_values('Date', ascending=False), use_container_width=True, hide_index=True)
        
    else:
        st.info("⚠️ Performance log missing.")
        
        # LOGGING BUTTON FOR BACKTEST
        if st.button("Run Backtest Simulation"):
            log_placeholder = st.empty()
            with st.spinner("Training models (approx 60s)..."):
                f = io.StringIO()
                try:
                    with redirect_stdout(f):
                        # Force paths
                        backtest.DATA_FILE = DATA_FILE
                        backtest.OUTPUT_FILE = PERF_FILE
                        backtest.run_backtest()
                    
                    output = f.getvalue()
                    log_placeholder.code(output) # SHOW US THE ERROR IF IT FAILS
                    
                    if os.path.exists(PERF_FILE):
                        st.success("Done!")
                        st.rerun()
                    else:
                        st.error("Backtest finished but file was not created. See logs above.")
                except Exception as e:
                    st.error(f"Backtest Crash: {e}")

# --- DIAGNOSTICS (Expand to check files) ---
with st.expander("🛠 System Check (Click if things are broken)"):
    st.write(f"**Current Directory:** `{os.getcwd()}`")
    st.write("**Files in Base Directory:**")
    st.code("\n".join(os.listdir(BASE_DIR)))
    
    st.write(f"**Training Data Found?** {os.path.exists(DATA_FILE)}")
    if os.path.exists(DATA_FILE):
        st.caption(f"Size: {os.path.getsize(DATA_FILE) / 1024:.1f} KB")