import streamlit as st
import pandas as pd
import os
import altair as alt

# --- CONFIG ---
st.set_page_config(page_title="CBB Quant Edge", page_icon="🏀", layout="centered")

st.title("🏀 CBB Quant Edge")
st.caption("v2.4 | Strategy: Efficiency Differentials | Mode: Spread Specialist")
st.divider()

tab1, tab2 = st.tabs(["📅 Daily Picks", "📈 Performance (Last 4 Weeks)"])

# ==========================================
# TAB 1: DAILY PICKS
# ==========================================
with tab1:
    if os.path.exists("daily_predictions.csv"):
        df = pd.read_csv("daily_predictions.csv")
        st.subheader(f"Today's Slate ({len(df)} Games)")
        
        df['Confidence'] = df['Conf'].apply(lambda x: f"{x:.1%}")
        display_cols = ['Date/Time', 'Matchup', 'Pick', 'Confidence', 'Rest']
        
        st.dataframe(
            df[display_cols].style.map(lambda x: "font-weight: bold", subset=['Pick']),
            use_container_width=True,
            height=(len(df) + 1) * 35 + 3,
            hide_index=True
        )
        st.caption("*Green Rows = Strong Edge (>54%). Grey Rows = Lean (>53%).*")

    else:
        st.info("⚠️ No predictions found. Run the engine first.")
        if st.button("Run Prediction Engine"):
            os.system("python3 predict.py")
            st.rerun()

# ==========================================
# TAB 2: PERFORMANCE
# ==========================================
with tab2:
    if os.path.exists("performance_log.csv"):
        hist = pd.read_csv("performance_log.csv")
        hist['date'] = pd.to_datetime(hist['date'])
        
        total_bets = len(hist)
        wins = hist['pick_correct'].sum()
        win_rate = wins / total_bets if total_bets > 0 else 0.0
        
        hist['units'] = hist['pick_correct'].apply(lambda x: 1.0 if x else -1.1)
        hist['cumulative_units'] = hist['units'].cumsum()
        total_profit = hist['units'].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Actionable Bets", total_bets)
        col2.metric("Win Rate", f"{win_rate:.1%}", delta=f"{win_rate - 0.5238:.1%} vs Vegas")
        col3.metric("Profit (Units)", f"{total_profit:+.2f}", delta_color="normal")
        
        st.divider()
        st.subheader("💰 Profit Trajectory")
        
        chart = alt.Chart(hist).mark_line(color='#4CAF50').encode(
            x=alt.X('date', title='Date'),
            y=alt.Y('cumulative_units', title='Cumulative Units Won'),
            tooltip=['date', 'cumulative_units']
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
        
        st.subheader("📜 Graded Bet Log")
        
        hist['Result'] = hist['pick_correct'].apply(lambda x: "✅ WIN" if x else "❌ LOSS")
        hist['Date'] = hist['date'].dt.strftime("%m/%d")
        hist['Conf'] = hist['conf'].apply(lambda x: f"{x:.1%}")
        
        # Round spread for cleaner UI
        hist['Spread'] = hist['picked_spread'].apply(lambda x: round(x * 2) / 2)
        
        # Create 'Pick' column combining Team + Spread
        hist['Pick'] = hist['picked_team'] + " " + hist['Spread'].astype(str)
        
        show_cols = ['Date', 'Pick', 'Conf', 'Result']
        
        display_hist = hist.sort_values('date', ascending=False)[show_cols]
        
        st.dataframe(
            display_hist,
            use_container_width=True,
            hide_index=True
        )
        
    else:
        st.warning("⚠️ No performance history found.")
        st.markdown("Run the **Walk-Forward Backtest** to generate an honest grade of the last 4 weeks.")
        if st.button("Run Backtest Simulation"):
            with st.spinner("Training historical models..."):
                os.system("python3 backtest.py")
            st.success("Backtest Complete!")
            st.rerun()