import streamlit as st
import pandas as pd
import os
import altair as alt
import predict
import io
from contextlib import redirect_stdout

# --- CONFIG ---
st.set_page_config(page_title="CBB Quant Edge", page_icon="🏀", layout="centered")

st.title("🏀 CBB Quant Edge")
st.caption("v2.5 | Strategy: Efficiency Differentials | Mode: Spread Specialist")
st.divider()

tab1, tab2 = st.tabs(["📅 Daily Picks", "📈 Performance (Last 4 Weeks)"])

# ==========================================
# TAB 1: DAILY PICKS
# ==========================================
with tab1:
    if os.path.exists("daily_predictions.csv"):
        df = pd.read_csv("daily_predictions.csv")
        st.subheader(f"Today's Slate ({len(df)} Games)")
        
        if 'Conf' in df.columns:
            df['Confidence'] = df['Conf'].apply(lambda x: f"{x:.1%}")
        
        display_cols = ['Date/Time', 'Matchup', 'Pick', 'Confidence', 'Rest']
        valid_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(
            df[valid_cols].style.map(lambda x: "font-weight: bold", subset=['Pick']),
            use_container_width=True,
            height=(len(df) + 1) * 35 + 3,
            hide_index=True
        )
        st.caption("*Green Rows = Strong Edge (>54%). Grey Rows = Lean (>53%).*")
        
        if st.button("Refresh Predictions"):
            with st.spinner("Updating..."):
                predict.main()
            st.rerun()

    else:
        st.warning("⚠️ No predictions found (daily_predictions.csv is missing).")
        st.markdown("### 🛠 Debugger")
        
        if st.button("Run Prediction Engine (With Logs)"):
            status_placeholder = st.empty()
            log_placeholder = st.empty()
            
            with st.spinner("Running engine..."):
                # CAPTURE THE LOGS
                f = io.StringIO()
                try:
                    with redirect_stdout(f):
                        predict.main()
                    output = f.getvalue()
                    
                    # Display the logs to the user
                    log_placeholder.code(output, language="text")
                    
                    if os.path.exists("daily_predictions.csv"):
                        status_placeholder.success("Success! Predictions generated.")
                        st.rerun()
                    else:
                        status_placeholder.error("Engine ran but produced no file. See logs above.")
                        
                except Exception as e:
                    st.error(f"Critical Error: {e}")