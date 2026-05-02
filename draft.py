"""NCAA Tournament Draft Helper -- Streamlit dashboard.

Usage: uv run streamlit run draft.py
"""

import html as html_mod
import os

import pandas as pd
import streamlit as st

from tournament import (
    build_sample_bracket,
    compute_probabilities,
    enrich_bracket,
    fetch_bracket_espn,
    load_hasla_latest,
    load_torvik_latest,
    load_torvik_team_map,
    parse_bracket_text,
    snake_order,
)

st.set_page_config(page_title="NCAA Draft Helper", layout="wide")

# ---------------------------------------------------------------------------
# CSS -- reuses palette and patterns from app.py
# ---------------------------------------------------------------------------

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

.stApp { background-color: var(--bg); }

.main .block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    background: var(--green-900);
    width: 280px !important;
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
    color: rgba(255,255,255,0.8) !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: var(--font-body) !important;
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

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: var(--font-body) !important;
    color: var(--green-900) !important;
    letter-spacing: -0.02em;
}

p, span, div, .stMarkdown { font-family: var(--font-body); }

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

.section-title {
    font-family: var(--font-body);
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--neutral-500);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 1rem 0 0.5rem 0;
}

.draft-log-entry {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--neutral-700);
    padding: 4px 0;
    border-bottom: 1px solid var(--neutral-100);
}

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

.streamlit-expanderHeader {
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--green-900);
    background: var(--neutral-50);
    border-radius: 8px;
}

.stDataFrame { font-family: var(--font-body); }

.stDataFrame [data-testid="stDataFrameResizable"] {
    border: 1px solid var(--neutral-200);
    border-radius: 10px;
    overflow: hidden;
}

hr {
    border: none;
    border-top: 1px solid var(--neutral-200);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAFT_RESULTS_FILE = os.path.join(BASE_DIR, "draft_results.csv")


def _esc(val) -> str:
    return html_mod.escape(str(val))


def _save_draft_picks():
    """Write current draft picks to CSV."""
    picks = st.session_state.get("draft_picks", [])
    if not picks:
        if os.path.exists(DRAFT_RESULTS_FILE):
            os.remove(DRAFT_RESULTS_FILE)
        return
    rows = []
    sim_df = st.session_state.get("sim_cache")
    for team_name, drafter_name, pick_num in picks:
        row = {"pick": pick_num + 1, "drafter": drafter_name, "team": team_name}
        if sim_df is not None:
            match = sim_df[sim_df["team"] == team_name]
            if not match.empty:
                row["seed"] = int(match["seed"].values[0])
                row["region"] = match["region"].values[0]
                row["barthag"] = match["barthag"].values[0]
                row["P(Champ)"] = match["P(Champ)"].values[0]
                row["Exp Wins"] = match["Exp Wins"].values[0]
        rows.append(row)
    pd.DataFrame(rows).to_csv(DRAFT_RESULTS_FILE, index=False)


def _load_draft_picks() -> list[tuple]:
    """Load draft picks from CSV if it exists."""
    if not os.path.exists(DRAFT_RESULTS_FILE):
        return []
    try:
        df = pd.read_csv(DRAFT_RESULTS_FILE)
        if df.empty:
            return []
        return [(row["team"], row["drafter"], int(row["pick"]) - 1) for _, row in df.iterrows()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "bracket" not in st.session_state:
    st.session_state["bracket"] = None
if "draft_picks" not in st.session_state:
    st.session_state["draft_picks"] = _load_draft_picks()
if "drafters" not in st.session_state:
    st.session_state["drafters"] = [f"Drafter {i+1}" for i in range(8)]
if "sim_cache" not in st.session_state:
    st.session_state["sim_cache"] = None
if "num_drafters" not in st.session_state:
    st.session_state["num_drafters"] = 8
if "my_drafter_idx" not in st.session_state:
    st.session_state["my_drafter_idx"] = 0
if "mode" not in st.session_state:
    st.session_state["mode"] = None


def _load_bracket_live():
    """Fetch bracket from ESPN and compute probabilities."""
    raw = fetch_bracket_espn()
    if raw:
        torvik_df = load_torvik_latest()
        hasla_df = load_hasla_latest()
        torvik_map_df = load_torvik_team_map()
        bracket = enrich_bracket(raw, torvik_df, hasla_df, torvik_map_df)
        st.session_state["bracket"] = bracket
        st.session_state["sim_cache"] = compute_probabilities(bracket)
        st.session_state["draft_picks"] = []
        _save_draft_picks()
        st.session_state["mode"] = "live"
        return True
    return False


def _load_bracket_sample():
    """Build sample bracket from Torvik ratings."""
    bracket = build_sample_bracket()
    if bracket:
        st.session_state["bracket"] = bracket
        st.session_state["sim_cache"] = compute_probabilities(bracket)
        st.session_state["draft_picks"] = []
        _save_draft_picks()
        st.session_state["mode"] = "test"
        return True
    return False


def _load_bracket_manual(raw_bracket):
    """Load bracket from manual paste."""
    torvik_df = load_torvik_latest()
    hasla_df = load_hasla_latest()
    torvik_map_df = load_torvik_team_map()
    bracket = enrich_bracket(raw_bracket, torvik_df, hasla_df, torvik_map_df)
    st.session_state["bracket"] = bracket
    if len(bracket) == 4:
        st.session_state["sim_cache"] = compute_probabilities(bracket)
        st.session_state["draft_picks"] = []
        _save_draft_picks()
        st.session_state["mode"] = "live"
    return bracket


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Draft Helper</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">NCAA Tournament Pool</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Mode selector
    mode_label = st.radio(
        "Mode",
        options=["test", "live"],
        format_func=lambda m: "Test -- sample bracket from Torvik ratings" if m == "test" else "Live -- fetch real bracket from ESPN",
        index=0 if st.session_state["mode"] != "live" else 1,
        key="mode_radio",
    )

    if mode_label == "test" and st.session_state["mode"] != "test":
        with st.spinner("Building sample bracket..."):
            if _load_bracket_sample():
                st.rerun()

    if mode_label == "live" and st.session_state["mode"] != "live":
        st.session_state["mode"] = "live"
        st.session_state["bracket"] = None
        st.session_state["sim_cache"] = None
        st.session_state["draft_picks"] = []
        _save_draft_picks()
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("**Drafters**")
    num_drafters = st.number_input("Number of drafters", min_value=2, max_value=16, value=st.session_state["num_drafters"], key="num_drafters_input")
    if num_drafters != st.session_state["num_drafters"]:
        st.session_state["num_drafters"] = num_drafters
        while len(st.session_state["drafters"]) < num_drafters:
            st.session_state["drafters"].append(f"Drafter {len(st.session_state['drafters'])+1}")
        st.session_state["drafters"] = st.session_state["drafters"][:num_drafters]

    for i in range(st.session_state["num_drafters"]):
        default = st.session_state["drafters"][i] if i < len(st.session_state["drafters"]) else f"Drafter {i+1}"
        name = st.text_input(f"#{i+1}", value=default, key=f"drafter_{i}", label_visibility="collapsed")
        if i < len(st.session_state["drafters"]):
            st.session_state["drafters"][i] = name

    my_idx = st.selectbox(
        "Your seat",
        options=list(range(st.session_state["num_drafters"])),
        format_func=lambda i: st.session_state["drafters"][i],
        index=st.session_state["my_drafter_idx"],
    )
    st.session_state["my_drafter_idx"] = my_idx

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Bracket actions depend on mode
    if st.session_state["mode"] == "live":
        st.markdown("**Bracket**")
        if st.button("Fetch Bracket (ESPN)"):
            with st.spinner("Fetching bracket..."):
                if _load_bracket_live():
                    st.success("Bracket loaded from ESPN.")
                else:
                    st.error("Could not fetch bracket from ESPN. Use manual entry below.")

        with st.expander("Manual bracket entry"):
            st.caption("Paste one region at a time. Format: '1 Duke' per line. Play-in: '11 VCU / SDSU'")
            manual_region = st.text_input("Region name", value="South", key="manual_region")
            manual_text = st.text_area("Bracket text", height=200, key="manual_text")
            if st.button("Add Region"):
                if manual_text.strip() and manual_region.strip():
                    parsed = parse_bracket_text(manual_text, manual_region)
                    raw_bracket = st.session_state.get("_raw_bracket", {})
                    raw_bracket[manual_region] = parsed
                    st.session_state["_raw_bracket"] = raw_bracket
                    bracket = _load_bracket_manual(raw_bracket)
                    if len(bracket) == 4:
                        st.success(f"All 4 regions loaded. {sum(len(v) for v in bracket.values())} teams ready.")
                    else:
                        st.info(f"Region '{manual_region}' added. {len(bracket)}/4 regions loaded.")

    elif st.session_state["mode"] == "test":
        if st.button("Reset Sample Bracket"):
            with st.spinner("Rebuilding sample bracket..."):
                if _load_bracket_sample():
                    st.success("Sample bracket reset.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if st.button("Undo Last Pick"):
        if st.session_state["draft_picks"]:
            removed = st.session_state["draft_picks"].pop()
            _save_draft_picks()
            st.success(f"Undid: {removed[1]} - {removed[0]}")
        else:
            st.warning("No picks to undo.")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

sim_df = st.session_state.get("sim_cache")

if sim_df is None or sim_df.empty:
    st.markdown("### NCAA Tournament Draft Helper")
    if st.session_state["mode"] == "live":
        st.markdown("Click **Fetch Bracket (ESPN)** in the sidebar to load the real bracket.")
    else:
        st.markdown("Select a mode from the sidebar to begin.")
    st.stop()

# Mode banner
if st.session_state["mode"] == "test":
    st.caption("TEST MODE -- sample bracket built from current Torvik ratings, not the real tournament bracket")

# Build drafted set and current pick info
drafted_teams = {pick[0] for pick in st.session_state["draft_picks"]}
total_picks = len(sim_df)
pick_order = snake_order(st.session_state["num_drafters"], total_picks)
current_pick_num = len(st.session_state["draft_picks"])
drafters = st.session_state["drafters"]
my_idx = st.session_state["my_drafter_idx"]
my_name = drafters[my_idx]

# Derive status column
sim_display = sim_df.copy()
status_map = {}
for team_name, drafter_name, pick_num in st.session_state["draft_picks"]:
    status_map[team_name] = drafter_name
sim_display["Status"] = sim_display["team"].map(lambda t: status_map.get(t, "Available"))

available_df = sim_display[sim_display["Status"] == "Available"]
my_teams = sim_display[sim_display["Status"] == my_name]

# KPIs
teams_available = len(available_df)
best_my_champ = f"{my_teams['P(Champ)'].max():.5%}" if not my_teams.empty else "--"
best_available_row = available_df.iloc[0] if not available_df.empty else None
best_available_str = (
    f"{_esc(best_available_row['team'])} ({best_available_row['P(Champ)']:.5%})"
    if best_available_row is not None else "--"
)
if current_pick_num < len(pick_order):
    current_drafter = drafters[pick_order[current_pick_num]]
    current_pick_str = f"#{current_pick_num + 1}: {_esc(current_drafter)}"
else:
    current_pick_str = "Draft complete"

st.markdown(f'''
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-value">{teams_available}</div>
        <div class="kpi-label">Available</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{best_my_champ}</div>
        <div class="kpi-label">Your Best P(Champ)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{best_available_str}</div>
        <div class="kpi-label">Best Available</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{current_pick_str}</div>
        <div class="kpi-label">Current Pick</div>
    </div>
</div>
''', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Draft action bar
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Draft a Team</div>', unsafe_allow_html=True)
col_team, col_drafter, col_btn = st.columns([3, 2, 1])

available_names = available_df["team"].tolist()
with col_team:
    draft_team = st.selectbox(
        "Team",
        options=available_names,
        format_func=lambda t: f"{t} ({available_df.loc[available_df['team'] == t, 'seed'].values[0]}-seed, "
                              f"{available_df.loc[available_df['team'] == t, 'P(Champ)'].values[0]:.5%})",
        label_visibility="collapsed",
    )

# Default drafter to whoever is on the clock
default_drafter_idx = pick_order[current_pick_num] if current_pick_num < len(pick_order) else 0
with col_drafter:
    draft_drafter = st.selectbox(
        "Drafter",
        options=drafters[:st.session_state["num_drafters"]],
        index=default_drafter_idx,
        label_visibility="collapsed",
    )

with col_btn:
    if st.button("Draft"):
        if draft_team and draft_drafter:
            st.session_state["draft_picks"].append(
                (draft_team, draft_drafter, current_pick_num)
            )
            _save_draft_picks()
            st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Team rankings table
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Team Rankings</div>', unsafe_allow_html=True)

# Filters
filter_col1, filter_col2, filter_col3 = st.columns([2, 3, 3])
with filter_col1:
    view_filter = st.segmented_control(
        "View",
        options=["Available", "All", "My Team"],
        default="Available",
        label_visibility="collapsed",
    )
with filter_col2:
    all_regions = sorted(sim_display["region"].unique())
    region_filter = st.multiselect("Region", options=all_regions, default=all_regions, label_visibility="collapsed")
with filter_col3:
    search_text = st.text_input("Search", placeholder="Search teams...", label_visibility="collapsed")

# Apply filters
filtered = sim_display.copy()
if view_filter == "Available":
    filtered = filtered[filtered["Status"] == "Available"]
elif view_filter == "My Team":
    filtered = filtered[filtered["Status"] == my_name]

if region_filter:
    filtered = filtered[filtered["region"].isin(region_filter)]

if search_text:
    mask = filtered["team"].str.contains(search_text, case=False, na=False)
    filtered = filtered[mask]

# Format for display
display_cols = ["team", "seed", "region", "conf", "barthag", "hasla_rank",
                "P(S16)", "P(F4)", "P(Champ)", "Exp Wins", "Status"]
display_df = filtered[display_cols].copy()

# Format percentages
for col in ["P(S16)", "P(F4)", "P(Champ)"]:
    display_df[col] = display_df[col].map(lambda x: f"{x:.5%}")
display_df["Exp Wins"] = display_df["Exp Wins"].map(lambda x: f"{x:.2f}")
display_df["barthag"] = display_df["barthag"].map(lambda x: f"{x:.4f}")
display_df["hasla_rank"] = display_df["hasla_rank"].map(lambda x: str(int(x)) if x > 0 else "--")

display_df = display_df.rename(columns={
    "team": "Team",
    "seed": "Seed",
    "region": "Region",
    "conf": "Conf",
    "barthag": "Barthag",
    "hasla_rank": "HasLA",
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=False,
    height=600,
)

# ---------------------------------------------------------------------------
# Draft log
# ---------------------------------------------------------------------------

with st.expander(f"Draft Log ({len(st.session_state['draft_picks'])} picks)"):
    if not st.session_state["draft_picks"]:
        st.caption("No picks yet.")
    else:
        for team_name, drafter_name, pick_num in st.session_state["draft_picks"]:
            row = sim_df[sim_df["team"] == team_name]
            seed = int(row["seed"].values[0]) if not row.empty else "?"
            p_champ = f"{row['P(Champ)'].values[0]:.5%}" if not row.empty else "?"
            st.markdown(
                f'<div class="draft-log-entry">'
                f'Pick {pick_num + 1}: <strong>{_esc(drafter_name)}</strong> selects '
                f'<strong>{_esc(team_name)}</strong> ({seed}-seed, P(Champ): {p_champ})'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# My team summary
# ---------------------------------------------------------------------------

if not my_teams.empty:
    with st.expander(f"My Team: {my_name} ({len(my_teams)} teams)"):
        my_display = my_teams[["team", "seed", "region", "P(Champ)", "Exp Wins"]].copy()
        my_display["P(Champ)"] = my_display["P(Champ)"].map(lambda x: f"{x:.5%}")
        my_display["Exp Wins"] = my_display["Exp Wins"].map(lambda x: f"{x:.2f}")
        total_exp_wins = my_teams["Exp Wins"].sum()
        total_champ = my_teams["P(Champ)"].sum()
        st.dataframe(my_display, use_container_width=True, hide_index=True)
        st.caption(f"Combined P(Champ): {total_champ:.5%} | Total Exp Wins: {total_exp_wins:.2f}")
