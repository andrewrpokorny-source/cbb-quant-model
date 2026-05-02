"""NCAA Tournament bracket simulation via log5 matchup probabilities.

Pure logic module -- no Streamlit dependency. Provides bracket fetching/parsing,
log5 win probability computation, and full tournament simulation to produce
P(Championship) for each team.
"""

import json
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches

import numpy as np
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Standard bracket seed pairings per region (R64 matchups)
BRACKET_PAIRINGS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

# Standard Final Four region pairing: (region_idx_0 vs region_idx_1), (region_idx_2 vs region_idx_3)
# ESPN convention: regions listed as top-left, bottom-left, top-right, bottom-right
# FF matchups: index 0 vs 1, index 2 vs 3
FF_PAIRINGS = [(0, 1), (2, 3)]

REGIONS_ORDER = ["South", "East", "West", "Midwest"]

ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "NCG"]


@dataclass
class TournamentTeam:
    name: str
    seed: int
    region: str
    barthag: float
    conf: str = ""
    adj_oe: float = 0.0
    adj_de: float = 0.0
    hasla_rank: int = 0
    record: str = ""
    wab: float = 0.0
    is_playin: bool = False
    playin_partner: str = ""


def log5(p_a: float, p_b: float) -> float:
    """P(A beats B) using log5 formula with barthag ratings.

    p_a and p_b are each team's barthag (probability of beating an average team).
    """
    if p_a <= 0:
        return 0.0
    if p_b <= 0:
        return 1.0
    if p_a >= 1:
        return 1.0
    if p_b >= 1:
        return 0.0
    num = p_a * (1 - p_b)
    denom = p_a * (1 - p_b) + p_b * (1 - p_a)
    if denom == 0:
        return 0.5
    return num / denom


def snake_order(num_drafters: int, total_picks: int) -> list[int]:
    """Generate snake draft order (0-indexed drafter indices).

    Round 1: 0,1,...,n-1; Round 2: n-1,...,1,0; etc.
    """
    order = []
    forward = list(range(num_drafters))
    backward = list(reversed(forward))
    rnd = 0
    while len(order) < total_picks:
        seq = forward if rnd % 2 == 0 else backward
        for idx in seq:
            if len(order) >= total_picks:
                break
            order.append(idx)
        rnd += 1
    return order


def _weighted_playin_barthag(barthag_a: float, barthag_b: float) -> float:
    """Weighted barthag for a play-in pair, weighted by each team's log5 P(win)."""
    p_a_wins = log5(barthag_a, barthag_b)
    return p_a_wins * barthag_a + (1 - p_a_wins) * barthag_b


def simulate_region(teams: list[TournamentTeam]) -> dict[str, list[float]]:
    """Simulate a 16-team region through E8, returning per-round win probabilities.

    teams must be ordered by seed position matching BRACKET_PAIRINGS:
    [1-seed, 16-seed, 8-seed, 9-seed, 5-seed, 12-seed, 4-seed, 13-seed,
     6-seed, 11-seed, 3-seed, 14-seed, 7-seed, 10-seed, 2-seed, 15-seed]

    Returns dict mapping team name -> [P(win_R64), P(win_R32), P(win_S16), P(win_E8)]
    """
    n = len(teams)
    if n != 16:
        raise ValueError(f"Region must have 16 teams, got {n}")

    names = [t.name for t in teams]
    barthags = np.array([t.barthag for t in teams], dtype=float)

    # prob[round][slot] = probability that team originally in `slot` is alive after round
    # Round 0 = start (all 1.0), Round 1 = after R64, etc.
    num_rounds = 4  # R64, R32, S16, E8
    # Track P(team i reaches and wins round r)
    # alive[i] = probability team i is still in the tournament
    alive = np.ones(n, dtype=float)
    round_probs = {name: [] for name in names}

    # Walk through 4 rounds
    bracket_size = n
    for rnd in range(num_rounds):
        num_games = bracket_size // 2
        new_alive = np.zeros(n, dtype=float)

        # In each round, teams are grouped in consecutive pairs within their bracket half
        # Group size = 2^(rnd+1), pairs within each group compete
        group_size = 2 ** (rnd + 1)
        half = group_size // 2

        for g_start in range(0, n, group_size):
            top_half = list(range(g_start, g_start + half))
            bot_half = list(range(g_start + half, g_start + group_size))

            # Each team in top_half can face each team in bot_half
            for i in top_half:
                if alive[i] == 0:
                    continue
                win_prob_i = 0.0
                for j in bot_half:
                    if alive[j] == 0:
                        continue
                    p_ij = log5(barthags[i], barthags[j])
                    win_prob_i += alive[j] * p_ij
                new_alive[i] = alive[i] * win_prob_i

            for j in bot_half:
                if alive[j] == 0:
                    continue
                win_prob_j = 0.0
                for i in top_half:
                    if alive[i] == 0:
                        continue
                    p_ji = log5(barthags[j], barthags[i])
                    win_prob_j += alive[i] * p_ji
                new_alive[j] = alive[j] * win_prob_j

        alive = new_alive
        for idx, name in enumerate(names):
            round_probs[name].append(alive[idx])

        bracket_size = num_games

    return round_probs


def simulate_final_four(
    region_winners: list[tuple[str, dict[str, float]]],
    region_order: list[str],
) -> dict[str, list[float]]:
    """Simulate Final Four and Championship from regional results.

    region_winners: list of (region_name, {team_name: P(regional_champion)}) for each region.
    region_order: ordered list of region names matching FF_PAIRINGS indices.

    Returns dict mapping team_name -> [P(win_F4_game), P(win_Championship)]
    """
    # Build region index mapping
    region_idx = {name: i for i, name in enumerate(region_order)}

    # Collect all teams with their regional champion probability and barthag-lookup
    all_teams: dict[str, dict] = {}

    for region_name, team_probs in region_winners:
        for team_name, prob_champion in team_probs.items():
            if prob_champion > 0:
                all_teams[team_name] = {
                    "region": region_name,
                    "p_regional": prob_champion,
                }

    # We need barthags for FF matchups -- get them from region_winners structure
    result = {name: [0.0, 0.0] for name in all_teams}

    # FF: two semifinal games
    for sem_idx, (r_a_idx, r_b_idx) in enumerate(FF_PAIRINGS):
        r_a = region_order[r_a_idx]
        r_b = region_order[r_b_idx]
        teams_a = {n: d for n, d in all_teams.items() if d["region"] == r_a}
        teams_b = {n: d for n, d in all_teams.items() if d["region"] == r_b}

        for na, da in teams_a.items():
            ff_win = 0.0
            for nb, db in teams_b.items():
                p_win = log5(da["barthag"], db["barthag"])
                ff_win += da["p_regional"] * db["p_regional"] * p_win
            result[na][0] = ff_win

        for nb, db in teams_b.items():
            ff_win = 0.0
            for na, da in teams_a.items():
                p_win = log5(db["barthag"], da["barthag"])
                ff_win += db["p_regional"] * da["p_regional"] * p_win
            result[nb][0] = ff_win

    # Championship: winners of sem 0 vs sem 1
    sem0_regions = [region_order[FF_PAIRINGS[0][0]], region_order[FF_PAIRINGS[0][1]]]
    sem1_regions = [region_order[FF_PAIRINGS[1][0]], region_order[FF_PAIRINGS[1][1]]]
    sem0_teams = {n: d for n, d in all_teams.items() if d["region"] in sem0_regions}
    sem1_teams = {n: d for n, d in all_teams.items() if d["region"] in sem1_regions}

    for na, da in sem0_teams.items():
        champ_prob = 0.0
        for nb, db in sem1_teams.items():
            p_win = log5(da["barthag"], db["barthag"])
            champ_prob += result[na][0] * result[nb][0] * p_win
        result[na][1] = champ_prob

    for nb, db in sem1_teams.items():
        champ_prob = 0.0
        for na, da in sem0_teams.items():
            p_win = log5(db["barthag"], da["barthag"])
            champ_prob += result[nb][0] * result[na][0] * p_win
        result[nb][1] = champ_prob

    return result


def compute_probabilities(
    bracket: dict[str, list[TournamentTeam]],
    region_order: list[str] | None = None,
) -> pd.DataFrame:
    """Run full tournament simulation and return a DataFrame with all probabilities.

    bracket: dict mapping region_name -> list of 16 TournamentTeam in bracket order.
    region_order: list of region names controlling FF pairings. Defaults to REGIONS_ORDER.

    Returns DataFrame with columns:
        team, seed, region, conf, barthag, adj_oe, adj_de, hasla_rank, record, wab,
        is_playin, playin_partner,
        P(R32), P(S16), P(F4), P(Champ), expected_wins
    """
    if region_order is None:
        region_order = list(bracket.keys())

    # Simulate each region
    regional_results = {}
    team_lookup = {}
    for region_name in region_order:
        teams = bracket[region_name]
        region_probs = simulate_region(teams)
        regional_results[region_name] = region_probs
        for t in teams:
            team_lookup[t.name] = t

    # Build region_winners for FF simulation with barthag info
    region_winners = []
    for region_name in region_order:
        probs = regional_results[region_name]
        # P(regional champion) = P(win_E8) = last element in region probs
        champ_probs = {name: rounds[-1] for name, rounds in probs.items()}
        # Attach barthag to all_teams in simulate_final_four -- need to patch
        region_winners.append((region_name, champ_probs))

    # Patch barthag into the all_teams dict used by simulate_final_four
    # We'll call it directly with barthag info
    all_teams_ff: dict[str, dict] = {}
    for region_name, team_probs in region_winners:
        for team_name, prob_champion in team_probs.items():
            if prob_champion > 0:
                t = team_lookup[team_name]
                all_teams_ff[team_name] = {
                    "region": region_name,
                    "p_regional": prob_champion,
                    "barthag": t.barthag,
                }

    # FF simulation (inline to pass barthag)
    ff_result = {name: [0.0, 0.0] for name in all_teams_ff}

    for r_a_idx, r_b_idx in FF_PAIRINGS:
        r_a = region_order[r_a_idx]
        r_b = region_order[r_b_idx]
        teams_a = {n: d for n, d in all_teams_ff.items() if d["region"] == r_a}
        teams_b = {n: d for n, d in all_teams_ff.items() if d["region"] == r_b}

        for na, da in teams_a.items():
            ff_win = 0.0
            for nb, db in teams_b.items():
                p_win = log5(da["barthag"], db["barthag"])
                ff_win += da["p_regional"] * db["p_regional"] * p_win
            ff_result[na][0] = ff_win

        for nb, db in teams_b.items():
            ff_win = 0.0
            for na, da in teams_a.items():
                p_win = log5(db["barthag"], da["barthag"])
                ff_win += db["p_regional"] * da["p_regional"] * p_win
            ff_result[nb][0] = ff_win

    sem0_regions = {region_order[FF_PAIRINGS[0][0]], region_order[FF_PAIRINGS[0][1]]}
    sem1_regions = {region_order[FF_PAIRINGS[1][0]], region_order[FF_PAIRINGS[1][1]]}
    sem0_teams = {n: d for n, d in all_teams_ff.items() if d["region"] in sem0_regions}
    sem1_teams = {n: d for n, d in all_teams_ff.items() if d["region"] in sem1_regions}

    for na, da in sem0_teams.items():
        champ_prob = 0.0
        for nb, db in sem1_teams.items():
            p_win = log5(da["barthag"], db["barthag"])
            champ_prob += ff_result[na][0] * ff_result[nb][0] * p_win
        ff_result[na][1] = champ_prob

    for nb, db in sem1_teams.items():
        champ_prob = 0.0
        for na, da in sem0_teams.items():
            p_win = log5(db["barthag"], da["barthag"])
            champ_prob += ff_result[nb][0] * ff_result[na][0] * p_win
        ff_result[nb][1] = champ_prob

    # Build output rows
    rows = []
    for region_name in region_order:
        region_probs = regional_results[region_name]
        for team_name, round_wins in region_probs.items():
            t = team_lookup[team_name]
            # round_wins = [P(win_R64), P(win_R32), P(win_S16), P(win_E8)]
            p_r32 = round_wins[0]  # survived R64
            p_s16 = round_wins[1]  # survived R32
            p_e8 = round_wins[2] if len(round_wins) > 2 else 0.0
            p_f4 = round_wins[3] if len(round_wins) > 3 else 0.0  # regional champ

            ff_probs = ff_result.get(team_name, [0.0, 0.0])
            p_ff_win = ff_probs[0]  # won FF semifinal
            p_champ = ff_probs[1]  # won championship

            # Expected wins = sum of all round probabilities
            expected_wins = p_r32 + p_s16 + p_e8 + p_f4 + p_ff_win + p_champ

            rows.append({
                "team": team_name,
                "seed": t.seed,
                "region": t.region,
                "conf": t.conf,
                "barthag": t.barthag,
                "adj_oe": t.adj_oe,
                "adj_de": t.adj_de,
                "hasla_rank": t.hasla_rank,
                "record": t.record,
                "wab": t.wab,
                "is_playin": t.is_playin,
                "playin_partner": t.playin_partner,
                "P(R32)": p_r32,
                "P(S16)": p_s16,
                "P(E8)": p_e8,
                "P(F4)": p_f4,
                "P(F4 Win)": p_ff_win,
                "P(Champ)": p_champ,
                "Exp Wins": expected_wins,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("P(Champ)", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df


# ---------------------------------------------------------------------------
# Team name resolution
# ---------------------------------------------------------------------------

# Reuse normalization from torvik.py
_TORVIK_NAME_OVERRIDES = {
    "california baptist": "cal baptist",
    "college of charleston": "charleston",
    "boston college eagles": "boston college",
    "boston university terriers": "boston university",
    "grambling tigers": "grambling st",
    "iu indianapolis jaguars": "iu indy",
    "uic": "illinois chicago",
    "long island university sharks": "liu",
    "loyola maryland greyhounds": "loyola md",
    "mass lowell": "umass lowell",
    "mcneese cowboys": "mcneese st",
    "mississippi": "mississippi",
    "mount st marys": "mount st marys",
    "nc state": "nc state",
    "nicholls colonels": "nicholls st",
    "omaha mavericks": "nebraska omaha",
    "north carolina state": "nc state",
    "ole miss": "mississippi",
    "saint josephs": "st josephs",
    "saint marys": "st marys",
    "saint peters": "st peters",
    "sam houston bearkats": "sam houston st",
    "se louisiana lions": "southeastern louisiana",
    "southern methodist": "smu",
    "st thomas minnesota tommies": "st thomas",
    "texas am corpus christi": "texas am corpus chris",
    "ul monroe warhawks": "louisiana monroe",
    "ut martin skyhawks": "tennessee martin",
    "ut rio grande valley": "ut rio grande valley",
}


def _normalize_name(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    replacements = {
        "&": " and ",
        "@": " at ",
        "'": "",
        ".": " ",
        "-": " ",
        "/": " ",
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    text = re.sub(r"\bsaint\b", "st", text)
    text = re.sub(r"\bmount\b", "mt", text)
    text = re.sub(r"\bstate\b", "st", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _TORVIK_NAME_OVERRIDES.get(text, text)


def _load_team_map_json() -> dict:
    path = os.path.join(BASE_DIR, "team_map.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def resolve_team_to_torvik(
    name: str,
    torvik_names: set[str],
    torvik_normalized: dict[str, str],
    team_map_json: dict | None = None,
    torvik_team_map_df: pd.DataFrame | None = None,
) -> str | None:
    """Resolve an ESPN team name to a Torvik team name.

    Chain: exact match -> team_map.json -> torvik_team_map.csv -> normalize -> fuzzy.
    Returns None if no match found.
    """
    if not name:
        return None

    # 1. Exact match on Torvik names
    if name in torvik_names:
        return name

    # 2. team_map.json (ESPN -> canonical)
    if team_map_json:
        mapped = team_map_json.get(name)
        if mapped and mapped in torvik_names:
            return mapped

    # 3. torvik_team_map.csv (ESPN -> Torvik)
    if torvik_team_map_df is not None and not torvik_team_map_df.empty:
        match_rows = torvik_team_map_df[torvik_team_map_df["team"] == name]
        if not match_rows.empty:
            torvik_name = match_rows.iloc[0].get("torvik_team")
            if pd.notna(torvik_name) and torvik_name in torvik_names:
                return torvik_name

    # 4. Normalized match
    norm = _normalize_name(name)
    if norm in torvik_normalized:
        return torvik_normalized[norm]

    # 5. Try with mascot stripped (last word)
    tokens = name.split()
    for trim in (1, 2):
        if len(tokens) > trim:
            candidate = " ".join(tokens[:-trim])
            if candidate in torvik_names:
                return candidate
            norm_c = _normalize_name(candidate)
            if norm_c in torvik_normalized:
                return torvik_normalized[norm_c]

    # 6. Fuzzy match
    close = get_close_matches(norm, list(torvik_normalized.keys()), n=1, cutoff=0.82)
    if close:
        score = SequenceMatcher(None, norm, close[0]).ratio()
        if score >= 0.85:
            return torvik_normalized[close[0]]

    return None


# ---------------------------------------------------------------------------
# Bracket fetching -- ESPN API
# ---------------------------------------------------------------------------

def fetch_bracket_espn(league: str = "mens") -> dict[str, list[dict]] | None:
    """Fetch NCAA tournament bracket from ESPN API.

    The scoreboard endpoint requires a date range covering the full tournament
    (Selection Sunday through championship) to return all 67 games. Without it,
    only the current day's games are returned.

    Returns dict mapping region_name -> list of {name, seed, is_playin, playin_partner}
    ordered by bracket position (matching BRACKET_PAIRINGS).
    Returns None if bracket data is unavailable.
    """
    sport_path = "mens-college-basketball" if league == "mens" else "womens-college-basketball"

    # Tournament window: mid-March through early April
    from datetime import datetime
    now = datetime.now()
    year = now.year
    start = f"{year}0314"
    end = f"{year}0410"
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"{sport_path}/scoreboard?groups=100&limit=200&dates={start}-{end}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    events = data.get("events", [])
    if not events:
        return None

    # Parse teams from R64 events and play-in (First Four) events.
    # Later-round events have TBD teams (seed 99) -- skip those.
    teams_by_region: dict[str, dict[int, dict]] = {}
    playin_matchups: list[tuple[str, dict, dict]] = []  # (region, team1, team2)

    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]

        notes = " ".join(n.get("headline", "") for n in comp.get("notes", []))
        is_playin = "first four" in notes.lower()
        is_first_round = "1st round" in notes.lower()

        # Only care about First Four and 1st Round games
        if not is_playin and not is_first_round:
            continue

        # Extract region from notes (e.g. "... - East Region - 1st Round")
        region = ""
        for r in ["South", "East", "West", "Midwest"]:
            if r in notes:
                region = r
                break

        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        parsed_teams = []
        for c in competitors:
            team_data = c.get("team", {})
            team_name = team_data.get("displayName", team_data.get("shortDisplayName", ""))
            seed_raw = c.get("curatedRank", {}).get("current", 0)
            try:
                seed = int(seed_raw)
            except (ValueError, TypeError):
                seed = 0
            # Skip TBD placeholders (seed 99 or 0)
            if seed in (0, 99):
                continue
            parsed_teams.append({
                "name": team_name,
                "seed": seed,
                "is_playin": False,
                "playin_partner": "",
            })

        if is_playin and len(parsed_teams) == 2 and region:
            playin_matchups.append((region, parsed_teams[0], parsed_teams[1]))
            continue

        if is_first_round and region:
            if region not in teams_by_region:
                teams_by_region[region] = {}
            for t in parsed_teams:
                # First occurrence wins (don't overwrite with TBD play-in slot)
                if t["seed"] not in teams_by_region[region]:
                    teams_by_region[region][t["seed"]] = t

    if not teams_by_region:
        return None

    # Merge play-in pairs into their bracket slot
    for region, t1, t2 in playin_matchups:
        seed = t1["seed"]  # both share the same seed
        if region not in teams_by_region:
            teams_by_region[region] = {}
        # The R64 matchup may already have one of these teams listed;
        # replace or create the slot with the play-in pair
        combined = {
            "name": f"{t1['name']} / {t2['name']}",
            "seed": seed,
            "is_playin": True,
            "playin_partner": t2["name"],
        }
        teams_by_region[region][seed] = combined

    # Order teams by bracket position
    result = {}
    for region_name, seed_map in teams_by_region.items():
        ordered = []
        for s1, s2 in BRACKET_PAIRINGS:
            for s in (s1, s2):
                team = seed_map.get(s, {
                    "name": f"TBD ({s})", "seed": s,
                    "is_playin": False, "playin_partner": "",
                })
                ordered.append(team)
        result[region_name] = ordered

    return result


# ---------------------------------------------------------------------------
# Bracket text parser (fallback)
# ---------------------------------------------------------------------------

def parse_bracket_text(text: str, region: str) -> list[dict]:
    """Parse pasted bracket text into team dicts.

    Accepts lines like:
        1 Duke
        16 A Team / B Team
        8 Michigan St.

    Returns list of dicts with {name, seed, is_playin, playin_partner} in bracket order.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    seed_map: dict[int, dict] = {}

    for line in lines:
        match = re.match(r"^(\d{1,2})\s+(.+)$", line)
        if not match:
            continue
        seed = int(match.group(1))
        team_str = match.group(2).strip()

        if "/" in team_str:
            parts = [p.strip() for p in team_str.split("/", 1)]
            seed_map[seed] = {
                "name": f"{parts[0]} / {parts[1]}",
                "seed": seed,
                "is_playin": True,
                "playin_partner": parts[1],
            }
        else:
            seed_map[seed] = {
                "name": team_str,
                "seed": seed,
                "is_playin": False,
                "playin_partner": "",
            }

    ordered = []
    for s1, s2 in BRACKET_PAIRINGS:
        for s in (s1, s2):
            team = seed_map.get(s, {"name": f"TBD ({s})", "seed": s, "is_playin": False, "playin_partner": ""})
            ordered.append(team)

    return ordered


# ---------------------------------------------------------------------------
# Data loading and team enrichment
# ---------------------------------------------------------------------------

def load_torvik_latest(path: str | None = None) -> pd.DataFrame:
    """Load latest Torvik ratings snapshot."""
    if path is None:
        path = os.path.join(BASE_DIR, "torvik_ratings_snapshots.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    latest = df["snapshot_date"].max()
    return df[df["snapshot_date"] == latest].copy()


def load_hasla_latest(path: str | None = None) -> pd.DataFrame:
    """Load latest HasLA rankings snapshot."""
    if path is None:
        path = os.path.join(BASE_DIR, "hasla_rank_snapshots.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    latest = df["snapshot_date"].max()
    return df[df["snapshot_date"] == latest].copy()


def load_torvik_team_map(path: str | None = None) -> pd.DataFrame:
    """Load torvik_team_map.csv."""
    if path is None:
        path = os.path.join(BASE_DIR, "torvik_team_map.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def enrich_bracket(
    raw_bracket: dict[str, list[dict]],
    torvik_df: pd.DataFrame,
    hasla_df: pd.DataFrame,
    torvik_team_map_df: pd.DataFrame | None = None,
) -> dict[str, list[TournamentTeam]]:
    """Merge bracket teams with Torvik ratings and HasLA ranks.

    Returns dict mapping region -> list[TournamentTeam] in bracket order.
    """
    team_map_json = _load_team_map_json()

    # Build lookup structures
    torvik_names = set(torvik_df["team"].dropna().unique()) if not torvik_df.empty else set()
    torvik_normalized = {}
    for name in torvik_names:
        norm = _normalize_name(name)
        torvik_normalized[norm] = name

    torvik_by_name = {}
    if not torvik_df.empty:
        for _, row in torvik_df.iterrows():
            torvik_by_name[row["team"]] = row

    hasla_by_name = {}
    if not hasla_df.empty:
        # HasLA names may differ from Torvik names -- build normalized lookup
        for _, row in hasla_df.iterrows():
            hasla_by_name[row["team"]] = row

    result = {}
    for region_name, team_dicts in raw_bracket.items():
        teams = []
        for td in team_dicts:
            espn_name = td["name"]
            seed = td["seed"]
            is_playin = td.get("is_playin", False)
            playin_partner = td.get("playin_partner", "")

            # Resolve to Torvik name
            torvik_name = resolve_team_to_torvik(
                espn_name, torvik_names, torvik_normalized,
                team_map_json, torvik_team_map_df,
            )

            # If play-in pair name contains "/", try resolving the first team
            if torvik_name is None and "/" in espn_name:
                first_team = espn_name.split("/")[0].strip()
                torvik_name = resolve_team_to_torvik(
                    first_team, torvik_names, torvik_normalized,
                    team_map_json, torvik_team_map_df,
                )

            barthag = 0.5
            conf = ""
            adj_oe = 0.0
            adj_de = 0.0
            record = ""
            wab = 0.0

            if torvik_name and torvik_name in torvik_by_name:
                row = torvik_by_name[torvik_name]
                barthag = float(row.get("barthag", 0.5))
                conf = str(row.get("conf", ""))
                adj_oe = float(row.get("adj_oe", 0.0))
                adj_de = float(row.get("adj_de", 0.0))
                record = str(row.get("record", ""))
                wab = float(row.get("wab", 0.0))

            # Handle play-in weighted barthag
            if is_playin and playin_partner:
                partner_torvik = resolve_team_to_torvik(
                    playin_partner, torvik_names, torvik_normalized,
                    team_map_json, torvik_team_map_df,
                )
                if partner_torvik and partner_torvik in torvik_by_name:
                    partner_row = torvik_by_name[partner_torvik]
                    partner_barthag = float(partner_row.get("barthag", 0.5))
                    barthag = _weighted_playin_barthag(barthag, partner_barthag)

            # HasLA rank lookup
            hasla_rank = 0
            display_name = torvik_name or espn_name
            if display_name in hasla_by_name:
                hasla_rank = int(hasla_by_name[display_name].get("hasla_rank", 0))

            teams.append(TournamentTeam(
                name=espn_name,
                seed=seed,
                region=region_name,
                barthag=barthag,
                conf=conf,
                adj_oe=adj_oe,
                adj_de=adj_de,
                hasla_rank=hasla_rank,
                record=record,
                wab=wab,
                is_playin=is_playin,
                playin_partner=playin_partner,
            ))

        result[region_name] = teams

    return result


# ---------------------------------------------------------------------------
# Sample bracket for testing the UI
# ---------------------------------------------------------------------------

def build_sample_bracket() -> dict[str, list[TournamentTeam]] | None:
    """Build a hypothetical 68-team bracket from the latest Torvik snapshot.

    Uses real team ratings. Seeds are assigned by barthag rank (S-curve across
    4 regions). Includes 4 play-in pairs at the 11 and 16 seeds.
    Returns enriched bracket ready for compute_probabilities(), or None if
    Torvik data is unavailable.
    """
    torvik_df = load_torvik_latest()
    hasla_df = load_hasla_latest()
    if torvik_df.empty:
        return None

    # Sort by barthag descending -- top 68 teams make the field
    ranked = torvik_df.sort_values("barthag", ascending=False).head(68).reset_index(drop=True)

    regions = ["South", "East", "West", "Midwest"]
    num_regions = len(regions)

    # S-curve assignment: seed 1 goes 0,1,2,3; seed 2 goes 3,2,1,0; etc.
    # We need 64 regular slots + 4 play-in pairs (8 extra teams)
    # First 64 teams fill seeds 1-16 across 4 regions via S-curve
    # Last 4 teams pair with teams at seeds 11a,11b,16a,16b
    region_seeds: dict[str, dict[int, dict]] = {r: {} for r in regions}

    # Map overall rank -> (seed, region) via S-curve
    team_idx = 0
    for seed_line in range(1, 17):
        order = list(range(num_regions)) if seed_line % 2 == 1 else list(reversed(range(num_regions)))
        for region_idx in order:
            if team_idx >= len(ranked):
                break
            row = ranked.iloc[team_idx]
            region_seeds[regions[region_idx]][seed_line] = {
                "name": row["team"],
                "seed": seed_line,
                "is_playin": False,
                "playin_partner": "",
            }
            team_idx += 1

    # Make 4 play-in pairs: two at 16-seed, two at 11-seed
    # Teams 64-67 become play-in partners
    playin_slots = [
        (regions[0], 16),  # 16-seed play-in in region 0
        (regions[1], 16),  # 16-seed play-in in region 1
        (regions[2], 11),  # 11-seed play-in in region 2
        (regions[3], 11),  # 11-seed play-in in region 3
    ]
    for i, (region, seed) in enumerate(playin_slots):
        partner_idx = 64 + i
        if partner_idx < len(ranked):
            partner_row = ranked.iloc[partner_idx]
            existing = region_seeds[region][seed]
            existing["is_playin"] = True
            existing["playin_partner"] = partner_row["team"]
            existing["name"] = f"{existing['name']} / {partner_row['team']}"

    # Build raw bracket in BRACKET_PAIRINGS order per region
    raw_bracket: dict[str, list[dict]] = {}
    for region in regions:
        ordered = []
        for s1, s2 in BRACKET_PAIRINGS:
            for s in (s1, s2):
                team = region_seeds[region].get(
                    s, {"name": f"TBD ({s})", "seed": s, "is_playin": False, "playin_partner": ""}
                )
                ordered.append(team)
        raw_bracket[region] = ordered

    return enrich_bracket(raw_bracket, torvik_df, hasla_df, load_torvik_team_map())
