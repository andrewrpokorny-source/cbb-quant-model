"""
Neutral Site Experiment: compare 4 approaches for handling neutral-site games.

Approaches:
  0. Baseline         -- current production features (is_home binary)
  1. +is_neutral      -- add binary neutral flag alongside is_home
  2. distance_adv     -- replace is_home with log travel-distance differential
  3. both             -- keep is_home, add is_neutral + distance_advantage

Workflow:
  1. Load processed training data
  2. Re-fetch ESPN for each game date to get neutralSite + venue address (cached)
  3. Infer each team's home venue from non-neutral home games
  4. Geocode unique city/state locations via Nominatim (cached)
  5. Compute distance_advantage = log1p(opp_dist) - log1p(team_dist)
  6. Walk-forward backtest all 4 feature sets
  7. Print comparison table
"""

import argparse
import json
import os
import time
from datetime import timedelta
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import requests
from geopy.geocoders import Nominatim
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from league_config import get_league_artifact_paths, get_scoreboard_base_url, normalize_league
from model import FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
HIGH_CONF_THRESHOLD = 0.53


# ---------------------------------------------------------------------------
# Team name normalization -- CSV uses short names, ESPN uses displayName
# ---------------------------------------------------------------------------

def _norm_key(name):
    """Reduce a name to a canonical form for matching."""
    s = name.lower().strip()
    s = s.replace("st.", "state").replace("&", "and").replace("-", " ").replace("'", "")
    # Remove mascot names by dropping the last word if >1 word and it's likely a mascot
    # (ESPN displayName = "Alabama Crimson Tide", location = "Alabama")
    return " ".join(s.split())


def build_team_name_map(league="mens"):
    """Fetch ESPN teams, build normalized lookup: any_name -> espn displayName."""
    from league_config import get_league_settings
    sport_path = get_league_settings(league)["sport_path"]

    cache = os.path.join(CACHE_DIR, f"team_names_{league}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    url = ("http://site.api.espn.com/apis/site/v2/sports/basketball/"
           f"{sport_path}/teams?limit=500")
    res = requests.get(url, timeout=10).json()
    teams = res["sports"][0]["leagues"][0]["teams"]

    # exact_map: various exact strings -> displayName
    # norm_map:  normalized key -> displayName (for fuzzy fallback)
    exact_map = {}
    norm_map = {}
    for t in teams:
        tm = t["team"]
        dn = tm["displayName"]
        loc = tm.get("location", "")

        # Register exact matches
        for v in (dn, loc, tm.get("shortDisplayName", "")):
            if v:
                exact_map[v] = dn

        # Register normalized matches (location is most useful for old CSV names)
        for v in (dn, loc, tm.get("shortDisplayName", "")):
            if v:
                norm_map[_norm_key(v)] = dn

    mapping = {"exact": exact_map, "norm": norm_map}
    with open(cache, "w") as f:
        json.dump(mapping, f)
    return mapping


def resolve_team_name(csv_name, name_map, _cache={}):
    """Map a CSV team name to its ESPN displayName."""
    if csv_name in _cache:
        return _cache[csv_name]

    # 1. Exact match
    result = name_map["exact"].get(csv_name)
    if result:
        _cache[csv_name] = result
        return result

    # 2. Normalized match
    result = name_map["norm"].get(_norm_key(csv_name))
    if result:
        _cache[csv_name] = result
        return result

    # 3. Give up
    _cache[csv_name] = csv_name
    return csv_name

# Fallback state centroids (lat, lon) when geocoding fails
STATE_CENTROIDS = {
    "AL": (32.8, -86.8), "AK": (64.2, -152.5), "AZ": (34.0, -111.1),
    "AR": (35.2, -91.8), "CA": (36.8, -119.4), "CO": (39.1, -105.4),
    "CT": (41.6, -72.7), "DC": (38.9, -77.0), "DE": (39.0, -75.5),
    "FL": (27.8, -81.7), "GA": (33.0, -83.5), "HI": (19.9, -155.6),
    "ID": (44.2, -114.4), "IL": (40.3, -89.0), "IN": (40.3, -86.1),
    "IA": (42.0, -93.2), "KS": (38.5, -98.0), "KY": (37.7, -84.7),
    "LA": (31.2, -92.1), "ME": (45.4, -69.4), "MD": (39.0, -76.6),
    "MA": (42.4, -71.4), "MI": (43.3, -84.5), "MN": (46.7, -94.7),
    "MS": (32.7, -89.7), "MO": (38.5, -92.3), "MT": (46.8, -110.4),
    "NE": (41.1, -98.3), "NV": (38.8, -116.4), "NH": (43.5, -71.6),
    "NJ": (40.1, -74.5), "NM": (34.8, -106.2), "NY": (43.0, -75.0),
    "NC": (35.6, -79.8), "ND": (47.5, -100.5), "OH": (40.4, -82.8),
    "OK": (35.6, -96.9), "OR": (44.6, -120.5), "PA": (40.6, -77.2),
    "RI": (41.6, -71.5), "SC": (33.9, -81.2), "SD": (43.7, -99.9),
    "TN": (35.7, -86.6), "TX": (31.1, -97.6), "UT": (39.3, -111.1),
    "VT": (44.6, -72.6), "VA": (37.8, -78.2), "WA": (47.4, -120.7),
    "WV": (38.5, -80.5), "WI": (44.3, -89.5), "WY": (43.1, -107.6),
}


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 3959 * asin(sqrt(a))


# ---------------------------------------------------------------------------
# Step 2 -- Fetch neutral-site + venue data from ESPN (one request per date)
# ---------------------------------------------------------------------------

def fetch_neutral_site_data(dates, base_url, cache_file):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cached = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)

    needed = [d for d in dates if d not in cached]
    if not needed:
        print(f"   All {len(cached)} dates already cached")
        return cached

    print(f"   Fetching {len(needed)} dates from ESPN ({len(cached)} cached)...")
    for i, date_str in enumerate(needed):
        url = f"{base_url}&dates={date_str.replace('-', '')}"
        try:
            res = requests.get(url, timeout=10).json()
            day = []
            for event in res.get("events", []):
                comp = event["competitions"][0]
                home = comp["competitors"][0]
                away = comp["competitors"][1]
                venue = comp.get("venue", {})
                addr = venue.get("address", {})
                day.append({
                    "home": home["team"]["displayName"],
                    "away": away["team"]["displayName"],
                    "neutral": comp.get("neutralSite", False),
                    "v_city": addr.get("city", ""),
                    "v_state": addr.get("state", ""),
                })
            cached[date_str] = day
        except Exception as e:
            print(f"      warn: {date_str} failed ({e})")
            cached[date_str] = []
        if (i + 1) % 30 == 0:
            print(f"      {i + 1}/{len(needed)} done")
        time.sleep(0.25)

    with open(cache_file, "w") as f:
        json.dump(cached, f)
    print(f"   Cached {len(cached)} dates total")
    return cached


# ---------------------------------------------------------------------------
# Step 3 -- Enrich training rows with is_neutral + venue city/state
# ---------------------------------------------------------------------------

def enrich_data(df, espn, name_map):
    lookup = {}
    for date_str, games in espn.items():
        for g in games:
            info = {
                "is_neutral": int(g["neutral"]),
                "v_city": g["v_city"],
                "v_state": g["v_state"],
            }
            # Index by both original ESPN name and normalized form
            for team_name in (g["home"], g["away"]):
                lookup[(date_str, team_name)] = info

    date_strs = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # Resolve CSV names to ESPN displayNames for matching
    resolved = df["team"].map(lambda n: resolve_team_name(n, name_map))
    keys = list(zip(date_strs, resolved))

    is_neutral = []
    v_city = []
    v_state = []
    for k in keys:
        info = lookup.get(k)
        if info:
            is_neutral.append(info["is_neutral"])
            v_city.append(info["v_city"])
            v_state.append(info["v_state"])
        else:
            is_neutral.append(0)
            v_city.append("")
            v_state.append("")

    df["is_neutral"] = is_neutral
    df["venue_city"] = v_city
    df["venue_state"] = v_state

    matched = sum(1 for c in v_city if c)
    print(f"   Matched {matched}/{len(df)} rows with venue info ({matched / len(df):.0%})")
    neutral_rows = sum(is_neutral)
    print(f"   Neutral-site rows: {neutral_rows} ({neutral_rows / len(df):.1%})")
    return df


# ---------------------------------------------------------------------------
# Step 4 -- Infer team home locations from non-neutral home games
# ---------------------------------------------------------------------------

def build_team_homes(espn):
    counts = {}  # team -> { "city, ST": count }
    for games in espn.values():
        for g in games:
            if g["neutral"] or not g["v_city"]:
                continue
            team = g["home"]
            loc = f"{g['v_city']}, {g['v_state']}"
            counts.setdefault(team, {})
            counts[team][loc] = counts[team].get(loc, 0) + 1

    homes = {}
    for team, locs in counts.items():
        homes[team] = max(locs, key=locs.get)
    return homes


# ---------------------------------------------------------------------------
# Step 5 -- Geocode unique city/state locations (Nominatim, cached)
# ---------------------------------------------------------------------------

def geocode_locations(locations, cache_file):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cached = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)

    needed = [loc for loc in locations if loc and loc not in cached]
    if needed:
        print(f"   Geocoding {len(needed)} locations ({len(cached)} cached)...")
        geolocator = Nominatim(user_agent="cbb-quant-neutral-site-experiment")
        for loc in needed:
            try:
                result = geolocator.geocode(loc + ", USA", timeout=10)
                if result:
                    cached[loc] = [result.latitude, result.longitude]
                else:
                    # Fallback to state centroid
                    st = loc.split(",")[-1].strip()
                    cached[loc] = list(STATE_CENTROIDS.get(st, [None, None]))
            except Exception:
                st = loc.split(",")[-1].strip()
                cached[loc] = list(STATE_CENTROIDS.get(st, [None, None]))
            time.sleep(1.05)

        with open(cache_file, "w") as f:
            json.dump(cached, f)

    geo = {}
    for loc, coords in cached.items():
        if coords and coords[0] is not None:
            geo[loc] = (coords[0], coords[1])
    print(f"   {len(geo)} locations geocoded successfully")
    return geo


# ---------------------------------------------------------------------------
# Step 6 -- Compute distance_advantage feature
# ---------------------------------------------------------------------------

def compute_distance_features(df, team_homes, geo, name_map):
    resolved_team = df["team"].map(lambda n: resolve_team_name(n, name_map))
    resolved_opp = df["opponent"].map(lambda n: resolve_team_name(n, name_map))
    team_home_loc = resolved_team.map(team_homes)
    opp_home_loc = resolved_opp.map(team_homes)
    venue_loc = df["venue_city"] + ", " + df["venue_state"]

    team_dist = []
    opp_dist = []
    for i in range(len(df)):
        vloc = venue_loc.iloc[i]
        tloc = team_home_loc.iloc[i] if pd.notna(team_home_loc.iloc[i]) else ""
        oloc = opp_home_loc.iloc[i] if pd.notna(opp_home_loc.iloc[i]) else ""

        vc = geo.get(vloc)
        tc = geo.get(tloc)
        oc = geo.get(oloc)

        td = haversine(tc[0], tc[1], vc[0], vc[1]) if (tc and vc) else np.nan
        od = haversine(oc[0], oc[1], vc[0], vc[1]) if (oc and vc) else np.nan
        team_dist.append(td)
        opp_dist.append(od)

    df["distance_advantage"] = np.log1p(opp_dist) - np.log1p(team_dist)
    df["distance_advantage"] = df["distance_advantage"].fillna(0)

    valid = (df["distance_advantage"] != 0).sum()
    print(f"   Distance advantage computed for {valid}/{len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Walk-forward backtest engine
# ---------------------------------------------------------------------------

def build_pipeline():
    base = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def run_backtest(df, features, label, weeks_back):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["spread_abs"] = df["spread"].abs()
    df["spread_squared"] = df["spread"] ** 2
    df["last_game"] = df.groupby("team")["date"].shift(1)
    df["rest_days"] = (df["date"] - df["last_game"]).dt.days.fillna(7).clip(upper=7)

    end_date = df["date"].max() + timedelta(days=1)
    start_date = end_date - timedelta(weeks=weeks_back)

    pipeline = build_pipeline()
    current = start_date
    logs = []

    while current < end_date:
        nxt = current + timedelta(days=7)

        past = df[df["date"] < current]
        valid = [f for f in features if f in past.columns]
        if len(valid) != len(features):
            current = nxt
            continue

        train = past.dropna(subset=valid + ["ats_win"])
        if len(train) < 50:
            current = nxt
            continue

        clf = clone(pipeline)
        clf.fit(train[valid].astype(float), train["ats_win"].astype(int))

        mask = (df["date"] >= current) & (df["date"] < nxt) & (df["is_home"] == 1)
        week = df[mask].dropna(subset=valid).copy()

        if len(week) > 0:
            probs = clf.predict_proba(week[valid].astype(float))[:, 1]
            week["prob_home"] = probs
            week["conf"] = week["prob_home"].apply(lambda x: max(x, 1 - x))
            week["pick_correct"] = np.where(
                week["prob_home"] > 0.5,
                week["ats_win"] == 1,
                week["ats_win"] == 0,
            )
            logs.append(
                week[["date", "team", "opponent", "conf", "pick_correct",
                       "prob_home", "ats_win", "is_neutral"]]
            )
        current = nxt

    if not logs:
        return {"label": label, "accuracy": 0, "brier": 1,
                "hc_acc": 0, "hc_bets": 0, "total": 0, "roi": 0,
                "neutral_acc": 0, "neutral_n": 0, "home_acc": 0, "home_n": 0}

    full = pd.concat(logs)
    total = len(full)
    acc = full["pick_correct"].sum() / total
    brier = brier_score_loss(full["ats_win"].astype(int), full["prob_home"].astype(float))

    hc = full[full["conf"] >= HIGH_CONF_THRESHOLD]
    hc_bets = len(hc)
    hc_acc = hc["pick_correct"].sum() / hc_bets if hc_bets > 0 else 0
    payout = 100 / 110
    roi = (hc["pick_correct"].sum() * payout) - ((hc_bets - hc["pick_correct"].sum()))

    # Breakdown: neutral vs non-neutral
    neu = full[full["is_neutral"] == 1]
    home = full[full["is_neutral"] == 0]
    neu_acc = neu["pick_correct"].mean() if len(neu) > 0 else 0
    home_acc = home["pick_correct"].mean() if len(home) > 0 else 0

    return {
        "label": label,
        "accuracy": acc,
        "brier": brier,
        "hc_acc": hc_acc,
        "hc_bets": hc_bets,
        "total": total,
        "roi": roi,
        "neutral_acc": neu_acc,
        "neutral_n": len(neu),
        "home_acc": home_acc,
        "home_n": len(home),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="mens")
    parser.add_argument("--weeks", type=int, default=16,
                        help="Backtest window in weeks (default 16 to capture early-season tournaments)")
    args = parser.parse_args()

    league = normalize_league(args.league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    base_url = get_scoreboard_base_url(league)
    weeks = args.weeks

    print("=" * 70)
    print("  NEUTRAL SITE EXPERIMENT -- 4-way walk-forward backtest")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading training data...")
    df = pd.read_csv(data_file, low_memory=False)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    print(f"   {len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")

    # 2. Fetch ESPN neutral-site data
    print("\n[2/6] Fetching neutral-site data from ESPN...")
    dates = sorted(df["date"].dt.strftime("%Y-%m-%d").unique())
    cache = os.path.join(CACHE_DIR, f"neutral_{league}.json")
    espn = fetch_neutral_site_data(dates, base_url, cache)

    # 3. Build team name mapping (short CSV names <-> ESPN displayNames)
    print("\n[3/7] Building team name mapping...")
    name_map = build_team_name_map(league)
    print(f"   {len(name_map['exact'])} exact + {len(name_map['norm'])} normalized entries")

    # 4. Enrich
    print("\n[4/7] Enriching rows...")
    df = enrich_data(df, espn, name_map)

    # 5. Team home locations
    print("\n[5/7] Inferring team home locations...")
    team_homes = build_team_homes(espn)
    print(f"   {len(team_homes)} teams with known home venues")

    # 6. Geocode
    print("\n[6/7] Geocoding venues and team homes...")
    all_locs = set()
    for games in espn.values():
        for g in games:
            if g["v_city"]:
                all_locs.add(f"{g['v_city']}, {g['v_state']}")
    all_locs.update(team_homes.values())
    all_locs.discard("")

    geo_cache = os.path.join(CACHE_DIR, "geocode.json")
    geo = geocode_locations(list(all_locs), geo_cache)

    df = compute_distance_features(df, team_homes, geo, name_map)

    # 7. Backtests
    print(f"\n[7/7] Running backtests ({weeks}-week window)...")
    print()

    baseline = list(FEATURES)
    a1 = list(FEATURES) + ["is_neutral"]
    a2 = [f for f in FEATURES if f != "is_home"] + ["distance_advantage"]
    a3 = list(FEATURES) + ["is_neutral", "distance_advantage"]

    configs = [
        (baseline, "Baseline (is_home only)"),
        (a1, "1: +is_neutral"),
        (a2, "2: distance_advantage (no is_home)"),
        (a3, "3: +is_neutral +distance_advantage"),
    ]

    results = []
    for feats, label in configs:
        print(f"   {label}...")
        r = run_backtest(df, feats, label, weeks)
        results.append(r)
        print(f"      Acc {r['accuracy']:.1%} | Brier {r['brier']:.4f} | "
              f"HC {r['hc_acc']:.1%} ({r['hc_bets']}) | ROI {r['roi']:+.2f}U")

    # Summary
    print()
    print("=" * 90)
    hdr = f"{'Approach':<42} {'Acc':>5} {'Brier':>7} {'HC Acc':>7} {'HC#':>4} {'ROI':>7} | {'Neu':>5} {'n':>4} {'Home':>5} {'n':>5}"
    print(hdr)
    print("-" * 90)
    for r in results:
        print(
            f"{r['label']:<42} {r['accuracy']:>5.1%} {r['brier']:>7.4f} "
            f"{r['hc_acc']:>6.1%} {r['hc_bets']:>4} {r['roi']:>+7.2f}U | "
            f"{r['neutral_acc']:>5.1%} {r['neutral_n']:>4} "
            f"{r['home_acc']:>5.1%} {r['home_n']:>5}"
        )
    print("=" * 90)
    print(f"\nNeu = accuracy on neutral-site games | Home = accuracy on home-court games")


if __name__ == "__main__":
    main()
