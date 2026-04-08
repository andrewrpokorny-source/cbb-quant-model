"""Venue and distance utilities for neutral-site detection."""

import json
import os
import time
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
# Tracked geocode file ships with the repo; .cache/geocode.json holds runtime additions
GEOCODE_TRACKED = os.path.join(BASE_DIR, "venue_geocode.json")
TEAM_LOCATIONS_FILE = os.path.join(BASE_DIR, "data/team_locations.csv")

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
    """Great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 3959 * asin(sqrt(a))


def load_geocode_cache():
    """Load geocode results: tracked file first, then .cache/ overlay."""
    result = {}
    for path in (GEOCODE_TRACKED, os.path.join(CACHE_DIR, "geocode.json")):
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            for k, v in raw.items():
                if v and v[0] is not None:
                    result[k] = tuple(v)
    return result


def save_geocode_cache(cache):
    """Persist runtime geocode additions to .cache/ (tracked file is updated manually)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "geocode.json")
    with open(path, "w") as f:
        json.dump({k: list(v) for k, v in cache.items()}, f)


def geocode_location(loc, cache):
    """Geocode a 'city, STATE' string. Updates cache in place. Returns (lat, lon) or None."""
    if loc in cache:
        return cache[loc]

    try:
        from geopy.geocoders import Nominatim
        geo = Nominatim(user_agent="cbb-quant-model")
        result = geo.geocode(loc + ", USA", timeout=10)
        if result:
            cache[loc] = (result.latitude, result.longitude)
            time.sleep(1.05)
            return cache[loc]
    except ImportError:
        pass  # geopy not installed; fall through to state centroid
    except Exception as e:
        print(f"      WARNING: Geocoding failed for '{loc}' ({type(e).__name__}: {e}), falling back to state centroid")

    # Fallback to state centroid
    st = loc.split(",")[-1].strip()
    coords = STATE_CENTROIDS.get(st)
    if coords:
        cache[loc] = coords
    return coords


def load_team_locations(league=None):
    """Load tracked canonical team locations keyed by team name."""
    if not os.path.exists(TEAM_LOCATIONS_FILE):
        return {}

    df = pd.read_csv(TEAM_LOCATIONS_FILE)
    if league is not None and "league" in df.columns:
        df = df[df["league"] == league]

    if df.empty:
        return {}

    return dict(zip(df["team"], df["venue_loc"]))


def infer_team_home_locations(df):
    """Infer each team's home city/state from non-neutral home games in the data.

    Expects columns: team, is_home, is_neutral (or neutral_site), venue_city, venue_state.
    Returns dict: team_name -> "city, STATE".
    """
    neutral_col = "is_neutral" if "is_neutral" in df.columns else "neutral_site"
    mask = (df["is_home"] == 1) & (df.get(neutral_col, 0) == 0)
    home_games = df.loc[mask].copy()

    if "venue_city" not in home_games.columns or home_games["venue_city"].isna().all():
        return {}

    home_games = home_games.dropna(subset=["venue_city", "venue_state"])
    home_games["venue_loc"] = home_games["venue_city"] + ", " + home_games["venue_state"]

    counts = home_games.groupby(["team", "venue_loc"]).size().reset_index(name="n")
    idx = counts.groupby("team")["n"].idxmax()
    best = counts.loc[idx]

    return dict(zip(best["team"], best["venue_loc"]))


def build_team_home_locations(df, league=None):
    """Return team home locations, preferring tracked canonical data over inference."""
    homes = load_team_locations(league=league)
    inferred = infer_team_home_locations(df)
    for team, venue_loc in inferred.items():
        homes.setdefault(team, venue_loc)
    return homes


def compute_distance_advantage(team_home_loc, opp_home_loc, venue_city, venue_state, geo_cache):
    """Compute log distance advantage for a single game.

    Returns float: positive means opponent travels farther (advantage for team).
    Returns 0.0 if any location is missing.
    """
    if not venue_city or not venue_state:
        return 0.0

    venue_loc = f"{venue_city}, {venue_state}"
    vc = geo_cache.get(venue_loc) or geocode_location(venue_loc, geo_cache)
    tc = (geo_cache.get(team_home_loc) or geocode_location(team_home_loc, geo_cache)) if team_home_loc else None
    oc = (geo_cache.get(opp_home_loc) or geocode_location(opp_home_loc, geo_cache)) if opp_home_loc else None

    if not vc or not tc or not oc:
        return 0.0

    td = haversine(tc[0], tc[1], vc[0], vc[1])
    od = haversine(oc[0], oc[1], vc[0], vc[1])
    return float(np.log1p(od) - np.log1p(td))


def compute_distance_advantage_bulk(df, team_homes, geo_cache):
    """Add distance_advantage column to a DataFrame.

    Expects columns: team, opponent, venue_city, venue_state.
    """
    team_locs = df["team"].map(team_homes)
    opp_locs = df["opponent"].map(team_homes)

    results = []
    for i in range(len(df)):
        tloc = team_locs.iloc[i] if pd.notna(team_locs.iloc[i]) else None
        oloc = opp_locs.iloc[i] if pd.notna(opp_locs.iloc[i]) else None
        vcity = df["venue_city"].iloc[i] if pd.notna(df["venue_city"].iloc[i]) else ""
        vstate = df["venue_state"].iloc[i] if pd.notna(df["venue_state"].iloc[i]) else ""
        results.append(compute_distance_advantage(tloc, oloc, vcity, vstate, geo_cache))

    df["distance_advantage"] = results
    return df
