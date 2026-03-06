"""Tests for venue.py distance and geocoding utilities."""

import os

import numpy as np
import pandas as pd
import pytest

from venue import (
    compute_distance_advantage,
    compute_distance_advantage_bulk,
    build_team_home_locations,
    infer_team_home_locations,
    haversine,
    geocode_location,
    load_geocode_cache,
    load_team_locations,
    GEOCODE_TRACKED,
    TEAM_LOCATIONS_FILE,
    STATE_CENTROIDS,
)


# ---------------------------------------------------------------------------
# haversine
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_returns_zero(self):
        assert haversine(40.0, -74.0, 40.0, -74.0) == 0.0

    def test_known_distance(self):
        # NYC to LA ~2,451 miles
        d = haversine(40.7128, -74.0060, 34.0522, -118.2437)
        assert 2400 < d < 2500

    def test_symmetric(self):
        d1 = haversine(40.0, -74.0, 34.0, -118.0)
        d2 = haversine(34.0, -118.0, 40.0, -74.0)
        assert d1 == pytest.approx(d2)


# ---------------------------------------------------------------------------
# compute_distance_advantage
# ---------------------------------------------------------------------------

class TestComputeDistanceAdvantage:
    def test_returns_zero_when_venue_missing(self):
        cache = {}
        assert compute_distance_advantage("A, NY", "B, CA", "", "", cache) == 0.0
        assert compute_distance_advantage("A, NY", "B, CA", "City", "", cache) == 0.0

    def test_returns_zero_when_team_home_missing(self):
        cache = {"Vegas, NV": (36.17, -115.14)}
        assert compute_distance_advantage(None, "B, CA", "Vegas", "NV", cache) == 0.0
        assert compute_distance_advantage("A, NY", None, "Vegas", "NV", cache) == 0.0

    def test_positive_when_opponent_travels_more(self):
        # Team is near venue, opponent is far
        cache = {
            "Durham, NC": (35.99, -78.90),
            "Charlotte, NC": (35.23, -80.84),
            "Boston, MA": (42.36, -71.06),
        }
        adv = compute_distance_advantage(
            "Durham, NC", "Boston, MA", "Charlotte", "NC", cache
        )
        assert adv > 0, "Team closer to venue should have positive advantage"

    def test_negative_when_team_travels_more(self):
        cache = {
            "Durham, NC": (35.99, -78.90),
            "Charlotte, NC": (35.23, -80.84),
            "Boston, MA": (42.36, -71.06),
        }
        adv = compute_distance_advantage(
            "Boston, MA", "Durham, NC", "Charlotte", "NC", cache
        )
        assert adv < 0, "Team farther from venue should have negative advantage"

    def test_symmetric_magnitude(self):
        cache = {
            "Durham, NC": (35.99, -78.90),
            "Charlotte, NC": (35.23, -80.84),
            "Boston, MA": (42.36, -71.06),
        }
        adv1 = compute_distance_advantage(
            "Durham, NC", "Boston, MA", "Charlotte", "NC", cache
        )
        adv2 = compute_distance_advantage(
            "Boston, MA", "Durham, NC", "Charlotte", "NC", cache
        )
        assert adv1 == pytest.approx(-adv2)

    def test_geocodes_missing_locations_on_fresh_cache(self):
        """Repro for finding #1: fresh cache must geocode team homes, not just venue."""
        cache = {}
        # Use state-centroid-resolvable locations (geopy won't be called in CI,
        # but geocode_location falls back to STATE_CENTROIDS)
        adv = compute_distance_advantage(
            "Charlotte, NC", "Boston, MA", "Las Vegas", "NV", cache
        )
        # All three should have been geocoded via state centroid fallback
        assert "Las Vegas, NV" in cache
        assert "Charlotte, NC" in cache
        assert "Boston, MA" in cache
        assert adv != 0.0, "Should produce non-zero distance with state centroid fallback"


# ---------------------------------------------------------------------------
# compute_distance_advantage_bulk
# ---------------------------------------------------------------------------

class TestComputeDistanceAdvantageBulk:
    def test_adds_column(self):
        df = pd.DataFrame({
            "team": ["Duke Blue Devils", "Boston College Eagles"],
            "opponent": ["Boston College Eagles", "Duke Blue Devils"],
            "venue_city": ["Charlotte", "Charlotte"],
            "venue_state": ["NC", "NC"],
        })
        homes = {
            "Duke Blue Devils": "Durham, NC",
            "Boston College Eagles": "Boston, MA",
        }
        cache = {
            "Durham, NC": (35.99, -78.90),
            "Charlotte, NC": (35.23, -80.84),
            "Boston, MA": (42.36, -71.06),
        }
        result = compute_distance_advantage_bulk(df, homes, cache)
        assert "distance_advantage" in result.columns
        assert len(result) == 2
        # Home/away rows should mirror
        assert result["distance_advantage"].iloc[0] == pytest.approx(
            -result["distance_advantage"].iloc[1]
        )


# ---------------------------------------------------------------------------
# build_team_home_locations
# ---------------------------------------------------------------------------

class TestBuildTeamHomeLocations:
    def test_picks_most_common_venue(self):
        df = pd.DataFrame({
            "team": ["Duke"] * 5 + ["Duke"],
            "is_home": [1, 1, 1, 1, 1, 0],
            "is_neutral": [0, 0, 0, 0, 1, 0],
            "venue_city": ["Durham", "Durham", "Durham", "Raleigh", "Charlotte", "Boston"],
            "venue_state": ["NC", "NC", "NC", "NC", "NC", "MA"],
        })
        homes = build_team_home_locations(df)
        assert homes["Duke"] == "Durham, NC"

    def test_excludes_neutral_games(self):
        df = pd.DataFrame({
            "team": ["Duke"] * 3,
            "is_home": [1, 1, 1],
            "is_neutral": [1, 1, 0],
            "venue_city": ["Charlotte", "Charlotte", "Durham"],
            "venue_state": ["NC", "NC", "NC"],
        })
        homes = build_team_home_locations(df)
        assert homes["Duke"] == "Durham, NC"

    def test_empty_when_no_venue_data(self):
        df = pd.DataFrame({
            "team": ["Duke"],
            "is_home": [1],
            "is_neutral": [0],
            "venue_city": [None],
            "venue_state": [None],
        })
        homes = infer_team_home_locations(df)
        assert len(homes) == 0

    def test_infer_team_home_locations_matches_legacy_behavior(self):
        df = pd.DataFrame({
            "team": ["Duke"] * 4,
            "is_home": [1, 1, 1, 0],
            "is_neutral": [0, 0, 1, 0],
            "venue_city": ["Durham", "Durham", "Charlotte", "Boston"],
            "venue_state": ["NC", "NC", "NC", "MA"],
        })
        homes = infer_team_home_locations(df)
        assert homes["Duke"] == "Durham, NC"

    def test_prefers_canonical_team_location_and_fills_missing(self, monkeypatch, tmp_path):
        canonical = tmp_path / "team_locations.csv"
        canonical.write_text(
            "league,team,city,state,venue_loc,latitude,longitude,source\n"
            "mens,Duke,Durham,NC,\"Durham, NC\",35.99,-78.90,canonical\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("venue.TEAM_LOCATIONS_FILE", str(canonical))

        df = pd.DataFrame({
            "team": ["Duke", "UNC Greensboro"],
            "is_home": [1, 1],
            "is_neutral": [0, 0],
            "venue_city": ["Raleigh", "Greensboro"],
            "venue_state": ["NC", "NC"],
        })
        homes = build_team_home_locations(df, league="mens")
        assert homes["Duke"] == "Durham, NC"
        assert homes["UNC Greensboro"] == "Greensboro, NC"


class TestLoadTeamLocations:
    def test_loads_tracked_team_locations(self):
        assert os.path.exists(TEAM_LOCATIONS_FILE), "team_locations.csv must be tracked in repo"
        mens_homes = load_team_locations("mens")
        womens_homes = load_team_locations("womens")
        assert len(mens_homes) > 300
        assert len(womens_homes) > 300


# ---------------------------------------------------------------------------
# geocode_location fallback
# ---------------------------------------------------------------------------

class TestGeocodeLocation:
    def test_populates_cache_on_miss(self):
        cache = {}
        result = geocode_location("Durham, NC", cache)
        assert result is not None
        assert "Durham, NC" in cache
        assert len(result) == 2

    def test_cached_result_returned(self):
        cache = {"Durham, NC": (35.99, -78.90)}
        result = geocode_location("Durham, NC", cache)
        assert result == (35.99, -78.90)


class TestLoadGeocodeCache:
    def test_tracked_file_exists_and_loads(self):
        assert os.path.exists(GEOCODE_TRACKED), "venue_geocode.json must be tracked in repo"
        cache = load_geocode_cache()
        assert len(cache) > 100, f"Expected 300+ entries, got {len(cache)}"
        # Spot check a known location
        sample = next(iter(cache.values()))
        assert len(sample) == 2
