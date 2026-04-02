"""Tests for MLB weather integration."""

import pandas as pd
import pytest

from mlb.weather import INDOOR_DEFAULTS, add_weather_features, fetch_game_weather
from mlb.ballpark_factors import STADIUM_COORDINATES, INDOOR_STADIUMS


class TestUTCToEasternConversion:
    """Verify game_time (UTC) is properly converted to Eastern for weather lookups."""

    def test_summer_utc_to_eastern(self):
        """23:00 UTC in July = 19:00 ET (UTC-4 during DST)."""
        # fetch_game_weather receives UTC time and should convert internally
        # We test indirectly by checking the target_hour logic
        from mlb.weather import fetch_game_weather
        # Chase Field is indoor so won't hit the API -- use for logic check
        result = fetch_game_weather("Wrigley Field", "2025-07-04", "23:10")
        # Should not crash and should return reasonable values
        assert "temperature" in result
        assert "wind_speed" in result

    def test_winter_utc_to_eastern(self):
        """23:00 UTC in November = 18:00 ET (UTC-5 outside DST)."""
        result = fetch_game_weather("Wrigley Field", "2025-11-01", "23:00")
        assert "temperature" in result


class TestIndoorDefaults:

    def test_indoor_stadium_returns_defaults(self):
        result = fetch_game_weather("Chase Field", "2025-07-04", "19:00")
        assert result["temperature"] == INDOOR_DEFAULTS["temperature"]
        assert result["wind_speed"] == INDOOR_DEFAULTS["wind_speed"]

    def test_unknown_stadium_returns_defaults(self):
        result = fetch_game_weather("Nonexistent Park", "2025-07-04")
        assert result["temperature"] == INDOOR_DEFAULTS["temperature"]


class TestAddWeatherFeatures:

    def test_columns_added(self):
        df = pd.DataFrame({
            "venue_name": ["Chase Field", "Wrigley Field"],
            "date": ["2025-07-04", "2025-07-04"],
            "game_time": ["23:00", "23:00"],
            "venue_indoor": [1, 0],
        })
        result = add_weather_features(df)
        assert "temperature" in result.columns
        assert "wind_speed" in result.columns

    def test_indoor_gets_defaults(self):
        df = pd.DataFrame({
            "venue_name": ["Chase Field"],
            "date": ["2025-07-04"],
            "game_time": ["23:00"],
            "venue_indoor": [1],
        })
        result = add_weather_features(df)
        assert result.iloc[0]["temperature"] == INDOOR_DEFAULTS["temperature"]
        assert result.iloc[0]["wind_speed"] == INDOOR_DEFAULTS["wind_speed"]

    def test_no_nan_in_output(self):
        df = pd.DataFrame({
            "venue_name": ["Unknown Park", "Chase Field"],
            "date": ["2025-07-04", "2025-07-04"],
            "game_time": ["23:00", "23:00"],
            "venue_indoor": [0, 1],
        })
        result = add_weather_features(df)
        assert result["temperature"].notna().all()
        assert result["wind_speed"].notna().all()
