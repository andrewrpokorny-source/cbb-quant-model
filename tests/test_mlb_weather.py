"""Tests for MLB weather integration."""

import pandas as pd
import pytest

from mlb.weather import INDOOR_DEFAULTS, add_weather_features, fetch_game_weather
from mlb.ballpark_factors import STADIUM_COORDINATES, INDOOR_STADIUMS


class TestUTCToEasternConversion:
    """Verify game_time (UTC) is properly converted to Eastern for weather lookups."""

    def test_summer_dst_conversion(self):
        """23:00 UTC on July 4 = 19:00 EDT (UTC-4)."""
        from mlb.weather import _utc_hour_to_eastern
        assert _utc_hour_to_eastern("2025-07-04", 23) == 19

    def test_winter_est_conversion(self):
        """23:00 UTC on January 15 = 18:00 EST (UTC-5)."""
        from mlb.weather import _utc_hour_to_eastern
        assert _utc_hour_to_eastern("2025-01-15", 23) == 18

    def test_march_before_dst_starts(self):
        """March 1, 2025 is still EST (DST starts March 9). 23:00 UTC = 18:00 EST."""
        from mlb.weather import _utc_hour_to_eastern
        assert _utc_hour_to_eastern("2025-03-01", 23) == 18

    def test_march_after_dst_starts(self):
        """March 10, 2025 is EDT (DST started March 9). 23:00 UTC = 19:00 EDT."""
        from mlb.weather import _utc_hour_to_eastern
        assert _utc_hour_to_eastern("2025-03-10", 23) == 19

    def test_november_before_dst_ends(self):
        """November 1, 2025 is still EDT (DST ends November 2). 23:00 UTC = 19:00 EDT."""
        from mlb.weather import _utc_hour_to_eastern
        assert _utc_hour_to_eastern("2025-11-01", 23) == 19

    def test_november_after_dst_ends(self):
        """November 3, 2025 is EST (DST ended November 2). 23:00 UTC = 18:00 EST."""
        from mlb.weather import _utc_hour_to_eastern
        assert _utc_hour_to_eastern("2025-11-03", 23) == 18


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
