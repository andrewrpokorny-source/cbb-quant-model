"""Tests for MLB weather integration."""

import pandas as pd
import pytest

from mlb.weather import INDOOR_DEFAULTS, add_weather_features, fetch_game_weather
from mlb.ballpark_factors import STADIUM_COORDINATES, INDOOR_STADIUMS


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
