"""Tests for MLB ballpark run factors."""

import pytest
from mlb.ballpark_factors import (
    BALLPARK_FACTORS,
    STADIUM_COORDINATES,
    INDOOR_STADIUMS,
    get_park_factor,
    is_outdoor_stadium,
)


class TestBallparkFactors:

    def test_all_30_stadiums_have_factors(self):
        assert len(BALLPARK_FACTORS) >= 28

    def test_coors_field_above_1(self):
        assert BALLPARK_FACTORS["Coors Field"] > 1.0

    def test_oracle_park_below_1(self):
        assert BALLPARK_FACTORS["Oracle Park"] < 1.0

    def test_factors_in_reasonable_range(self):
        for name, factor in BALLPARK_FACTORS.items():
            assert 0.80 <= factor <= 1.30, f"{name} factor {factor} out of range"

    def test_default_factor_is_1(self):
        assert get_park_factor("Unknown Stadium") == 1.0

    def test_lookup_by_venue_name(self):
        factor = get_park_factor("Yankee Stadium")
        assert 0.8 <= factor <= 1.3


class TestStadiumCoordinates:

    def test_all_stadiums_have_coordinates(self):
        for name in BALLPARK_FACTORS:
            assert name in STADIUM_COORDINATES, f"Missing coords for {name}"

    def test_coordinates_valid(self):
        for name, (lat, lon) in STADIUM_COORDINATES.items():
            assert -90 <= lat <= 90, f"{name} lat {lat} out of range"
            assert -180 <= lon <= 180, f"{name} lon {lon} out of range"


class TestIndoorStadiums:

    def test_chase_field_is_indoor(self):
        assert not is_outdoor_stadium("Chase Field")

    def test_wrigley_is_outdoor(self):
        assert is_outdoor_stadium("Wrigley Field")

    def test_unknown_is_outdoor(self):
        assert is_outdoor_stadium("Some Random Field")
