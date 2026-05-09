"""Tests for general-purpose helpers in dashboard_helpers.py."""

from __future__ import annotations

import math

import pandas as pd

from dashboard_helpers import pilot_stake_units


def test_below_cap_returns_unchanged():
    assert pilot_stake_units(0.25) == 0.25


def test_at_cap_returns_cap():
    assert pilot_stake_units(0.5) == 0.5


def test_above_cap_returns_cap():
    assert pilot_stake_units(1.0) == 0.5


def test_nan_returns_zero():
    assert pilot_stake_units(float("nan")) == 0.0


def test_none_returns_zero():
    assert pilot_stake_units(None) == 0.0


def test_string_input_returns_zero():
    assert pilot_stake_units("not-a-number") == 0.0


def test_string_numeric_input_parsed():
    assert pilot_stake_units("0.3") == 0.3


def test_negative_returns_zero():
    assert pilot_stake_units(-0.1) == 0.0


def test_zero_returns_zero():
    assert pilot_stake_units(0.0) == 0.0


def test_custom_cap_respected():
    assert pilot_stake_units(0.5, cap=0.25) == 0.25


def test_custom_cap_below_value():
    assert pilot_stake_units(0.1, cap=1.0) == 0.1


def test_pandas_na_returns_zero():
    """Direct pd.NA must not raise; should collapse to zero like NaN."""
    assert pilot_stake_units(pd.NA) == 0.0
