"""Tests for betting/kelly.py: Kelly sizing and unit calibration."""

import pytest

from betting.kelly import kelly_fraction, recommended_units


# Standard -110 implied probability
STD_IP = 0.5238


class TestKellyFraction:
    """Tests for kelly_fraction."""

    def test_no_edge_returns_zero(self):
        assert kelly_fraction(0.0, STD_IP) == 0.0

    def test_negative_edge_returns_zero(self):
        assert kelly_fraction(-0.05, STD_IP) == 0.0

    def test_positive_edge(self):
        result = kelly_fraction(0.08, STD_IP)
        # quarter Kelly: 0.25 * 0.08 / (1 - 0.5238) = 0.042
        assert result == pytest.approx(0.042, abs=0.001)

    def test_capped_at_ten_percent(self):
        """Large edge should be capped at 0.10."""
        result = kelly_fraction(0.40, STD_IP)
        assert result == pytest.approx(0.10)

    def test_invalid_implied_prob(self):
        assert kelly_fraction(0.08, 0.0) == 0.0
        assert kelly_fraction(0.08, 1.0) == 0.0


class TestRecommendedUnits:
    """Tests for recommended_units calibration."""

    def test_below_threshold_returns_zero(self):
        """Negative edge -> 0 units."""
        assert recommended_units(-0.01, STD_IP) == 0.0

    def test_eight_pct_edge(self):
        """8% edge (STRONG threshold) -> 1.5U."""
        units = recommended_units(0.08, STD_IP)
        assert units == 1.5

    def test_twelve_pct_edge(self):
        """12% edge -> 2.0U."""
        units = recommended_units(0.12, STD_IP)
        assert units == 2.0

    def test_fifteen_pct_edge(self):
        """15% edge -> 2.5U."""
        units = recommended_units(0.15, STD_IP)
        assert units == 2.5

    def test_max_units_reachable(self):
        """Very large edge should hit exactly 3.0U (not 2.5)."""
        units = recommended_units(0.30, STD_IP)
        assert units == 3.0

    def test_max_units_not_exceeded(self):
        """Even extreme edge should not exceed 3.0U."""
        units = recommended_units(0.50, STD_IP)
        assert units == 3.0

    def test_rounds_to_half_unit(self):
        """Units should be rounded to nearest 0.5."""
        units = recommended_units(0.08, STD_IP)
        assert units * 2 == int(units * 2)
