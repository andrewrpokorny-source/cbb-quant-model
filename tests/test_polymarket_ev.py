"""Tests for Polymarket fee model in betting.ev_calculator."""

import pytest

from betting.ev_calculator import (
    POLYMARKET_TAKER_FEE_COEFF,
    polymarket_fee_cents,
    polymarket_implied_prob,
    analyze_polymarket_bet,
    kalshi_fee_cents,
    kalshi_implied_prob,
    EdgeRating,
)


class TestPolymarketFeeCoefficient:
    """The sports fee coefficient should be 3%."""

    def test_coefficient_is_three_percent(self):
        assert POLYMARKET_TAKER_FEE_COEFF == 0.03


class TestPolymarketFeeCents:
    """Fee formula: 0.03 * P * (1-P) * 100."""

    def test_fee_at_50c_peak(self):
        fee = polymarket_fee_cents(50.0)
        assert abs(fee - 0.75) < 0.001

    def test_fee_at_30c(self):
        fee = polymarket_fee_cents(30.0)
        expected = 0.03 * 0.3 * 0.7 * 100
        assert abs(fee - expected) < 0.001

    def test_fee_at_0c_is_zero(self):
        assert polymarket_fee_cents(0.0) == 0.0

    def test_fee_at_100c_is_zero(self):
        assert polymarket_fee_cents(100.0) == 0.0

    def test_fee_symmetric(self):
        assert abs(polymarket_fee_cents(30.0) - polymarket_fee_cents(70.0)) < 0.001


class TestPolymarketImpliedProb:
    """Implied prob = price/100 + fee adjustment."""

    def test_at_50c(self):
        prob = polymarket_implied_prob(50.0)
        # 0.50 + 0.03 * 0.5 * 0.5 = 0.5075
        assert abs(prob - 0.5075) < 0.0001

    def test_at_70c(self):
        prob = polymarket_implied_prob(70.0)
        expected = 0.70 + 0.03 * 0.7 * 0.3
        assert abs(prob - expected) < 0.0001

    def test_cheaper_than_kalshi(self):
        """Polymarket 3% fee should always produce lower implied prob than Kalshi 7%."""
        for price in [20, 30, 40, 50, 60, 70, 80]:
            poly = polymarket_implied_prob(float(price))
            kalshi = kalshi_implied_prob(float(price))
            assert poly < kalshi, f"At {price}c: Poly {poly:.4f} >= Kalshi {kalshi:.4f}"

    def test_more_edge_than_kalshi(self):
        """Same model prob should yield more edge on Polymarket due to lower fees."""
        model_prob = 0.60
        poly_edge = model_prob - polymarket_implied_prob(50.0)
        kalshi_edge = model_prob - kalshi_implied_prob(50.0)
        assert poly_edge > kalshi_edge


class TestAnalyzePolymarketBet:
    """Full bet analysis with fee-adjusted pricing."""

    def test_strong_edge_returns_strong(self):
        result = analyze_polymarket_bet(model_prob=0.65, poly_yes_price=50.0)
        assert result["rating"] == EdgeRating.STRONG

    def test_no_edge_returns_pass(self):
        result = analyze_polymarket_bet(model_prob=0.50, poly_yes_price=50.0)
        assert result["rating"] == EdgeRating.PASS

    def test_edge_is_positive_when_model_exceeds_implied(self):
        result = analyze_polymarket_bet(model_prob=0.60, poly_yes_price=50.0)
        assert result["edge"] > 0
        assert result["edge_pct"] > 0

    def test_ev_positive_for_value_bet(self):
        result = analyze_polymarket_bet(model_prob=0.65, poly_yes_price=50.0)
        assert result["ev"] > 0

    def test_returns_all_expected_keys(self):
        result = analyze_polymarket_bet(model_prob=0.55, poly_yes_price=45.0)
        for key in ("edge", "edge_pct", "rating", "implied_prob", "model_prob", "ev"):
            assert key in result
