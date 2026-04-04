"""Tests for betting.ev_calculator edge rating logic."""

import pytest

from betting.ev_calculator import (
    EdgeRating,
    american_odds_to_implied_prob,
    get_rating,
    kalshi_fee_cents,
    kalshi_implied_prob,
    analyze_bet,
    VALUE_RATINGS,
    RATING_RANK,
    STRONG_THRESHOLD,
    GOOD_THRESHOLD,
    MARGINAL_THRESHOLD,
)


class TestGetRating:
    """Boundary tests for get_rating()."""

    def test_strong_at_threshold(self):
        assert get_rating(0.08) == EdgeRating.STRONG

    def test_strong_above_threshold(self):
        assert get_rating(0.12) == EdgeRating.STRONG

    def test_good_at_threshold(self):
        assert get_rating(0.04) == EdgeRating.GOOD

    def test_good_between_thresholds(self):
        assert get_rating(0.06) == EdgeRating.GOOD

    def test_good_just_below_strong(self):
        assert get_rating(0.0799) == EdgeRating.GOOD

    def test_marginal_at_threshold(self):
        assert get_rating(0.02) == EdgeRating.MARGINAL

    def test_marginal_between_thresholds(self):
        assert get_rating(0.03) == EdgeRating.MARGINAL

    def test_marginal_just_below_good(self):
        assert get_rating(0.0399) == EdgeRating.MARGINAL

    def test_pass_just_below_marginal(self):
        assert get_rating(0.0199) == EdgeRating.PASS

    def test_pass_zero_edge(self):
        assert get_rating(0.0) == EdgeRating.PASS

    def test_pass_negative_edge(self):
        assert get_rating(-0.05) == EdgeRating.PASS


class TestEdgeRatingValues:
    """Verify enum string values match what CSV consumers expect."""

    def test_strong_value(self):
        assert EdgeRating.STRONG.value == "STRONG"

    def test_good_value(self):
        assert EdgeRating.GOOD.value == "GOOD"

    def test_marginal_value(self):
        assert EdgeRating.MARGINAL.value == "MARGINAL"

    def test_pass_value(self):
        assert EdgeRating.PASS.value == "PASS"


class TestConstants:
    """Verify centralized constants are consistent."""

    def test_value_ratings_includes_strong_and_good(self):
        assert "STRONG" in VALUE_RATINGS
        assert "GOOD" in VALUE_RATINGS

    def test_value_ratings_excludes_marginal_and_pass(self):
        assert "MARGINAL" not in VALUE_RATINGS
        assert "PASS" not in VALUE_RATINGS

    def test_rating_rank_ordering(self):
        assert RATING_RANK["STRONG"] > RATING_RANK["GOOD"]
        assert RATING_RANK["GOOD"] > RATING_RANK["MARGINAL"]
        assert RATING_RANK["MARGINAL"] > RATING_RANK["PASS"]

    def test_thresholds_descending(self):
        assert STRONG_THRESHOLD > GOOD_THRESHOLD > MARGINAL_THRESHOLD > 0


class TestKalshiFeeCents:
    """Tests for kalshi_fee_cents()."""

    def test_fee_at_50_cents(self):
        # Max fee: 0.07 * 0.5 * 0.5 * 100 = 1.75c
        assert kalshi_fee_cents(50) == pytest.approx(1.75)

    def test_fee_at_0_cents(self):
        assert kalshi_fee_cents(0) == pytest.approx(0.0)

    def test_fee_at_100_cents(self):
        assert kalshi_fee_cents(100) == pytest.approx(0.0)

    def test_fee_symmetry(self):
        assert kalshi_fee_cents(30) == pytest.approx(kalshi_fee_cents(70))

    def test_fee_at_40_cents(self):
        # 0.07 * 0.4 * 0.6 * 100 = 1.68c
        assert kalshi_fee_cents(40) == pytest.approx(1.68)


class TestKalshiImpliedProb:
    """Tests for kalshi_implied_prob()."""

    def test_at_50_cents(self):
        # 0.5 + 0.07 * 0.5 * 0.5 = 0.5175
        assert kalshi_implied_prob(50) == pytest.approx(0.5175)

    def test_at_0_cents(self):
        assert kalshi_implied_prob(0) == pytest.approx(0.0)

    def test_at_100_cents(self):
        assert kalshi_implied_prob(100) == pytest.approx(1.0)

    def test_at_30_cents(self):
        # 0.3 + 0.07 * 0.3 * 0.7 = 0.3147
        assert kalshi_implied_prob(30) == pytest.approx(0.3147)

    def test_always_geq_raw_price(self):
        for price in [10, 25, 40, 50, 60, 75, 90]:
            assert kalshi_implied_prob(price) >= price / 100.0


class TestAnalyzeBet:
    """Tests for analyze_bet() with fee-adjusted calculations."""

    def test_returns_all_keys(self):
        result = analyze_bet(model_prob=0.6, kalshi_yes_price=50)
        assert set(result.keys()) == {"edge", "edge_pct", "rating", "implied_prob", "model_prob", "ev"}

    def test_edge_accounts_for_fees(self):
        result = analyze_bet(model_prob=0.6, kalshi_yes_price=50)
        # implied = 0.5175, edge = 0.6 - 0.5175 = 0.0825
        assert result["edge"] == pytest.approx(0.0825)
        assert result["implied_prob"] == pytest.approx(0.5175)

    def test_payout_includes_fee(self):
        result = analyze_bet(model_prob=0.6, kalshi_yes_price=50)
        effective_cost = 50 + 1.75  # fee at 50c
        expected_payout = (100 - effective_cost) / effective_cost
        expected_ev = 0.6 * expected_payout - 0.4 * 1.0
        assert result["ev"] == pytest.approx(expected_ev)

    def test_zero_price(self):
        result = analyze_bet(model_prob=0.5, kalshi_yes_price=0)
        assert result["ev"] == pytest.approx(-0.5)


class TestAmericanOddsToImpliedProb:
    """Tests for american_odds_to_implied_prob()."""

    def test_minus_110(self):
        assert american_odds_to_implied_prob("-110") == pytest.approx(0.5238, abs=0.001)

    def test_minus_150(self):
        # 150 / 250 = 0.600
        assert american_odds_to_implied_prob("-150") == pytest.approx(0.600)

    def test_plus_130(self):
        # 100 / 230 = 0.4348
        assert american_odds_to_implied_prob("+130") == pytest.approx(0.4348, abs=0.001)

    def test_plus_100(self):
        assert american_odds_to_implied_prob("+100") == pytest.approx(0.500)

    def test_minus_100(self):
        assert american_odds_to_implied_prob("-100") == pytest.approx(0.500)

    def test_heavy_favorite(self):
        # -300: 300 / 400 = 0.75
        assert american_odds_to_implied_prob("-300") == pytest.approx(0.75)

    def test_heavy_underdog(self):
        # +300: 100 / 400 = 0.25
        assert american_odds_to_implied_prob("+300") == pytest.approx(0.25)

    def test_numeric_input(self):
        assert american_odds_to_implied_prob(-110) == pytest.approx(0.5238, abs=0.001)

    def test_invalid_returns_none(self):
        assert american_odds_to_implied_prob("EVEN") is None
        assert american_odds_to_implied_prob("") is None
        assert american_odds_to_implied_prob(None) is None

    def test_zero_returns_none(self):
        assert american_odds_to_implied_prob("0") is None

    def test_nan_string_returns_none(self):
        assert american_odds_to_implied_prob("NaN") is None
        assert american_odds_to_implied_prob("nan") is None
        assert american_odds_to_implied_prob(float("nan")) is None
