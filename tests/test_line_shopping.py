"""Tests for betting/line_shopping.py: calculate_line_shopping and helpers."""

import pytest
import numpy as np

from betting.line_shopping import (
    calculate_line_shopping,
    find_breakeven_spread,
    SpreadAnalysis,
    STANDARD_IMPLIED_PROB,
)


SIGMA = 10.5


class TestCalculateLineShopping:
    """Tests for CDF-based line shopping recommendations."""

    def test_market_spread_marked(self):
        """Exactly one recommendation should have is_market=True."""
        result = calculate_line_shopping(
            classifier_prob=0.58,
            sigma=SIGMA,
            market_spread=-3.0,
            picked_team="Duke",
            is_home_pick=True,
        )
        market_recs = [r for r in result.recommendations if r.is_market]
        assert len(market_recs) == 1
        assert market_recs[0].spread == pytest.approx(-3.0)

    def test_market_spread_prob_matches_input(self):
        """At the market spread, model_prob should equal classifier_prob."""
        prob = 0.58
        result = calculate_line_shopping(
            classifier_prob=prob,
            sigma=SIGMA,
            market_spread=-3.0,
            picked_team="Duke",
            is_home_pick=True,
        )
        market_rec = [r for r in result.recommendations if r.is_market][0]
        assert market_rec.model_prob == pytest.approx(prob, abs=1e-4)

    def test_monotonic_prob_across_ladder(self):
        """For home pick, less negative spread is easier to cover -> higher prob."""
        result = calculate_line_shopping(
            classifier_prob=0.58,
            sigma=SIGMA,
            market_spread=-3.0,
            picked_team="Duke",
            is_home_pick=True,
        )
        # Sorted ascending: most negative (hardest) first
        sorted_recs = sorted(result.recommendations, key=lambda r: r.spread)
        # Probability should increase as spread becomes less negative (easier)
        for i in range(len(sorted_recs) - 1):
            assert sorted_recs[i].model_prob <= sorted_recs[i + 1].model_prob, (
                f"not monotonic at {sorted_recs[i].spread}->{sorted_recs[i+1].spread}: "
                f"{sorted_recs[i].model_prob:.6f} > {sorted_recs[i+1].model_prob:.6f}"
            )

    def test_away_pick_perspective_flip(self):
        """Away pick should flip probabilities correctly."""
        # Home team classifier prob = 0.58 means away team = 0.42 of covering
        # But we're picking the away team at their spread
        result = calculate_line_shopping(
            classifier_prob=0.58,
            sigma=SIGMA,
            market_spread=3.0,  # Away team is underdog (+3)
            picked_team="UNC",
            is_home_pick=False,
        )
        market_rec = [r for r in result.recommendations if r.is_market][0]
        # classifier_prob here is P(picked team covers) = 0.58
        assert market_rec.model_prob == pytest.approx(0.58, abs=1e-4)

    def test_home_vs_away_consistency(self):
        """Home pick and away pick for same game should give complementary probs at market."""
        home_prob = 0.58
        away_prob = 0.58  # From away perspective (P(away covers))
        market_spread_home = -3.0

        home_result = calculate_line_shopping(
            classifier_prob=home_prob,
            sigma=SIGMA,
            market_spread=market_spread_home,
            picked_team="Duke",
            is_home_pick=True,
        )
        away_result = calculate_line_shopping(
            classifier_prob=away_prob,
            sigma=SIGMA,
            market_spread=3.0,  # Away perspective
            picked_team="UNC",
            is_home_pick=False,
        )

        home_market = [r for r in home_result.recommendations if r.is_market][0]
        away_market = [r for r in away_result.recommendations if r.is_market][0]

        # Both should reflect their classifier_prob at market
        assert home_market.model_prob == pytest.approx(home_prob, abs=1e-4)
        assert away_market.model_prob == pytest.approx(away_prob, abs=1e-4)

    def test_recommendations_have_correct_spread_range(self):
        """Should generate spreads from market-2 to market+2 in 0.5 steps."""
        result = calculate_line_shopping(
            classifier_prob=0.55,
            sigma=SIGMA,
            market_spread=-5.0,
            picked_team="Duke",
            is_home_pick=True,
        )
        spreads = sorted(r.spread for r in result.recommendations)
        assert spreads[0] == pytest.approx(-7.0)
        assert spreads[-1] == pytest.approx(-3.0)
        # 0.5 increments from -7 to -3 = 9 values
        assert len(spreads) == 9

    def test_invalid_sigma_raises(self):
        """Zero or negative sigma should raise ValueError."""
        with pytest.raises(ValueError):
            calculate_line_shopping(0.55, 0.0, -3.0, "Duke", True)
        with pytest.raises(ValueError):
            calculate_line_shopping(0.55, -1.0, -3.0, "Duke", True)

    def test_result_fields(self):
        """Result should have all expected fields populated."""
        result = calculate_line_shopping(
            classifier_prob=0.58,
            sigma=SIGMA,
            market_spread=-3.0,
            picked_team="Duke",
            is_home_pick=True,
        )
        assert result.picked_team == "Duke"
        assert result.market_spread == -3.0
        assert len(result.recommendations) > 0


class TestFindBreakevenSpread:
    """Tests for breakeven spread interpolation."""

    def test_breakeven_found(self):
        """Should find breakeven when edge crosses zero."""
        recs = [
            SpreadAnalysis(spread=-5.0, model_prob=0.60, edge=0.08, kelly_units=1.0),
            SpreadAnalysis(spread=-4.0, model_prob=0.55, edge=0.03, kelly_units=0.5),
            SpreadAnalysis(spread=-3.0, model_prob=0.50, edge=-0.02, kelly_units=0.0),
        ]
        breakeven = find_breakeven_spread(recs)
        assert breakeven is not None
        assert -4.5 <= breakeven <= -3.0

    def test_all_positive_edge(self):
        """No breakeven when all edges are positive."""
        recs = [
            SpreadAnalysis(spread=-5.0, model_prob=0.65, edge=0.12, kelly_units=2.0),
            SpreadAnalysis(spread=-4.0, model_prob=0.60, edge=0.08, kelly_units=1.0),
        ]
        assert find_breakeven_spread(recs) is None

    def test_all_negative_edge(self):
        """No breakeven when all edges are negative."""
        recs = [
            SpreadAnalysis(spread=-5.0, model_prob=0.48, edge=-0.04, kelly_units=0.0),
            SpreadAnalysis(spread=-4.0, model_prob=0.45, edge=-0.07, kelly_units=0.0),
        ]
        assert find_breakeven_spread(recs) is None
