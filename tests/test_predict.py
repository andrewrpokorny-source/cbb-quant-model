import pytest
from predict import get_spread_model_label, _get_std_implied_prob, _compute_std_edge
from betting import STANDARD_IMPLIED_PROB


def test_predict_banner_uses_uncalibrated_label_for_mens():
    assert get_spread_model_label("mens") == "GBM"


def test_predict_banner_uses_uncalibrated_label_for_womens():
    assert get_spread_model_label("womens") == "GBM"


class TestStdImpliedProbFromRealOdds:
    """Verify Std_Rating uses real ESPN odds, not hardcoded -110."""

    def test_home_spread_odds_used_for_home_pick(self):
        game = {"home_spread_odds": "-115", "away_spread_odds": "-105"}
        # Home pick at -115: 115/215 = 0.5349
        result = _get_std_implied_prob(game, is_home_pick=True, bet_type="spread")
        assert result == pytest.approx(0.5349, abs=0.001)

    def test_away_spread_odds_used_for_away_pick(self):
        game = {"home_spread_odds": "-115", "away_spread_odds": "-105"}
        # Away pick at -105: 105/205 = 0.5122
        result = _get_std_implied_prob(game, is_home_pick=False, bet_type="spread")
        assert result == pytest.approx(0.5122, abs=0.001)

    def test_falls_back_to_standard_when_no_odds(self):
        game = {}
        result = _get_std_implied_prob(game, is_home_pick=True)
        assert result == STANDARD_IMPLIED_PROB

    def test_falls_back_on_empty_string_odds(self):
        game = {"home_spread_odds": "", "away_spread_odds": ""}
        result = _get_std_implied_prob(game, is_home_pick=True)
        assert result == STANDARD_IMPLIED_PROB

    def test_falls_back_on_unparseable_odds(self):
        game = {"home_spread_odds": "EVEN"}
        result = _get_std_implied_prob(game, is_home_pick=True)
        assert result == STANDARD_IMPLIED_PROB

    def test_falls_back_on_nan_odds(self):
        game = {"home_spread_odds": "NaN"}
        result = _get_std_implied_prob(game, is_home_pick=True)
        assert result == STANDARD_IMPLIED_PROB

    def test_moneyline_odds_used_for_ml_bet_type(self):
        game = {"home_ml_odds": "-150", "away_ml_odds": "+130"}
        # Home ML at -150: 150/250 = 0.600
        result = _get_std_implied_prob(game, is_home_pick=True, bet_type="ml")
        assert result == pytest.approx(0.600)

    def test_away_ml_odds(self):
        game = {"home_ml_odds": "-150", "away_ml_odds": "+130"}
        # Away ML at +130: 100/230 = 0.4348
        result = _get_std_implied_prob(game, is_home_pick=False, bet_type="ml")
        assert result == pytest.approx(0.4348, abs=0.001)


class TestComputeStdEdge:
    """Verify edge uses real odds, not fixed -110."""

    def test_edge_with_real_spread_odds(self):
        game = {"home_spread_odds": "-115", "away_spread_odds": "-105"}
        # conf=0.55, home pick at -115 (implied=0.5349)
        edge = _compute_std_edge(0.55, game, is_home_pick=True)
        assert edge == pytest.approx(0.0151, abs=0.001)

    def test_edge_different_from_fixed_110(self):
        """Real odds should give a different edge than the old hardcoded -110."""
        game_with_odds = {"home_spread_odds": "-120", "away_spread_odds": "-100"}
        game_without_odds = {}
        conf = 0.56
        edge_real = _compute_std_edge(conf, game_with_odds, is_home_pick=True)
        edge_fixed = _compute_std_edge(conf, game_without_odds, is_home_pick=True)
        # -120 implied = 0.5455, -110 implied = 0.5238
        # edge_real = 0.56 - 0.5455 = 0.0145
        # edge_fixed = 0.56 - 0.5238 = 0.0362
        assert edge_real != pytest.approx(edge_fixed, abs=0.005)
        assert edge_real < edge_fixed  # harder to beat -120 than -110

    def test_heavy_juice_reduces_edge_to_zero(self):
        """A game with heavy -130 juice should have less edge than -110."""
        game = {"home_spread_odds": "-130"}
        # conf=0.56, implied at -130 = 130/230 = 0.5652
        edge = _compute_std_edge(0.56, game, is_home_pick=True)
        assert edge < 0  # model is 56% but breakeven is 56.5%

    def test_plus_odds_makes_edge_larger(self):
        """Underdog spread at +100 means lower implied prob = more edge."""
        game = {"away_spread_odds": "+100"}
        # +100 implied = 0.500
        edge = _compute_std_edge(0.55, game, is_home_pick=False)
        assert edge == pytest.approx(0.05)  # 55% - 50% = 5%
