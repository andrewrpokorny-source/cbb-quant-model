import numpy as np
import pytest

from mlb.market_v2 import (
    MARKET_V2_FEATURES,
    build_shadow_columns,
    market_home_no_vig_probability,
    predict_market_v2_home_prob,
)


class FixedProbModel:
    def __init__(self, prob_home):
        self.prob_home = prob_home

    def predict_proba(self, X):
        return np.array([[1.0 - self.prob_home, self.prob_home]])


def _feature_row():
    return {feature: 0.0 for feature in MARKET_V2_FEATURES if feature != "market_home_no_vig_prob"}


def _bundle(prob_home=0.62):
    return {
        "model": FixedProbModel(prob_home),
        "features": list(MARKET_V2_FEATURES),
    }


def test_market_home_no_vig_probability_removes_overround():
    prob = market_home_no_vig_probability("-150", "+130")

    assert prob == pytest.approx(0.5798, abs=0.0001)


def test_predict_market_v2_home_prob_scores_live_moneyline_feature():
    result = predict_market_v2_home_prob(
        _bundle(0.62),
        _feature_row(),
        "-150",
        "+130",
    )

    assert result["status"] == "ok"
    assert result["prob_home"] == pytest.approx(0.62)
    assert result["pick_home"] is True
    assert result["market_home_no_vig_prob"] == pytest.approx(0.5798, abs=0.0001)
    assert result["edge_vs_market"] == pytest.approx(0.0402, abs=0.0001)


def test_predict_market_v2_home_prob_reports_missing_odds():
    result = predict_market_v2_home_prob(_bundle(), _feature_row(), "", "")

    assert result["status"] == "odds_missing"
    assert result["market_home_no_vig_prob"] is None


def test_build_shadow_columns_are_additive_and_do_not_replace_pick():
    columns = build_shadow_columns(
        _bundle(0.62),
        _feature_row(),
        home_team="New York Yankees",
        away_team="Boston Red Sox",
        home_moneyline="-150",
        away_moneyline="+130",
        production_pick="Boston Red Sox",
    )

    assert "Pick" not in columns
    assert columns["MarketV2_Status"] == "ok"
    assert columns["MarketV2_Pick"] == "New York Yankees"
    assert columns["MarketV2_Agrees_With_Production"] is False
    assert columns["MarketV2_Edge_vs_Market"] == pytest.approx(0.0402, abs=0.0001)


def test_build_shadow_columns_reports_model_missing():
    columns = build_shadow_columns(
        None,
        _feature_row(),
        home_team="New York Yankees",
        away_team="Boston Red Sox",
        home_moneyline="-150",
        away_moneyline="+130",
        production_pick="Boston Red Sox",
    )

    assert columns["MarketV2_Status"] == "model_missing"
    assert columns["MarketV2_Pick"] == ""
