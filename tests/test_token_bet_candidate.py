"""Tests for the MLB token-bet confluence mask."""

from __future__ import annotations

import pandas as pd

from dashboard_helpers import token_bet_candidate_mask


def _row(
    *,
    rating="STRONG",
    status="ok",
    agrees=True,
    edge=0.05,
):
    return {
        "Std_Rating": rating,
        "MarketV2_Status": status,
        "MarketV2_Agrees_With_Production": agrees,
        "MarketV2_Edge_vs_Market": edge,
    }


def test_all_four_conditions_true_marks_candidate():
    df = pd.DataFrame([_row()])
    assert token_bet_candidate_mask(df).iloc[0] is True or bool(
        token_bet_candidate_mask(df).iloc[0]
    )


def test_pass_rating_excluded():
    df = pd.DataFrame([_row(rating="PASS")])
    assert not token_bet_candidate_mask(df).iloc[0]


def test_good_rating_included():
    df = pd.DataFrame([_row(rating="GOOD")])
    assert token_bet_candidate_mask(df).iloc[0]


def test_non_ok_shadow_status_excluded():
    df = pd.DataFrame([_row(status="odds_missing")])
    assert not token_bet_candidate_mask(df).iloc[0]


def test_shadow_disagrees_excluded():
    df = pd.DataFrame([_row(agrees=False)])
    assert not token_bet_candidate_mask(df).iloc[0]


def test_zero_or_negative_edge_excluded():
    df = pd.DataFrame([_row(edge=0.0), _row(edge=-0.01)])
    mask = token_bet_candidate_mask(df)
    assert not mask.iloc[0]
    assert not mask.iloc[1]


def test_nan_agrees_treated_as_false():
    df = pd.DataFrame([_row(agrees=float("nan"))])
    assert not token_bet_candidate_mask(df).iloc[0]


def test_blank_agrees_string_treated_as_false():
    """Empty-string agrees comes from the empty_shadow_columns path when
    MarketV2_Status is not 'ok'; should not promote to candidate."""
    df = pd.DataFrame([_row(agrees="", status="model_missing")])
    assert not token_bet_candidate_mask(df).iloc[0]


def test_string_edge_parsed_then_filtered():
    """Edge stored as string round-tripped through CSV must still gate."""
    df = pd.DataFrame([
        {**_row(), "MarketV2_Edge_vs_Market": "0.05"},
        {**_row(), "MarketV2_Edge_vs_Market": "-0.02"},
    ])
    mask = token_bet_candidate_mask(df)
    assert mask.iloc[0]
    assert not mask.iloc[1]


def test_missing_columns_yield_no_candidates():
    df = pd.DataFrame([{"unrelated": 1}])
    assert token_bet_candidate_mask(df).any() is False or not bool(
        token_bet_candidate_mask(df).any()
    )


def test_mixed_slate_picks_only_qualifying_rows():
    df = pd.DataFrame([
        _row(),                                        # candidate
        _row(rating="PASS"),                           # weak rating
        _row(status="odds_missing"),                   # no shadow data
        _row(agrees=False),                            # shadow disagrees
        _row(edge=0.0),                                # no positive edge
        _row(rating="GOOD", edge=0.01),                # candidate
    ])
    mask = token_bet_candidate_mask(df)
    assert int(mask.sum()) == 2
    assert bool(mask.iloc[0])
    assert bool(mask.iloc[5])
