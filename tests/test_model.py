"""Tests for model.py: temporal split helpers, cover_prob_at_spread, and load_model."""

import math
import tempfile
import os
import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from model import (
    FEATURES,
    MENS_FEATURES,
    TimeAwareCalibratedGBM,
    build_spread_estimator,
    cover_prob_at_spread,
    get_feature_list,
    load_model,
    prepare_time_ordered_training_frame,
    time_series_train_test_split,
    use_calibrated_spread_model,
    walk_forward_validate,
)


def test_production_features_keep_neutral_drop_distance():
    assert "is_neutral" in FEATURES
    assert "distance_advantage" not in FEATURES


def test_mens_feature_list_uses_torvik_priors_and_drops_dead_inputs():
    mens_features = get_feature_list("mens")
    assert mens_features == MENS_FEATURES
    assert "diff_eFG" not in mens_features
    assert "momentum_gap" not in mens_features
    assert "hasla_diff_rank_strength" not in mens_features
    assert "torvik_diff_adj_oe" in mens_features
    assert "torvik_diff_ftr" in mens_features


def test_mens_spread_model_defaults_to_uncalibrated_gbm():
    assert use_calibrated_spread_model("mens") is False
    assert isinstance(build_spread_estimator("mens"), GradientBoostingClassifier)


def test_womens_spread_model_keeps_calibration():
    assert use_calibrated_spread_model("womens") is True
    assert isinstance(build_spread_estimator("womens"), TimeAwareCalibratedGBM)


def test_calibrated_gbm_pickles_under_model_module():
    assert TimeAwareCalibratedGBM.__module__ == "model"


def test_calibrated_gbm_exposes_feature_names_after_fit():
    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = [0, 1, 0, 1]
    clf = TimeAwareCalibratedGBM(min_calibration_rows=9999)
    clf.fit(X, y)
    assert clf.feature_names_in_.tolist() == ["a", "b"]
    assert clf.n_features_in_ == 2


class TestTimeAwareTrainingSplit:
    def test_prepare_frame_sorts_by_date_not_input_order(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-03", "2026-01-01", "2026-01-02"],
                "team": ["A", "A", "A"],
                "opponent": ["B", "B", "B"],
                "is_home": [1, 1, 1],
                "is_neutral": [0, 0, 0],
                "distance_advantage": [0.1, 0.2, 0.3],
                "spread": [1.0, 2.0, 3.0],
                "rest_days": [2, 2, 2],
                "diff_eFG": [0.0, 0.0, 0.0],
                "diff_Rebound": [0.0, 0.0, 0.0],
                "diff_TO": [0.0, 0.0, 0.0],
                "momentum_gap": [0.0, 0.0, 0.0],
                "roll5_cover_margin": [0.0, 0.0, 0.0],
                "prev_games_played": [1, 1, 1],
                "opp_win_pct": [0.5, 0.5, 0.5],
                "prev_blowout_rate": [0.0, 0.0, 0.0],
                "prev_roll5_margin": [0.0, 0.0, 0.0],
                "prev_volatility": [1.0, 1.0, 1.0],
                "spread_abs": [1.0, 2.0, 3.0],
                "spread_squared": [1.0, 4.0, 9.0],
                "ats_win": [1, 0, 1],
            }
        )
        result = prepare_time_ordered_training_frame(df, ["is_home", "is_neutral", "distance_advantage", "spread", "rest_days", "diff_eFG", "diff_Rebound", "diff_TO", "momentum_gap", "roll5_cover_margin", "prev_games_played", "opp_win_pct", "prev_blowout_rate", "prev_roll5_margin", "prev_volatility", "spread_abs", "spread_squared"], "ats_win")
        assert result["date"].tolist() == ["2026-01-01", "2026-01-02", "2026-01-03"]

    def test_time_series_split_keeps_latest_rows_in_test_set(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01", "2026-01-01",
                        "2026-01-02", "2026-01-02",
                        "2026-01-03", "2026-01-03",
                        "2026-01-04", "2026-01-04",
                    ]
                ),
                "team": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "opponent": ["B", "A", "D", "C", "F", "E", "H", "G"],
                "is_home": [1, 0, 1, 0, 1, 0, 1, 0],
                "is_neutral": [0] * 8,
                "distance_advantage": np.linspace(0, 1, 8),
                "spread": np.arange(8, dtype=float),
                "rest_days": [2] * 8,
                "diff_eFG": [0.0] * 8,
                "diff_Rebound": [0.0] * 8,
                "diff_TO": [0.0] * 8,
                "momentum_gap": [0.0] * 8,
                "roll5_cover_margin": [0.0] * 8,
                "prev_games_played": np.arange(8),
                "opp_win_pct": [0.5] * 8,
                "prev_blowout_rate": [0.0] * 8,
                "prev_roll5_margin": [0.0] * 8,
                "prev_volatility": [1.0] * 8,
                "spread_abs": np.arange(8, dtype=float),
                "spread_squared": np.arange(8, dtype=float) ** 2,
                "ats_win": [1, 0, 0, 1, 1, 0, 0, 1],
            }
        )
        features = ["is_home", "is_neutral", "distance_advantage", "spread", "rest_days", "diff_eFG", "diff_Rebound", "diff_TO", "momentum_gap", "roll5_cover_margin", "prev_games_played", "opp_win_pct", "prev_blowout_rate", "prev_roll5_margin", "prev_volatility", "spread_abs", "spread_squared"]
        prepared = prepare_time_ordered_training_frame(df, features, "ats_win")
        X_train, X_test, y_train, y_test = time_series_train_test_split(
            prepared, features, "ats_win", test_size=0.25, bet_level_test=True
        )
        assert len(X_train) == 6  # first 3 games, both rows
        assert len(X_test) == 1   # latest game, home row only
        assert X_test["spread"].tolist() == [6.0]
        assert y_test.tolist() == [0]

    def test_prepare_frame_requires_date_column(self):
        df = pd.DataFrame({"ats_win": [1], "is_home": [1]})
        with pytest.raises(ValueError):
            prepare_time_ordered_training_frame(df, ["is_home"], "ats_win")

    def test_walk_forward_validate_scores_home_rows_only(self):
        rows = []
        base_features = {
            "is_neutral": 0,
            "spread": -2.5,
            "rest_days": 3,
            "hasla_diff_rank_strength": 0.1,
            "hasla_diff_off_rank_strength": 0.1,
            "hasla_diff_def_rank_strength": 0.1,
            "torvik_diff_adj_oe": 1.0,
            "torvik_diff_adj_de": 1.0,
            "torvik_diff_barthag": 0.1,
            "torvik_tempo_gap": 0.5,
            "torvik_diff_efg": 0.5,
            "torvik_diff_tor": 0.5,
            "torvik_diff_orb": 0.5,
            "torvik_diff_ftr": 0.5,
            "roll5_cover_margin": 0.0,
            "prev_games_played": 10,
            "opp_win_pct": 0.5,
            "prev_blowout_rate": 0.2,
            "prev_roll5_margin": 1.0,
            "prev_volatility": 9.0,
            "spread_abs": 2.5,
            "spread_squared": 6.25,
        }
        for game_idx in range(60):
            game_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=game_idx)
            ats_win = int(game_idx % 2 == 0)
            home_row = {
                "date": game_date,
                "team": f"H{game_idx}",
                "opponent": f"A{game_idx}",
                "is_home": 1,
                "ats_win": ats_win,
                **base_features,
            }
            away_row = {
                "date": game_date,
                "team": f"A{game_idx}",
                "opponent": f"H{game_idx}",
                "is_home": 0,
                "ats_win": 1 - ats_win,
                **base_features,
            }
            rows.extend([home_row, away_row])

        df = pd.DataFrame(rows)
        metrics = walk_forward_validate(
            df,
            get_feature_list("mens"),
            "ats_win",
            estimator_factory=lambda: TimeAwareCalibratedGBM(min_calibration_rows=9999),
            weeks_back=2,
        )

        assert metrics["total_bets"] > 0
        assert metrics["total_bets"] <= len(df[df["is_home"] == 1])


class TestCoverProbAtSpread:
    """Tests for CDF projection of classifier probability to alternate spreads."""

    SIGMA = 10.5  # Typical training sigma

    def test_anchor_invariant(self):
        """At market spread, should return exactly the classifier probability."""
        for p in [0.45, 0.50, 0.53, 0.60, 0.75]:
            result = cover_prob_at_spread(p, -3.0, -3.0, self.SIGMA)
            assert result == pytest.approx(p, abs=1e-6), (
                f"anchor failed: p={p}, got {result}"
            )

    def test_monotonic_home_favorite(self):
        """Less negative spread is easier to cover, so probability increases."""
        p = 0.55
        market = -5.0
        # Spreads from hardest to cover (-7) to easiest (-3)
        spreads = [-7.0, -6.0, -5.0, -4.0, -3.0]
        probs = [cover_prob_at_spread(p, market, s, self.SIGMA) for s in spreads]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1], (
                f"not monotonic at spreads {spreads[i]}->{spreads[i+1]}: "
                f"{probs[i]:.6f} >= {probs[i+1]:.6f}"
            )

    def test_monotonic_home_underdog(self):
        """More favorable spread (more positive) should increase cover prob."""
        p = 0.45
        market = 3.0
        spreads = [1.0, 2.0, 3.0, 4.0, 5.0]
        probs = [cover_prob_at_spread(p, market, s, self.SIGMA) for s in spreads]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1], (
                f"not monotonic at spreads {spreads[i]}->{spreads[i+1]}: "
                f"{probs[i]:.6f} >= {probs[i+1]:.6f}"
            )

    def test_symmetry(self):
        """p=0.5 at spread=0 should stay 0.5 at spread=0."""
        result = cover_prob_at_spread(0.5, 0.0, 0.0, self.SIGMA)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_output_bounded(self):
        """Output should always be in (0, 1)."""
        cases = [
            (0.001, -10.0, 10.0, self.SIGMA),
            (0.999, 10.0, -10.0, self.SIGMA),
            (0.5, 0.0, 20.0, self.SIGMA),
            (0.5, 0.0, -20.0, self.SIGMA),
        ]
        for p, ms, alt, sig in cases:
            result = cover_prob_at_spread(p, ms, alt, sig)
            assert 0.0 < result < 1.0, f"out of bounds: {result} for inputs {(p, ms, alt, sig)}"

    def test_large_harder_spread(self):
        """A much more negative spread is very hard to cover -> near 0."""
        result = cover_prob_at_spread(0.55, -5.0, -25.0, self.SIGMA)
        assert result < 0.10

    def test_large_easier_spread(self):
        """A large positive spread is very easy to cover -> near 1."""
        result = cover_prob_at_spread(0.55, -5.0, 15.0, self.SIGMA)
        assert result > 0.90

    def test_different_sigma_scales(self):
        """Smaller sigma should produce sharper probability changes."""
        p, market, alt = 0.55, -5.0, -7.0
        prob_small_sigma = cover_prob_at_spread(p, market, alt, 5.0)
        prob_large_sigma = cover_prob_at_spread(p, market, alt, 20.0)
        # Smaller sigma -> bigger change from anchor
        assert abs(prob_small_sigma - p) > abs(prob_large_sigma - p)


class TestLoadModel:
    """Tests for load_model backward compatibility."""

    def test_new_format(self, tmp_path):
        """New format: dict with 'model' and 'sigma' keys."""
        model_obj = {"fake": "model"}
        sigma_val = 10.5
        path = tmp_path / "model.pkl"
        joblib.dump({"model": model_obj, "sigma": sigma_val}, path)

        model, sigma = load_model(str(path))
        assert model == model_obj
        assert sigma == pytest.approx(10.5)

    def test_old_format_defaults_sigma(self, tmp_path):
        """Old format: raw model object, should default sigma to 11.0."""
        model_obj = {"fake": "old_model"}
        path = tmp_path / "model.pkl"
        joblib.dump(model_obj, path)

        model, sigma = load_model(str(path))
        assert model == model_obj
        assert sigma == pytest.approx(11.0)

    def test_new_format_missing_sigma_defaults(self, tmp_path):
        """New format dict without sigma key should default to 11.0."""
        model_obj = {"fake": "model"}
        path = tmp_path / "model.pkl"
        joblib.dump({"model": model_obj}, path)

        model, sigma = load_model(str(path))
        assert model == model_obj
        assert sigma == pytest.approx(11.0)

    def test_file_not_found(self):
        """Missing file should raise."""
        with pytest.raises((FileNotFoundError, IOError)):
            load_model("/nonexistent/path/model.pkl")
