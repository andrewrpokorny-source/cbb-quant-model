"""Tests for model.py: cover_prob_at_spread and load_model."""

import math
import tempfile
import os
import pytest
import joblib
import numpy as np

from model import cover_prob_at_spread, load_model


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
