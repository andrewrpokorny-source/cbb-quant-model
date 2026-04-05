"""Tests for MLB feature selection utilities."""

import numpy as np
import pandas as pd
import pytest

from mlb.feature_selection import compute_permutation_importance, prune_features


class TestPermutationImportance:

    def test_returns_all_features(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5), columns=["f1", "f2", "f3", "f4", "f5"])
        y = pd.Series(np.random.randint(0, 2, 200))
        importance = compute_permutation_importance(X, y)
        assert set(importance.keys()) == {"f1", "f2", "f3", "f4", "f5"}

    def test_importance_values_are_floats(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randint(0, 2, 200))
        importance = compute_permutation_importance(X, y)
        for v in importance.values():
            assert isinstance(v, float)


class TestPruneFeatures:

    def test_excludes_negative_importance(self):
        importance = {"f1": 0.05, "f2": -0.01, "f3": 0.00, "f4": 0.02}
        pruned = prune_features(importance, threshold=0.0)
        assert "f1" in pruned
        assert "f4" in pruned
        assert "f2" not in pruned
        assert "f3" not in pruned

    def test_empty_when_all_negative(self):
        importance = {"f1": -0.01, "f2": -0.05}
        assert prune_features(importance) == []
