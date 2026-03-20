"""Helpers for safely loading prediction CSV outputs."""

from __future__ import annotations

import os

import pandas as pd


def _has_non_whitespace_content(path: str, sample_size: int = 1024) -> bool:
    """Return True when the file starts with non-whitespace CSV content."""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return bool(handle.read(sample_size).strip())


def load_predictions_csv(path: str) -> tuple[pd.DataFrame, str]:
    """Load a predictions CSV, treating blank files as an empty no-picks result.

    Returns a tuple of ``(dataframe, status)`` where status is one of
    ``missing``, ``empty``, or ``loaded``.
    """
    if not os.path.exists(path):
        return pd.DataFrame(), "missing"

    if not _has_non_whitespace_content(path):
        return pd.DataFrame(), "empty"

    try:
        return pd.read_csv(path), "loaded"
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "empty"
