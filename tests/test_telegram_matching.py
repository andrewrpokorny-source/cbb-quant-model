"""Tests for settle-matching logic in telegram_bot.py.

Covers _find_matching_pending_bet scoring and _find_kalshi_pending_by_spread
fallback matching.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import os
import sys
import types


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _install_optional_stubs() -> None:
    """Install lightweight stubs so tests run without bot extras."""
    if not _module_available("telegram"):
        telegram_mod = types.ModuleType("telegram")

        class Update:
            pass

        telegram_mod.Update = Update
        sys.modules["telegram"] = telegram_mod

    if not _module_available("telegram.ext"):
        telegram_ext_mod = types.ModuleType("telegram.ext")

        class _FilterTerm:
            def __and__(self, other):
                return self

            def __rand__(self, other):
                return self

            def __invert__(self):
                return self

        class _ContextTypes:
            DEFAULT_TYPE = object

        class _ApplicationBuilder:
            def token(self, *_args, **_kwargs):
                return self

            def build(self):
                return self

        telegram_ext_mod.ApplicationBuilder = _ApplicationBuilder
        telegram_ext_mod.CommandHandler = lambda *args, **kwargs: None
        telegram_ext_mod.MessageHandler = lambda *args, **kwargs: None
        telegram_ext_mod.ContextTypes = _ContextTypes
        telegram_ext_mod.filters = types.SimpleNamespace(
            PHOTO=_FilterTerm(),
            TEXT=_FilterTerm(),
            COMMAND=_FilterTerm(),
        )
        sys.modules["telegram.ext"] = telegram_ext_mod

    if not _module_available("ocrmac"):
        ocrmac_mod = types.ModuleType("ocrmac")

        class _DummyOCR:
            def __init__(self, *_args, **_kwargs):
                pass

            def recognize(self):
                return []

        ocrmac_mod.ocrmac = types.SimpleNamespace(OCR=_DummyOCR)
        sys.modules["ocrmac"] = ocrmac_mod


_install_optional_stubs()
BOT = importlib.import_module("telegram_bot")

CSV_HEADERS = BOT.CSV_HEADERS


def _write_csv(path, rows):
    """Write a betting_history.csv with the given rows (list of dicts)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            full = {h: "" for h in CSV_HEADERS}
            full.update(row)
            writer.writerow(full)


# ---------------------------------------------------------------------------
# Fix 1: _find_matching_pending_bet scoring tests
# ---------------------------------------------------------------------------


def test_two_shared_team_correct_match(tmp_path, monkeypatch):
    """Two pending bets share a team; both teams from slip -> correct bet."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "FanDuel", "game": "UTSA vs UTEP", "line": "UTSA -3.5",
         "wager": "1.00", "result": "pending"},
        {"platform": "FanDuel", "game": "UTSA vs Rice", "line": "UTSA -5.5",
         "wager": "1.00", "result": "pending"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    partial = {"platform": "FanDuel", "wager": 1.00, "_teams": ["UTSA", "RICE"]}
    result = BOT._find_matching_pending_bet(partial)
    assert result is not None
    assert result["line"] == "UTSA -5.5"


def test_single_team_unique_match(tmp_path, monkeypatch):
    """Single team token uniquely identifies one bet."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "FanDuel", "game": "Duke vs UNC", "line": "Duke -2.5",
         "wager": "1.00", "result": "pending"},
        {"platform": "FanDuel", "game": "Kansas vs Baylor", "line": "Kansas -4.5",
         "wager": "1.00", "result": "pending"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    partial = {"platform": "FanDuel", "wager": 1.00, "_teams": ["BAYLOR"]}
    result = BOT._find_matching_pending_bet(partial)
    assert result is not None
    assert result["line"] == "Kansas -4.5"


def test_single_team_matches_multiple_returns_none(tmp_path, monkeypatch):
    """Single team token matches multiple bets (tied score) -> None."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "FanDuel", "game": "UTSA vs UTEP", "line": "UTSA -3.5",
         "wager": "1.00", "result": "pending"},
        {"platform": "FanDuel", "game": "UTSA vs Rice", "line": "UTSA -5.5",
         "wager": "1.00", "result": "pending"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    partial = {"platform": "FanDuel", "wager": 1.00, "_teams": ["UTSA"]}
    result = BOT._find_matching_pending_bet(partial)
    assert result is None


def test_no_teams_single_candidate_accepted(tmp_path, monkeypatch):
    """No teams + single candidate -> accepted."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "FanDuel", "game": "Duke vs UNC", "line": "Duke -2.5",
         "wager": "1.00", "result": "pending"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    partial = {"platform": "FanDuel", "wager": 1.00, "_teams": []}
    result = BOT._find_matching_pending_bet(partial)
    assert result is not None
    assert result["line"] == "Duke -2.5"


def test_no_teams_multiple_candidates_returns_none(tmp_path, monkeypatch):
    """No teams + multiple candidates -> None."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "FanDuel", "game": "Duke vs UNC", "line": "Duke -2.5",
         "wager": "1.00", "result": "pending"},
        {"platform": "FanDuel", "game": "Kansas vs Baylor", "line": "Kansas -4.5",
         "wager": "1.00", "result": "pending"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    partial = {"platform": "FanDuel", "wager": 1.00, "_teams": []}
    result = BOT._find_matching_pending_bet(partial)
    assert result is None


# ---------------------------------------------------------------------------
# Fix 2: _find_kalshi_pending_by_spread tests
# ---------------------------------------------------------------------------


def test_kalshi_abbreviated_ticker_matches_full_name(tmp_path, monkeypatch):
    """Abbreviated ticker line matches pending full-name bet by spread/side/wager."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "Kalshi", "game": "Pacific vs Saint Mary's",
         "line": "Saint Mary's -8.5 YES", "wager": "1.53",
         "result": "pending", "date": "2026-02-14"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    bet = {
        "platform": "Kalshi",
        "line": "SMC -8.5 YES",
        "wager": 1.53,
        "result": "win",
        "payout": 2.50,
        "profit": 0.97,
    }
    result = BOT._find_kalshi_pending_by_spread(bet)
    assert result is not None
    assert "Updated" in result


def test_kalshi_two_matching_pending_returns_none(tmp_path, monkeypatch):
    """Two pending Kalshi bets with same spread/side/wager -> None (ambiguous)."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "Kalshi", "game": "Pacific vs Saint Mary's",
         "line": "Saint Mary's -8.5 YES", "wager": "1.53",
         "result": "pending", "date": "2026-02-14"},
        {"platform": "Kalshi", "game": "Gonzaga vs BYU",
         "line": "BYU -8.5 YES", "wager": "1.53",
         "result": "pending", "date": "2026-02-14"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    bet = {
        "platform": "Kalshi",
        "line": "SMC -8.5 YES",
        "wager": 1.53,
        "result": "win",
        "payout": 2.50,
        "profit": 0.97,
    }
    result = BOT._find_kalshi_pending_by_spread(bet)
    assert result is None


def test_kalshi_fallback_non_kalshi_returns_none(tmp_path, monkeypatch):
    """Non-Kalshi bet -> None (platform filter)."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "Kalshi", "game": "Pacific vs Saint Mary's",
         "line": "Saint Mary's -8.5 YES", "wager": "1.53",
         "result": "pending"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    bet = {
        "platform": "FanDuel",
        "line": "SMC -8.5",
        "wager": 1.53,
        "result": "win",
        "payout": 2.50,
        "profit": 0.97,
    }
    result = BOT._find_kalshi_pending_by_spread(bet)
    assert result is None


def test_kalshi_fallback_different_spread_returns_none(tmp_path, monkeypatch):
    """Different spread -> None."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "Kalshi", "game": "Pacific vs Saint Mary's",
         "line": "Saint Mary's -8.5 YES", "wager": "1.53",
         "result": "pending"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    bet = {
        "platform": "Kalshi",
        "line": "SMC -10.5 YES",
        "wager": 1.53,
        "result": "win",
        "payout": 2.50,
        "profit": 0.97,
    }
    result = BOT._find_kalshi_pending_by_spread(bet)
    assert result is None
