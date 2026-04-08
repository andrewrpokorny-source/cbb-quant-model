"""Tests for settle-matching logic in telegram_bot.py.

Covers _find_matching_pending_bet scoring, _find_kalshi_pending_by_spread
fallback matching, and FanDuel settled bet routing/dedup.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import os
import sys
import types

import pytest


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


# ---------------------------------------------------------------------------
# Fix 3: FanDuel settled bet routing and BET ID dedup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_seen_bet_ids():
    """Clear cross-screenshot dedup set between tests."""
    BOT._SEEN_FD_BET_IDS.clear()
    yield
    BOT._SEEN_FD_BET_IDS.clear()


def test_settled_updates_pending_bet(tmp_path, monkeypatch):
    """Settled bet with matching pending -> updates the pending row."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "FanDuel", "game": "Drexel vs Towson",
         "line": "Drexel -1.5", "wager": "1.00", "odds": "-110",
         "result": "pending", "date": "2026-02-19"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    settled = {
        "platform": "FanDuel",
        "line": "Drexel -1.5",
        "wager": 1.00,
        "odds": "-110",
        "result": "win",
        "payout": 1.91,
        "profit": 0.91,
        "bet_id": "FD-ABC-001",
        "game": "Drexel vs Towson",
        "date": "2026-02-19",
    }
    msg = BOT.update_bet_result(settled)
    assert msg is not None
    assert "Updated" in msg
    assert "WIN" in msg

    # Verify CSV was updated
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["result"] == "win"
    assert rows[0]["payout"] == "1.91"
    assert rows[0]["bet_id"] == "FD-ABC-001"


def test_settled_no_pending_appends_as_settled(tmp_path, monkeypatch):
    """No matching pending bet -> append as settled."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    settled = {
        "platform": "FanDuel",
        "line": "Drexel -1.5",
        "wager": 1.00,
        "odds": "-110",
        "result": "win",
        "payout": 1.91,
        "profit": 0.91,
        "bet_id": "FD-ABC-001",
        "date": "2026-02-19",
    }
    row = BOT.append_bet(settled)
    assert row is not None
    assert row["result"] == "win"
    assert row["bet_id"] == "FD-ABC-001"


def test_settled_duplicate_bet_id_skipped(tmp_path, monkeypatch):
    """Same bet_id already in CSV -> duplicate skipped."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "FanDuel", "game": "Drexel vs Towson",
         "line": "Drexel -1.5", "wager": "1.00", "odds": "-110",
         "result": "win", "payout": "1.91", "profit": "0.91",
         "date": "2026-02-19", "bet_id": "FD-ABC-001"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    settled = {
        "platform": "FanDuel",
        "line": "Drexel -1.5",
        "wager": 1.00,
        "result": "win",
        "payout": 1.91,
        "profit": 0.91,
        "bet_id": "FD-ABC-001",
    }
    row = BOT.append_bet(settled)
    assert row is None  # duplicate


def test_settled_does_not_overwrite_existing_bet_id(tmp_path, monkeypatch):
    """Existing bet_id should not be overwritten by a different bet_id."""
    csv_path = tmp_path / "betting_history.csv"
    _write_csv(csv_path, [
        {"platform": "FanDuel", "game": "Drexel vs Towson",
         "line": "Drexel -1.5", "wager": "1.00", "odds": "-110",
         "result": "pending", "date": "2026-02-19", "bet_id": "FD-ORIGINAL"},
    ])
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))

    settled = {
        "platform": "FanDuel", "line": "Drexel -1.5", "wager": 1.00,
        "odds": "-110", "result": "win", "payout": 1.91, "profit": 0.91,
        "bet_id": "FD-DIFFERENT", "game": "Drexel vs Towson", "date": "2026-02-19",
    }
    msg = BOT.update_bet_result(settled)
    assert msg is not None

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["bet_id"] == "FD-ORIGINAL"  # not overwritten


def test_csv_migration_adds_bet_id_column(tmp_path, monkeypatch):
    """Pre-existing CSV without bet_id column gets bet_id added."""
    csv_path = tmp_path / "betting_history.csv"
    old_headers = ["date", "platform", "game", "bet_type", "line",
                   "odds", "wager", "result", "payout", "profit"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_headers)
        writer.writeheader()
        writer.writerow({
            "date": "2026-02-18", "platform": "FanDuel", "game": "Duke vs UNC",
            "bet_type": "spread", "line": "Duke -2.5", "odds": "-110",
            "wager": "1.00", "result": "win", "payout": "1.91", "profit": "0.91",
        })
    monkeypatch.setattr(BOT, "BETTING_HISTORY", str(csv_path))
    BOT.ensure_csv_exists()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert "bet_id" in reader.fieldnames
    assert len(rows) == 1
    assert rows[0]["bet_id"] == ""
    assert rows[0]["date"] == "2026-02-18"


# ---------------------------------------------------------------------------
# _resolve_game_date -- women's prediction file support
# ---------------------------------------------------------------------------


def test_resolve_game_date_finds_wbb_daily(tmp_path, monkeypatch):
    """_resolve_game_date should search daily_predictions_wbb.csv."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    wbb_csv = data_dir / "daily_predictions_wbb.csv"
    wbb_csv.write_text(
        "Pick,Matchup,Date/Time\n"
        "South Carolina Gamecocks,LSU @ South Carolina,03/06 7:00 PM\n"
    )
    monkeypatch.setattr(BOT, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(BOT, "DAILY_PREDICTIONS", str(data_dir / "daily_predictions.csv"))
    result = BOT._resolve_game_date("South Carolina")
    assert result == f"{BOT.datetime.now(BOT.pytz.timezone('US/Eastern')).year}-03-06"


def test_resolve_game_date_finds_wbb_dated(tmp_path, monkeypatch):
    """_resolve_game_date should search predictions_wbb_YYYYMMDD.csv files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    eastern = BOT.pytz.timezone("US/Eastern")
    today = BOT.datetime.now(eastern)
    date_str = today.strftime("%Y%m%d")
    wbb_csv = data_dir / f"predictions_wbb_{date_str}.csv"
    wbb_csv.write_text(
        "Pick,Matchup,Date/Time\n"
        "Iowa Hawkeyes,Iowa @ Nebraska,03/06 8:00 PM\n"
    )
    monkeypatch.setattr(BOT, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(BOT, "DAILY_PREDICTIONS", str(data_dir / "daily_predictions.csv"))
    result = BOT._resolve_game_date("Iowa")
    assert result == f"{today.year}-03-06"
