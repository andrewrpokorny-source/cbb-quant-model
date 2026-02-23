"""Regression tests for OCR bet-slip parsing using fixture text snapshots."""

from __future__ import annotations

from pathlib import Path
import importlib
import importlib.util
import sys
import types

import pytest


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _install_optional_stubs() -> None:
    """Install lightweight stubs so parser tests run without bot extras."""
    if not _module_available("telegram"):
        telegram_mod = types.ModuleType("telegram")

        class Update:  # pragma: no cover - only used when telegram isn't installed
            pass

        telegram_mod.Update = Update
        sys.modules["telegram"] = telegram_mod

    if not _module_available("telegram.ext"):
        telegram_ext_mod = types.ModuleType("telegram.ext")

        class _FilterTerm:  # pragma: no cover - only used when telegram isn't installed
            def __and__(self, other):
                return self

            def __rand__(self, other):
                return self

            def __invert__(self):
                return self

        class _ContextTypes:  # pragma: no cover
            DEFAULT_TYPE = object

        class _ApplicationBuilder:  # pragma: no cover
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

        class _DummyOCR:  # pragma: no cover
            def __init__(self, *_args, **_kwargs):
                pass

            def recognize(self):
                return []

        ocrmac_mod.ocrmac = types.SimpleNamespace(OCR=_DummyOCR)
        sys.modules["ocrmac"] = ocrmac_mod


def _load_bot_module():
    _install_optional_stubs()
    return importlib.import_module("telegram_bot")


BOT = _load_bot_module()
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ocr"


def _read_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def test_draftkings_fixture_regression() -> None:
    text = _read_fixture("draftkings_spread_single.txt")
    bets = BOT._parse_bet_slip_text(text, platform="DraftKings")

    assert len(bets) == 1
    bet = bets[0]
    assert bet["platform"] == "DraftKings"
    assert bet["game"] == "XAVIER vs MARQUETTE"
    assert bet["bet_type"] == "spread"
    assert bet["line"] == "MARQUETTE +5.5"
    assert bet["odds"] == "-110"
    assert bet["wager"] == 1.25


def test_fanduel_fixture_regression() -> None:
    text = _read_fixture("fanduel_spread_single.txt")
    bets = BOT._parse_bet_slip_text(text, platform="FanDuel")

    assert len(bets) == 1
    bet = bets[0]
    assert bet["platform"] == "FanDuel"
    assert bet["game"] == "Providence vs UConn"
    assert bet["bet_type"] == "spread"
    assert bet["line"] == "Providence +15.5"
    assert bet["odds"] == "-108"
    assert bet["wager"] == 1.25


def test_kalshi_fixture_regression() -> None:
    text = _read_fixture("kalshi_spread_single.txt")
    bets = BOT._parse_bet_slip_text(text, platform="Kalshi")

    assert len(bets) == 1
    bet = bets[0]
    assert bet["platform"] == "Kalshi"
    assert bet["game"] == "Providence vs UConn"
    assert bet["bet_type"] == "spread"
    assert bet["line"] == "UConn -15.5 YES"
    assert bet["odds"] == "n/a"
    assert bet["wager"] == 1.53


# ---------------------------------------------------------------------------
# FanDuel settled screenshot tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_seen_bet_ids():
    """Clear cross-screenshot dedup set between tests."""
    BOT._SEEN_FD_BET_IDS.clear()
    yield
    BOT._SEEN_FD_BET_IDS.clear()


def test_fanduel_settled_multi_card_parsing() -> None:
    raw = _read_fixture("fanduel_settled_multi.txt")
    bets = BOT._parse_fd_settled_cards(raw)

    assert len(bets) == 3

    # Card 1: Drexel
    b0 = bets[0]
    assert b0["platform"] == "FanDuel"
    assert b0["line"] == "Drexel -1.5"
    assert b0["game"] == "Drexel vs Towson"
    assert b0["odds"] == "-110"
    assert b0["wager"] == 1.0
    assert b0["result"] == "win"
    assert b0["payout"] == 1.91
    assert b0["profit"] == 0.91
    assert b0["date"] == "2026-02-19"
    assert b0["bet_id"] == "FD-ABC-001"

    # Card 2: Citadel
    b1 = bets[1]
    assert b1["line"] == "Citadel +5.5"
    assert b1["game"] == "The Citadel vs VMI"
    assert b1["odds"] == "-108"
    assert b1["wager"] == 1.0
    assert b1["result"] == "win"
    assert b1["bet_id"] == "FD-ABC-002"

    # Card 3: UTSA -- game name doesn't leak from card 1 or 2
    b2 = bets[2]
    assert b2["line"] == "UTSA +12.5"
    assert b2["game"] == "UTSA vs UTEP"
    assert b2["odds"] == "+100"
    assert b2["wager"] == 1.0
    assert b2["result"] == "win"
    assert b2["payout"] == 2.0
    assert b2["profit"] == 1.0
    assert b2["bet_id"] == "FD-ABC-003"


def test_fanduel_settled_void_detection() -> None:
    raw = _read_fixture("fanduel_settled_void.txt")
    bets = BOT._parse_fd_settled_cards(raw)

    assert len(bets) == 1
    bet = bets[0]
    assert bet["result"] == "void"
    assert bet["payout"] == bet["wager"]  # void: payout = wager
    assert bet["profit"] == 0.0
    assert bet["bet_id"] == "FD-VOID-001"


def test_fanduel_settled_context_detection() -> None:
    settled_text = "Some header\nWON ON FANDUEL\nDrexel -1.5\n$1.00"
    pending_text = "FanDuel\nSportsbook\nDrexel -1.5\n$1.00"
    assert BOT._detect_fd_settled(settled_text) is True
    assert BOT._detect_fd_settled(pending_text) is False

    returned_text = "RETURNED\nDrexel -1.5\n$1.00"
    assert BOT._detect_fd_settled(returned_text) is True


def test_fanduel_settled_bet_id_extraction() -> None:
    raw = _read_fixture("fanduel_settled_multi.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    ids = [b["bet_id"] for b in bets]
    assert ids == ["FD-ABC-001", "FD-ABC-002", "FD-ABC-003"]


def test_fanduel_settled_game_date_extraction() -> None:
    raw = _read_fixture("fanduel_settled_multi.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    assert all(b["date"] == "2026-02-19" for b in bets)


def test_fanduel_settled_does_not_break_pending() -> None:
    """Existing pending FanDuel fixture still parses correctly."""
    text = _read_fixture("fanduel_spread_single.txt")
    bets = BOT._parse_bet_slip_text(text, platform="FanDuel")

    assert len(bets) == 1
    bet = bets[0]
    assert bet["platform"] == "FanDuel"
    assert bet["line"] == "Providence +15.5"
    assert bet["wager"] == 1.25


def test_fanduel_settled_cross_screenshot_dedup() -> None:
    raw = _read_fixture("fanduel_settled_multi.txt")
    bets1 = BOT._parse_fd_settled_cards(raw)
    assert len(bets1) == 3

    # Same text again -- all BET IDs already seen
    bets2 = BOT._parse_fd_settled_cards(raw)
    assert len(bets2) == 0
