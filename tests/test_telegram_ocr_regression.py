"""Regression tests for OCR bet-slip parsing using fixture text snapshots."""

from __future__ import annotations

from pathlib import Path
import importlib
import importlib.util
import sys
import types


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
