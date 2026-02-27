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


def _actual_bets(bets: list[dict]) -> list[dict]:
    """Filter out _skipped entries, returning only fully-parsed bets."""
    return [b for b in bets if not b.get("_skipped")]


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


def test_fanduel_settled_loss_detection() -> None:
    raw = _read_fixture("fanduel_settled_loss.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    assert len(bets) == 1
    bet = bets[0]
    assert bet["result"] == "loss"
    assert bet["payout"] == 0.0
    assert bet["profit"] == -1.0
    assert bet["bet_id"] == "FD-LOSS-001"


def test_fanduel_settled_context_detection() -> None:
    settled_text = "Some header\nWON ON FANDUEL\nDrexel -1.5\n$1.00"
    pending_text = "FanDuel\nSportsbook\nDrexel -1.5\n$1.00"
    assert BOT._detect_fd_settled(settled_text) is True
    assert BOT._detect_fd_settled(pending_text) is False

    returned_text = "FANDUEL\nSPORTSBOOK\nRETURNED\nDrexel -1.5\n$1.00"
    assert BOT._detect_fd_settled(returned_text) is True

    # "returned" without "sportsbook" should not trigger
    ambiguous_text = "Player returned to lineup\nDrexel -1.5\n$1.00"
    assert BOT._detect_fd_settled(ambiguous_text) is False

    # PLACED: line triggers settled detection even without result keywords
    placed_text = "$1.50\nBET ID: 0/001\n$0.00\nPLACED: 2/19/2026 8:23AM ET"
    assert BOT._detect_fd_settled(placed_text) is True


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
    assert len(_actual_bets(bets1)) == 3

    # Same text again -- all BET IDs already seen
    bets2 = BOT._parse_fd_settled_cards(raw)
    assert len(_actual_bets(bets2)) == 0


def test_fanduel_settled_card_without_bet_id() -> None:
    """Card with no BET ID line still parses successfully."""
    card = (
        "FANDUEL\nSPORTSBOOK\nWON ON FANDUEL\n"
        "Drexel -1.5\nDrexel @ Towson\n$1.00\n-110\n$1.91\n"
        "FEB 19, 7:00PM ET"
    )
    bets = BOT._parse_fd_settled_cards(card)
    assert len(bets) == 1
    assert bets[0]["bet_id"] == ""
    assert bets[0]["line"] == "Drexel -1.5"


def test_fanduel_settled_different_ids_second_screenshot() -> None:
    """Different BET IDs in a second screenshot should be accepted."""
    raw1 = _read_fixture("fanduel_settled_multi.txt")
    bets1 = BOT._parse_fd_settled_cards(raw1)
    assert len(bets1) == 3

    raw2 = _read_fixture("fanduel_settled_void.txt")
    bets2 = BOT._parse_fd_settled_cards(raw2)
    assert len(bets2) == 1  # FD-VOID-001 is a different ID, should not be blocked


def test_fanduel_settled_real_ocr_parsing() -> None:
    """Real OCR output with PLACED lines parses complete cards correctly."""
    raw = _read_fixture("fanduel_settled_real_ocr.txt")
    all_bets = BOT._parse_fd_settled_cards(raw)
    bets = _actual_bets(all_bets)

    # Card 1 (top of scroll) is truncated -- no team/spread visible -> skipped
    # Card 4 (UTSA, bottom of scroll) has no BET ID -> skipped
    # Cards 2 and 3 should parse
    assert len(bets) == 2

    b0 = bets[0]
    assert b0["platform"] == "FanDuel"
    assert b0["line"] == "Drexel -1.5"
    assert b0["game"] == "Drexel vs Northeastern"
    assert b0["odds"] == "+102"
    assert b0["wager"] == 1.5
    assert b0["result"] == "win"
    assert b0["payout"] == 3.03
    assert b0["profit"] == 1.53
    assert b0["bet_id"] == "0/0084650/0000183"

    b1 = bets[1]
    assert b1["line"] == "Citadel +10.5"
    assert b1["game"] == "Samford vs Citadel"
    assert b1["odds"] == "-114"
    assert b1["wager"] == 2.0
    assert b1["result"] == "win"
    assert b1["payout"] == 3.75
    assert b1["profit"] == 1.75
    assert b1["bet_id"] == "0/0084650/0000182"


def test_fanduel_settled_incomplete_card_no_burn() -> None:
    """Truncated card should not burn BET ID, allowing retry from next screenshot."""
    # First screenshot: truncated card (no team/spread) followed by complete card
    raw = (
        "$1.50\nTOTAL WAGER\nBET ID: TEST-NOBURN\n$0.00\nRETURNED\n"
        "PLACED: 2/19/2026 8:23AM ET\n"
        "Drexel -1.5\nDrexel @ Northeastern\n$1.50\n"
        "BET ID: TEST-NOBURN\n+102\nFEB 19, 7:04PM ET\n$3.03\n"
        "WON ON FANDUEL\nPLACED: 2/19/2026 8:24AM ET\n"
    )
    all_bets = BOT._parse_fd_settled_cards(raw)
    bets = _actual_bets(all_bets)

    # First chunk is incomplete (no team/spread) -> skip dict, BET ID not burned
    # Second chunk has same BET ID with complete data -> parses successfully
    assert len(bets) == 1
    assert bets[0]["bet_id"] == "TEST-NOBURN"
    assert bets[0]["line"] == "Drexel -1.5"


def test_fanduel_settled_skip_entries_have_details() -> None:
    """Incomplete skip entries preserve BET ID, wager, and result for user feedback."""
    raw = _read_fixture("fanduel_settled_real_ocr.txt")
    all_bets = BOT._parse_fd_settled_cards(raw)
    skipped = [b for b in all_bets if b.get("_skipped")]
    incomplete = [s for s in skipped if s.get("_skip_reason") == "incomplete"]

    assert len(incomplete) >= 1
    for s in incomplete:
        assert s["_settled"] is True
        assert "bet_id" in s
        assert "wager" in s
        assert "result" in s


# ---------------------------------------------------------------------------
# FanDuel "Finished" format tests (scores visible, no WON/LOST banners)
# ---------------------------------------------------------------------------


def test_fanduel_finished_win_detection() -> None:
    """Finished format with payout > wager should detect as win."""
    raw = _read_fixture("fanduel_finished_win.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    assert len(actual) >= 1
    bet = actual[0]
    assert bet["result"] == "win"
    assert bet["line"] == "Alabama -14.5"
    assert bet["wager"] == 1.5
    assert bet["odds"] == "-110"
    assert bet["bet_id"] == "0/0084650/0000187"
    assert bet["profit"] > 0


def test_fanduel_finished_loss_detection() -> None:
    """$0.00 RETURNED with Finished should detect as loss, not void."""
    raw = _read_fixture("fanduel_finished_loss.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    assert len(actual) >= 1
    bet = actual[0]
    assert bet["result"] == "loss"
    assert bet["wager"] == 1.5
    assert bet["profit"] < 0
    assert bet["bet_id"] == "0/0084650/0000188"


def test_fanduel_finished_game_from_score_lines() -> None:
    """Finished format should extract game from team lines next to scores."""
    raw = _read_fixture("fanduel_finished_win.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    assert len(actual) >= 1
    assert actual[0]["game"]  # non-empty game name
    assert "Mississippi State" in actual[0]["game"] or "Alabama" in actual[0]["game"]


def test_fanduel_finished_date_from_predictions(tmp_path, monkeypatch) -> None:
    """Game date should resolve from prediction files, not PLACED date."""
    # Create a mock predictions file with Alabama game on 2/25
    pred_csv = tmp_path / "predictions_20260225.csv"
    pred_csv.write_text(
        "Date/Time,Matchup,Spread,Pick,Conf,Raw Odds,Rest\n"
        "02/25 09:00 PM,Mississippi State Bulldogs @ Alabama Crimson Tide,"
        "-14.5,Alabama Crimson Tide -14.5,0.7,ALA -14.5,3\n"
    )
    monkeypatch.setattr(BOT, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(BOT, "DAILY_PREDICTIONS", str(tmp_path / "daily_predictions.csv"))

    raw = _read_fixture("fanduel_finished_win.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    assert len(actual) >= 1
    assert actual[0]["date"] == "2026-02-25"


def test_fanduel_finished_date_falls_back_to_placed() -> None:
    """Without prediction files or game date header, date comes from PLACED line."""
    raw = _read_fixture("fanduel_finished_win.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    assert len(actual) >= 1
    # No "FEB 25, ..." game date header and no prediction file match,
    # so date should fall back to the PLACED date
    assert actual[0]["date"] == "2026-02-25"


def test_fanduel_finished_multi_card() -> None:
    """Multi-card Finished screenshot should parse each card correctly."""
    raw = _read_fixture("fanduel_finished_multi.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    by_id = {b["bet_id"]: b for b in actual if b.get("bet_id")}

    # Alabama -14.5 should be win
    assert "0/0084650/0000187" in by_id
    assert by_id["0/0084650/0000187"]["result"] == "win"
    assert by_id["0/0084650/0000187"]["line"] == "Alabama -14.5"

    # UTSA +4.5 should be win (spread header now captured with >= 0.5 threshold)
    assert "0/0084650/0000189" in by_id
    assert by_id["0/0084650/0000189"]["result"] == "win"
    assert by_id["0/0084650/0000189"]["line"] == "UTSA +4.5"

    # Cleveland State card is truncated at bottom of screenshot (no PLACED line),
    # so it may be incomplete -- but if parsed, must not be void
    if "0/0084650/0000188" in by_id:
        assert by_id["0/0084650/0000188"]["result"] != "void"


def test_fanduel_finished_is_detected_as_settled() -> None:
    """_detect_fd_settled should return True for Finished format."""
    raw = _read_fixture("fanduel_finished_win.txt")
    assert BOT._detect_fd_settled(raw) is True


def test_fanduel_settled_dedup_skip_entries() -> None:
    """Dedup skip entries have _skip_reason 'dedup' and the bet_id."""
    raw = _read_fixture("fanduel_settled_multi.txt")
    BOT._parse_fd_settled_cards(raw)  # first pass populates seen IDs

    bets2 = BOT._parse_fd_settled_cards(raw)  # second pass: all deduped
    assert len(_actual_bets(bets2)) == 0
    dedup = [b for b in bets2 if b.get("_skip_reason") == "dedup"]
    assert len(dedup) == 3
    for d in dedup:
        assert d["_skipped"] is True
        assert d["_settled"] is True
        assert d["bet_id"]  # non-empty
