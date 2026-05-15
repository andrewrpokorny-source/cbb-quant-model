"""Regression tests for OCR bet-slip parsing using fixture text snapshots."""

from __future__ import annotations

from pathlib import Path
import importlib
import importlib.util
import logging
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
    """Prediction-file date should take precedence over PLACED date.

    The fixture has PLACED: 2/25/2026, but the mock prediction file dates the
    game to 2/24 -- verifying prediction lookup wins over PLACED.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pred_csv = data_dir / "predictions_20260224.csv"
    pred_csv.write_text(
        "Date/Time,Matchup,Spread,Pick,Conf,Raw Odds,Rest\n"
        "02/24 09:00 PM,Mississippi State Bulldogs @ Alabama Crimson Tide,"
        "-14.5,Alabama Crimson Tide -14.5,0.7,ALA -14.5,3\n"
    )
    monkeypatch.setattr(BOT, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(BOT, "DAILY_PREDICTIONS", str(data_dir / "daily_predictions.csv"))

    raw = _read_fixture("fanduel_finished_win.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    assert len(actual) >= 1
    # Prediction date (2/24) should win over PLACED date (2/25)
    assert actual[0]["date"] == "2026-02-24"


def test_fanduel_finished_date_falls_back_to_placed(tmp_path, monkeypatch) -> None:
    """Without prediction files or game date header, date comes from PLACED line."""
    # Point prediction lookup at an empty directory so it can't resolve a date
    monkeypatch.setattr(BOT, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(BOT, "DAILY_PREDICTIONS", str(tmp_path / "daily_predictions.csv"))

    raw = _read_fixture("fanduel_finished_win.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    assert len(actual) >= 1
    # No prediction file match, so date should fall back to the PLACED date
    assert actual[0]["date"] == "2026-02-25"


def test_fanduel_finished_multi_card() -> None:
    """Multi-card Finished screenshot should parse each card correctly.

    UTSA card in the fixture has WON/TOTAL WAGER labels swapped vs the dollar
    amounts (wager=$1.00 visually, but OCR ordered the amounts as $1.91 then
    $1.00). That's an OCR-corrupt reading that now skips as ambiguous_recalc
    rather than being silently "rescued" by recalculating from odds.
    """
    raw = _read_fixture("fanduel_finished_multi.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)
    skipped = [b for b in bets if b.get("_skipped")]

    by_id = {b["bet_id"]: b for b in actual if b.get("bet_id")}
    skipped_by_id = {b["bet_id"]: b for b in skipped if b.get("bet_id")}

    # Alabama -14.5 should be win (payout > wager, no recalc needed)
    assert "0/0084650/0000187" in by_id
    assert by_id["0/0084650/0000187"]["result"] == "win"
    assert by_id["0/0084650/0000187"]["line"] == "Alabama -14.5"

    # UTSA +4.5: payout <= wager after parse -> ambiguous_recalc skip
    assert "0/0084650/0000189" in skipped_by_id
    assert skipped_by_id["0/0084650/0000189"]["_skip_reason"] == "ambiguous_recalc"
    assert skipped_by_id["0/0084650/0000189"]["line"] == "UTSA +4.5"

    # Only Alabama parses fully; UTSA is skipped, Cleveland State is truncated
    assert len(actual) == 1, f"Expected 1 parseable card, got {len(actual)}"


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


# ---------------------------------------------------------------------------
# FanDuel MONEYLINE format tests (no spread number; team and odds split lines)
# ---------------------------------------------------------------------------


def test_fanduel_moneyline_multi_card_parsing() -> None:
    """ML cards: team appears 2 lines above MONEYLINE marker; odds on the line between."""
    raw = _read_fixture("fanduel_settled_moneyline.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    assert len(actual) == 2

    win = actual[0]
    assert win["bet_id"] == "0/0084650/0000230"
    assert win["bet_type"] == "moneyline"
    assert win["line"] == "Tampa Bay Rays ML"
    assert win["odds"] == "-120"
    assert win["wager"] == 0.5
    assert win["result"] == "win"
    assert win["payout"] == 0.92
    assert win["profit"] == 0.42
    assert win["game"] == "Toronto Blue Jays vs Tampa Bay Rays"
    assert win["date"] == "2026-05-04"

    loss = actual[1]
    assert loss["bet_id"] == "0/0084650/0000232"
    assert loss["bet_type"] == "moneyline"
    assert loss["line"] == "Los Angeles Angels ML"
    assert loss["odds"] == "-158"
    assert loss["wager"] == 0.5
    # FanDuel "RETURNED" with $0.00 payout reflects a losing bet, not a refund.
    assert loss["result"] == "loss"
    assert loss["profit"] == -0.5


def test_fanduel_moneyline_underdog_plus_odds() -> None:
    """Plus-odds ML team should parse and game come from matchup teams."""
    card = (
        "Atlanta Braves\n+126\nMONEYLINE\n"
        "Atlanta Braves (J Ritc...\n1 0 0 0 0 3 0 0 0\n4\n"
        "Seattle Mariners (L Gil...\n0 0 0 0 0 5 0 0 0\n5\n"
        "$0.50\n$0.00\nTOTAL WAGER\nRETURNED\n"
        "BET ID: ML-PLUS-001\nPLACED: 5/4/2026 2:34PM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    bet = actual[0]
    assert bet["bet_type"] == "moneyline"
    assert bet["line"] == "Atlanta Braves ML"
    assert bet["odds"] == "+126"
    assert bet["game"] == "Atlanta Braves vs Seattle Mariners"
    assert bet["result"] == "loss"


def test_fanduel_spread_still_parses_after_ml_changes() -> None:
    """Existing spread fixtures must keep parsing as bet_type=spread."""
    raw = _read_fixture("fanduel_settled_multi.txt")
    bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(bets)

    assert len(actual) == 3
    for b in actual:
        assert b["bet_type"] == "spread"


def test_fanduel_moneyline_team_with_ampersand() -> None:
    """ML team containing '&' (e.g. Texas A&M) must be accepted."""
    card = (
        "Texas A&M\n-140\nMONEYLINE\n"
        "Texas A&M (M Bay...\n2 0 0 0 0 1 0 0 0\n3\n"
        "Mississippi St. (P Pe...\n0 0 0 0 0 0 0 0 0\n0\n"
        "$0.50\n$0.86\nTOTAL WAGER\nWON ON FANDUEL\n"
        "BET ID: ML-AMP-001\nPLACED: 5/5/2026 7:00PM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    bet = actual[0]
    assert bet["bet_type"] == "moneyline"
    assert bet["line"] == "Texas A&M ML"
    assert bet["game"] == "Texas A&M vs Mississippi St."


def test_fanduel_moneyline_qualified_team_miami_oh() -> None:
    """Closed-paren qualifier '(OH)' must survive fallback game detection."""
    card = (
        "Toledo\n-150\nMONEYLINE\n"
        "Miami (OH) (B Smi...\n0 0 0 0 0 0 0 0 0\n0\n"
        "Toledo (R Joh...\n2 0 0 1 0 0 0 0 0\n3\n"
        "$0.50\n$0.83\nTOTAL WAGER\nWON ON FANDUEL\n"
        "BET ID: ML-OH-001\nPLACED: 5/5/2026 7:00PM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    bet = actual[0]
    assert bet["bet_type"] == "moneyline"
    assert bet["line"] == "Toledo ML"
    assert bet["game"] == "Miami (OH) vs Toledo"


def test_fanduel_moneyline_reversed_odds_team_layout() -> None:
    """Some FanDuel ML cards render '<odds>\\n<team>\\nMONEYLINE' instead of the
    standard '<team>\\n<odds>\\nMONEYLINE'. Both layouts must parse."""
    card = (
        "-112\nSan Diego Padres\nMONEYLINE\n"
        "San Diego Padres (B...\n0 0 0 1 0 0 2 2 0\n5\n"
        "San Francisco Giants (D...\n0 0 0 1 0 0 0 0 1\n2\n"
        "$0.50\n$0.95\nTOTAL WAGER\nWON ON FANDUEL\n"
        "BET ID: ML-REV-001\nPLACED: 5/6/2026 2:22PM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    bet = actual[0]
    assert bet["bet_type"] == "moneyline"
    assert bet["line"] == "San Diego Padres ML"
    assert bet["odds"] == "-112"
    assert bet["game"] == "San Diego Padres vs San Francisco Giants"
    assert bet["result"] == "win"


def test_fanduel_moneyline_noise_line_between_odds_and_marker() -> None:
    """A spurious single-character OCR line between odds and MONEYLINE must
    not break ML parsing. Reproduces the 5/9 Colorado Rockies failure where
    'R' sat at i-1 above MONEYLINE, pushing the team/odds anchor to i-3/i-2.
    """
    card = (
        "Colorado Rockies\n+184\nR\nMONEYLINE\n"
        "Colorado Rockies (C...\n5 0 0 1 0 0 0 2\n9\n"
        "Philadelphia Phillies (J...\n0 0 0 2 0 5 0 0 0\n7\n"
        "$0.50\n$1.42\nTOTAL WAGER\nWON ON FANDUEL\n"
        "BET ID: ML-NOISE-001\nPLACED: 5/8/2026 9:14AM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    bet = actual[0]
    assert bet["bet_type"] == "moneyline"
    assert bet["line"] == "Colorado Rockies ML"
    assert bet["odds"] == "+184"
    assert bet["league"] == "mlb"
    assert bet["result"] == "win"


def test_fanduel_moneyline_noise_line_with_garbled_header() -> None:
    """Replays the literal 5/9 Colorado Rockies OCR -- garbled scroll-clipped
    header text above the first card AND a 'R' noise line between odds and
    MONEYLINE. The wider scan window must not be fooled by header lines that
    happen to satisfy team_re into picking the wrong team.
    """
    card = (
        "9:00\nMy Bets\nOpen\nSettled\nSaved\n"
        "ULTIU. VIVVUTUVVIVUVVLTT\nVIUILULU\n.....\n"
        "FANDUEL\nSPORTSBOOK\n"
        "Colorado Rockies\n+184\nR\nMONEYLINE\n"
        "Colorado Rockies (C...\n5 0 0 1 0 0 0 2\n9\n"
        "Philadelphia Phillies (J...\n0 0 0 2 0 5 0 0 0\n7\n"
        "$0.50\n$1.42\nTOTAL WAGER\nWON ON FANDUEL\n"
        "BET ID: ML-NOISE-002\nPLACED: 5/8/2026 9:14AM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    bet = actual[0]
    assert bet["bet_type"] == "moneyline"
    assert bet["line"] == "Colorado Rockies ML"
    assert bet["odds"] == "+184"


def test_fanduel_moneyline_reversed_layout_with_noise_line() -> None:
    """Reversed odds/team layout combined with a noise line between team and
    MONEYLINE. Real team is at odds_idx+1; verify we don't pick a stale
    team-shaped line at odds_idx-1.
    """
    card = (
        "FANDUEL\nSPORTSBOOK\n"
        "+118\nTampa Bay Rays\nT\nMONEYLINE\n"
        "Tampa Bay Rays (J Sc...\n0 0 1 0 0 0 0 0 0\n1\n"
        "Boston Red Sox (C Ea...\n0 0 1 1 0 0 0 0 0\n2\n"
        "$0.50\n$0.00\nTOTAL WAGER\nRETURNED\n"
        "BET ID: ML-REV-NOISE-001\nPLACED: 5/8/2026 9:14AM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    bet = actual[0]
    assert bet["bet_type"] == "moneyline"
    assert bet["line"] == "Tampa Bay Rays ML"
    assert bet["odds"] == "+118"
    assert bet["league"] == "mlb"


def test_fanduel_moneyline_mlb_team_sets_league_mlb() -> None:
    """ML cards on a known MLB franchise should tag league=mlb."""
    card = (
        "Tampa Bay Rays\n-120\nMONEYLINE\n"
        "Toronto Blue Jays (EL...\n0 0 1 0 0 0 0 0 0\n1\n"
        "Tampa Bay Rays (N M...\n3 0 0 0 0 2 0 0 0\n5\n"
        "$0.50\n$0.92\nTOTAL WAGER\nWON ON FANDUEL\n"
        "BET ID: ML-MLB-001\nPLACED: 5/4/2026 2:34PM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    assert actual[0]["league"] == "mlb"
    assert actual[0]["line"] == "Tampa Bay Rays ML"


def test_fanduel_moneyline_cbb_team_no_league_tag() -> None:
    """ML cards on a non-MLB, non-(W) team must NOT auto-tag league=mlb."""
    card = (
        "Houston\n-150\nMONEYLINE\n"
        "Houston\n78\n"
        "Memphis\n65\n"
        "$0.50\n$0.83\nTOTAL WAGER\nWON ON FANDUEL\n"
        "BET ID: ML-CBB-001\nPLACED: 3/15/2026 7:00PM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    bet = actual[0]
    assert bet["bet_type"] == "moneyline"
    assert bet["line"] == "Houston ML"
    # "Houston" alone is not an MLB franchise name ("Houston Astros" is),
    # so league should stay empty for the CBB-style bet.
    assert bet["league"] == ""


def test_fanduel_moneyline_w_qualifier_sets_womens_league() -> None:
    """'(W)' qualifier in fallback teams must trigger womens league detection."""
    card = (
        "South Carolina (W)\n-220\nMONEYLINE\n"
        "South Carolina (W)\n0 0 0 0 0 0 0 0 0\n78\n"
        "LSU (W)\n0 0 0 0 0 0 0 0 0\n65\n"
        "$0.50\n$0.73\nTOTAL WAGER\nWON ON FANDUEL\n"
        "BET ID: ML-W-001\nPLACED: 5/5/2026 7:00PM ET\n"
    )
    bets = BOT._parse_fd_settled_cards(card)
    actual = _actual_bets(bets)

    assert len(actual) == 1
    bet = actual[0]
    assert bet["bet_type"] == "moneyline"
    assert bet["league"] == "womens"
    assert bet["line"] == "South Carolina ML"
    assert bet["game"] == "South Carolina vs LSU"


# ---------------------------------------------------------------------------
# _ocr_sort_key tests
# ---------------------------------------------------------------------------

def test_ocr_sort_key_reading_order() -> None:
    """Synthetic two-column OCR results should sort top-to-bottom, left-to-right."""
    # (text, confidence, (x, y)) -- y=1 is top in macOS coords
    results = [
        ("left-col-top", 0.99, (0.05, 0.90)),
        ("right-col-top", 0.99, (0.55, 0.90)),
        ("left-col-mid", 0.99, (0.05, 0.50)),
        ("right-col-mid", 0.99, (0.55, 0.50)),
        ("left-col-bot", 0.99, (0.05, 0.10)),
        ("right-col-bot", 0.99, (0.55, 0.10)),
    ]
    sorted_results = sorted(results, key=BOT._ocr_sort_key)
    texts = [r[0] for r in sorted_results]
    assert texts == [
        "left-col-top", "right-col-top",
        "left-col-mid", "right-col-mid",
        "left-col-bot", "right-col-bot",
    ]


def test_ocr_sort_key_identical_y_tiebreak() -> None:
    """Items with identical y coordinates should sort left-to-right by x."""
    results = [
        ("right", 0.99, (0.80, 0.50)),
        ("left", 0.99, (0.10, 0.50)),
        ("middle", 0.99, (0.45, 0.50)),
    ]
    sorted_results = sorted(results, key=BOT._ocr_sort_key)
    texts = [r[0] for r in sorted_results]
    assert texts == ["left", "middle", "right"]


# ---------------------------------------------------------------------------
# RETURNED detection tests
# ---------------------------------------------------------------------------

def test_returned_with_zero_payout_is_loss() -> None:
    """Card with $0.00 payout and RETURNED should classify as loss."""
    card_text = "\n".join([
        "Spread",
        "Michigan +3.5",
        "-110",
        "$1.00",
        "$0.00",
        "RETURNED",
        "BET ID: 0/1234567/0000001",
        "PLACED: 2/25/2026 7:00 PM",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    actual = _actual_bets(bets)
    assert len(actual) == 1
    assert actual[0]["result"] == "loss"


def test_returned_with_refund_is_void() -> None:
    """Card with equal wager/payout and RETURNED should classify as void."""
    card_text = "\n".join([
        "Spread",
        "Michigan +3.5",
        "-110",
        "$1.00",
        "$1.00",
        "RETURNED",
        "BET ID: 0/1234567/0000002",
        "PLACED: 2/25/2026 7:00 PM",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    actual = _actual_bets(bets)
    assert len(actual) == 1
    assert actual[0]["result"] == "void"


# ---------------------------------------------------------------------------
# PLACED date validation test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Women's (W) suffix parsing tests
# ---------------------------------------------------------------------------


def test_fanduel_womens_pending_spread_parsing() -> None:
    """FanDuel pending bets with (W) suffix should parse spread and detect league."""
    text = _read_fixture("fanduel_womens_pending.txt")
    bets = BOT._parse_fd_blocks(text)

    assert len(bets) == 3

    # Villanova (W) -7.5
    b0 = bets[0]
    assert b0["platform"] == "FanDuel"
    assert b0["line"] == "Villanova -7.5"
    assert b0["odds"] == "-114"
    assert b0["wager"] == 3.0
    assert b0["league"] == "womens"
    assert b0["game"] == "Seton Hall vs Villanova"

    # Chattanooga (W) -6.5
    b1 = bets[1]
    assert b1["line"] == "Chattanooga -6.5"
    assert b1["odds"] == "-112"
    assert b1["wager"] == 3.0
    assert b1["league"] == "womens"
    assert b1["game"] == "Samford vs Chattanooga"

    # Colgate -1.5 (men's, no (W) suffix)
    b2 = bets[2]
    assert b2["line"] == "Colgate -1.5"
    assert b2["odds"] == "-115"
    assert b2["wager"] == 1.5
    assert b2["league"] == ""
    assert b2["game"] == "Colgate vs Lehigh"


def test_fanduel_womens_settled_parsing() -> None:
    """FanDuel settled screenshot with (W) suffix should parse all cards."""
    raw = _read_fixture("fanduel_womens_settled.txt")
    all_bets = BOT._parse_fd_settled_cards(raw)
    actual = _actual_bets(all_bets)

    by_id = {b["bet_id"]: b for b in actual if b.get("bet_id")}

    # Villanova (W) -7.5: won
    assert "0/0084650/0000207" in by_id
    v = by_id["0/0084650/0000207"]
    assert v["line"] == "Villanova -7.5"
    assert v["result"] == "win"
    assert v["wager"] == 3.0
    assert v["payout"] == 5.63
    assert v["profit"] == 2.63
    assert v["league"] == "womens"

    # Colgate -1.5: loss (men's)
    assert "0/0084650/0000205" in by_id
    c = by_id["0/0084650/0000205"]
    assert c["line"] == "Colgate -1.5"
    assert c["result"] == "loss"
    assert c["wager"] == 1.5
    assert c["league"] == ""

    # Chattanooga (W) -6.5: loss
    assert "0/0084650/0000206" in by_id
    ch = by_id["0/0084650/0000206"]
    assert ch["line"] == "Chattanooga -6.5"
    assert ch["result"] == "loss"
    assert ch["wager"] == 3.0
    assert ch["league"] == "womens"


def test_fanduel_womens_league_from_matchup_only() -> None:
    """league should be detected from matchup (W) even if spread line lacks it."""
    # Simulate OCR dropping (W) from spread line but keeping it on matchup
    text = (
        "Villanova -7.5\n"
        "-114\n"
        "Seton Hall (W) @ Villanova (W)\n"
        "$3.00\n"
    )
    bets = BOT._parse_fd_blocks(text)
    assert len(bets) == 1
    assert bets[0]["league"] == "womens"
    assert bets[0]["line"] == "Villanova -7.5"
    assert bets[0]["game"] == "Seton Hall vs Villanova"


def test_fanduel_womens_w_stripped_from_game_name() -> None:
    """(W) suffix should be stripped from game names in both paths."""
    text = _read_fixture("fanduel_womens_pending.txt")
    bets = BOT._parse_fd_blocks(text)

    for b in bets:
        assert "(W)" not in b.get("game", "")
        assert "(W)" not in b.get("line", "")


def test_placed_date_validation_rejects_out_of_range() -> None:
    """Out-of-range PLACED date (month 13, day 99) should produce no game_date."""
    card_text = "\n".join([
        "Spread",
        "Faketestuniv +3.5",
        "-110",
        "$1.00",
        "$2.00",
        "WON ON FANDUEL",
        "BET ID: 0/1234567/0000003",
        "PLACED: 13/99/2026 7:00 PM",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    actual = _actual_bets(bets)
    assert len(actual) == 1
    assert actual[0]["date"] == ""


# ---------------------------------------------------------------------------
# Hardened OCR confidence gating (RETURNED ambiguity + skip feedback)
# ---------------------------------------------------------------------------


def test_returned_with_zero_payout_and_one_in_range_amount_is_loss() -> None:
    """$0.00 payout fails the in-range filter; explicit $0.00 substring confirms loss."""
    card_text = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "Houston Astros ML",
        "+116",
        "MONEYLINE",
        "$0.50",
        "$0.00",
        "TOTAL WAGER",
        "RETURNED",
        "BET ID: 0/0084650/9999252",
        "PLACED: 5/10/2026 6:32PM ET",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    actual = _actual_bets(bets)
    assert len(actual) == 1
    assert actual[0]["result"] == "loss"
    assert actual[0]["payout"] == 0.0
    assert actual[0]["wager"] == 0.5


def test_returned_with_one_amount_and_no_zero_marker_is_ambiguous() -> None:
    """Missing payout amount could equally be a dropped winner -- skip, don't lock in loss."""
    card_text = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "Houston Astros ML",
        "+116",
        "MONEYLINE",
        "$0.50",
        "TOTAL WAGER",
        "RETURNED",
        "BET ID: 0/0084650/9999253",
        "PLACED: 5/10/2026 6:32PM ET",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    skipped = [b for b in bets if b.get("_skipped")]
    actual = _actual_bets(bets)

    assert actual == []
    assert len(skipped) == 1
    assert skipped[0]["_skip_reason"] == "ambiguous_returned"
    assert skipped[0]["bet_id"] == "0/0084650/9999253"
    # BET ID must not be burned -- a cleaner re-screenshot should re-parse
    assert "0/0084650/9999253" not in BOT._SEEN_FD_BET_IDS


def test_ambiguous_returned_does_not_burn_bet_id() -> None:
    """A re-screenshot with full data must still parse after an ambiguous skip."""
    ambiguous_card = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "Minnesota Twins ML",
        "+100",
        "MONEYLINE",
        "$0.50",
        "TOTAL WAGER",
        "RETURNED",
        "BET ID: 0/0084650/9999254",
        "PLACED: 5/10/2026 6:32PM ET",
    ])
    clean_card = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "Minnesota Twins ML",
        "+100",
        "MONEYLINE",
        "$0.50",
        "$0.00",
        "TOTAL WAGER",
        "RETURNED",
        "BET ID: 0/0084650/9999254",
        "PLACED: 5/10/2026 6:32PM ET",
    ])
    bets1 = BOT._parse_fd_settled_cards(ambiguous_card)
    assert _actual_bets(bets1) == []

    bets2 = BOT._parse_fd_settled_cards(clean_card)
    actual2 = _actual_bets(bets2)
    assert len(actual2) == 1
    assert actual2[0]["result"] == "loss"
    assert actual2[0]["bet_id"] == "0/0084650/9999254"


def test_incomplete_skip_includes_line_for_user_feedback() -> None:
    """Skip dict must expose the parsed team line so empty-bet_id cards aren't silent."""
    card_text = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "Minnesota Twins",
        "+100",
        "MONEYLINE",
        "Minnesota Twins ( Ryan) @ Cleveland Guardia...",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    skipped = [b for b in bets if b.get("_skipped")]
    assert len(skipped) == 1
    s = skipped[0]
    assert s["line"] == "Minnesota Twins ML"
    assert s["team"] == "Minnesota Twins"


def test_returned_with_zero_in_range_amounts_and_no_marker_is_ambiguous() -> None:
    """Zero usable amounts and no $0.00 marker -- can't even confirm a wager was placed."""
    card_text = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "Houston Astros ML",
        "+116",
        "MONEYLINE",
        "RETURNED",
        "BET ID: 0/0084650/9999260",
        "PLACED: 5/10/2026 6:32PM ET",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    skipped = [b for b in bets if b.get("_skipped")]
    assert skipped and skipped[0]["_skip_reason"] == "ambiguous_returned"


def test_returned_with_bare_zero_no_dollar_sign_is_ambiguous() -> None:
    """Bare '0.00' without a $ prefix is not enough -- the marker contract requires $0.00."""
    card_text = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "Houston Astros ML",
        "+116",
        "MONEYLINE",
        "$0.50",
        "0.00",
        "TOTAL WAGER",
        "RETURNED",
        "BET ID: 0/0084650/9999261",
        "PLACED: 5/10/2026 6:32PM ET",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    skipped = [b for b in bets if b.get("_skipped")]
    actual = _actual_bets(bets)
    assert actual == []
    assert skipped and skipped[0]["_skip_reason"] == "ambiguous_returned"


def test_returned_with_zero_dollar_zero_single_decimal_is_loss() -> None:
    """$0.0 (one trailing digit) is also a valid zero-payout marker."""
    card_text = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "Houston Astros ML",
        "+116",
        "MONEYLINE",
        "$0.50",
        "$0.0",
        "TOTAL WAGER",
        "RETURNED",
        "BET ID: 0/0084650/9999262",
        "PLACED: 5/10/2026 6:32PM ET",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    actual = _actual_bets(bets)
    assert len(actual) == 1
    assert actual[0]["result"] == "loss"


def test_won_card_with_corrupted_payout_amount_is_ambiguous() -> None:
    """Reproduce bet 0/0084650/0000254: OCR misread "$1.17" as "$1:17" (colon for period).

    The dollar regex stops at the colon, so:
      - parsed wager = $1 (from "$1:17")
      - parsed payout = $0.50 (the actual wager)
    The win-sanity recalc then "rescues" the bad payout from odds, producing a
    confidently-wrong ledger row ($1.00 stake / $2.34 payout for a bet that was
    actually $0.50 / $1.17). When the recalc fires we know one amount is
    already corrupt -- skip the card so the user can re-screenshot.
    """
    card_text = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "+134",
        "Washington Nationals",
        "MONEYLINE",
        "Washington Nationals...",
        "Cincinnati Reds (N Lo...",
        "$1:17",
        "$0.50",
        "TOTAL WAGER",
        "WON ON FANDUEL",
        "BET ID: 0/0084650/9999254",
        "PLACED: 5/12/2026 10:35PM ET",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    skipped = [b for b in bets if b.get("_skipped")]
    actual = _actual_bets(bets)

    assert actual == []
    assert len(skipped) == 1
    assert skipped[0]["_skip_reason"] == "ambiguous_recalc"
    assert skipped[0]["bet_id"] == "0/0084650/9999254"
    # BET ID must not be burned -- a cleaner re-screenshot should re-parse
    assert "0/0084650/9999254" not in BOT._SEEN_FD_BET_IDS


def test_ambiguous_recalc_does_not_burn_bet_id() -> None:
    """A clean re-screenshot must still parse after an ambiguous_recalc skip."""
    corrupted = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "+134",
        "Washington Nationals",
        "MONEYLINE",
        "Washington Nationals...",
        "Cincinnati Reds (N Lo...",
        "$1:17",
        "$0.50",
        "TOTAL WAGER",
        "WON ON FANDUEL",
        "BET ID: 0/0084650/9999264",
        "PLACED: 5/12/2026 10:35PM ET",
    ])
    clean = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "+134",
        "Washington Nationals",
        "MONEYLINE",
        "Washington Nationals...",
        "Cincinnati Reds (N Lo...",
        "$0.50",
        "$1.17",
        "TOTAL WAGER",
        "WON ON FANDUEL",
        "BET ID: 0/0084650/9999264",
        "PLACED: 5/12/2026 10:35PM ET",
    ])
    bets1 = BOT._parse_fd_settled_cards(corrupted)
    assert _actual_bets(bets1) == []

    bets2 = BOT._parse_fd_settled_cards(clean)
    actual2 = _actual_bets(bets2)
    assert len(actual2) == 1
    assert actual2[0]["result"] == "win"
    assert actual2[0]["wager"] == 0.5
    assert actual2[0]["payout"] == 1.17
    assert actual2[0]["bet_id"] == "0/0084650/9999264"


def test_normal_winning_card_still_parses_after_recalc_change() -> None:
    """Regression: an ordinary winning card (payout > wager) must keep parsing."""
    card_text = "\n".join([
        "FANDUEL",
        "SPORTSBOOK",
        "Boston Red Sox",
        "-132",
        "MONEYLINE",
        "Philadelphia Phillies (...",
        "Boston Red Sox (S Gr...",
        "$0.50",
        "$0.88",
        "TOTAL WAGER",
        "WON ON FANDUEL",
        "BET ID: 0/0084650/9999255",
        "PLACED: 5/12/2026 10:35PM ET",
    ])
    bets = BOT._parse_fd_settled_cards(card_text)
    actual = _actual_bets(bets)
    assert len(actual) == 1
    assert actual[0]["result"] == "win"
    assert actual[0]["wager"] == 0.5
    assert actual[0]["payout"] == 0.88


# ---------------------------------------------------------------------------
# Skip-feedback helper tests (extracted from handle_photo for testability)
# ---------------------------------------------------------------------------


def test_format_missed_card_label_prefers_line_over_team() -> None:
    label = BOT._format_missed_card_label(
        {"line": "Drexel -1.5", "team": "Drexel", "game": "Drexel vs Towson"}
    )
    assert label.startswith("Drexel -1.5")
    assert "Drexel vs Towson" not in label  # game not appended once line is present


def test_format_missed_card_label_falls_back_to_team_then_game() -> None:
    assert BOT._format_missed_card_label({"team": "Drexel"}) == "Drexel"
    assert BOT._format_missed_card_label({"game": "Drexel vs Towson"}) == "Drexel vs Towson"


def test_format_missed_card_label_handles_all_empty_fields() -> None:
    assert BOT._format_missed_card_label({}) == "card with no identifying fields"
    assert (
        BOT._format_missed_card_label({"line": "", "team": "", "bet_id": "", "wager": 0})
        == "card with no identifying fields"
    )


def test_format_missed_card_label_hides_internal_result_values() -> None:
    """`pending` and `ambiguous_returned` are internal sentinels -- don't surface them."""
    pending = BOT._format_missed_card_label({"line": "X ML", "result": "pending"})
    ambiguous = BOT._format_missed_card_label({"line": "X ML", "result": "ambiguous_returned"})
    assert "PENDING" not in pending
    assert "AMBIGUOUS_RETURNED" not in ambiguous


def test_build_skip_feedback_emits_one_line_per_needs_review_card() -> None:
    skipped = [
        {"_skip_reason": "incomplete", "line": "Drexel -1.5"},
        {"_skip_reason": "ambiguous_returned", "line": "Houston Astros ML",
         "bet_id": "BID-9", "wager": 0.5},
    ]
    msgs = BOT._build_skip_feedback_messages(skipped, has_actual_settled=False)
    missed = [m for m in msgs if m.startswith("Missed [")]
    assert len(missed) == 2
    assert any("Drexel -1.5" in m and "[incomplete]" in m for m in missed)
    assert any("Houston Astros ML" in m and "ambiguous" in m for m in missed)


def test_build_skip_feedback_uses_no_identifying_fields_label_when_empty() -> None:
    msgs = BOT._build_skip_feedback_messages(
        [{"_skip_reason": "incomplete"}], has_actual_settled=False
    )
    assert any("card with no identifying fields" in m for m in msgs)


def test_build_skip_feedback_collapses_all_dedup_only_into_summary() -> None:
    msgs = BOT._build_skip_feedback_messages(
        [
            {"_skip_reason": "dedup", "bet_id": "A"},
            {"_skip_reason": "dedup", "bet_id": "B"},
        ],
        has_actual_settled=False,
    )
    assert msgs == ["2 bet(s) already processed from a previous screenshot."]


def test_build_skip_feedback_emits_unsettled_footer() -> None:
    msgs = BOT._build_skip_feedback_messages(
        [{"_skip_reason": "unsettled", "line": "Drexel -1.5"}],
        has_actual_settled=False,
    )
    joined = "\n".join(msgs)
    assert "Drexel -1.5" in joined
    assert "open/unsettled" in joined


def test_build_skip_feedback_surfaces_unknown_skip_reasons(caplog) -> None:
    """An unrecognized _skip_reason must be logged AND shown to the user."""
    caplog.set_level(logging.ERROR, logger="telegram_bot")
    msgs = BOT._build_skip_feedback_messages(
        [{"_skip_reason": "future_reason_we_have_not_added_yet", "line": "X ML"}],
        has_actual_settled=False,
    )
    assert any("Unknown OCR _skip_reason" in r.getMessage() for r in caplog.records)
    assert any("X ML" in m for m in msgs)  # surfaced, not silently dropped


def test_build_skip_feedback_empty_input_returns_empty() -> None:
    assert BOT._build_skip_feedback_messages([], has_actual_settled=True) == []
    assert BOT._build_skip_feedback_messages([], has_actual_settled=False) == []
