"""Unit tests for settle_kalshi.py."""

import csv
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from settle_kalshi import (
    _bet_type_from_ticker,
    _clean_game_title,
    _existing_bet_ids,
    _find_pending_kalshi_match,
    _league_from_ticker,
    _parse_date,
    _parse_settlement,
    _read_existing_rows,
    _reconstruct_line,
    _result_from_profit,
    _read_sync_ts,
    _write_sync_ts,
    settle_to_csv,
    CSV_HEADERS,
    SYNC_STATE_FILE,
)


# ---------------------------------------------------------------------------
# _league_from_ticker
# ---------------------------------------------------------------------------

class TestLeagueFromTicker:
    def test_mens(self):
        assert _league_from_ticker("KXNCAAMBSPREAD-26JAN10TEXALA-ALA13") == "mens"

    def test_womens(self):
        assert _league_from_ticker("KXNCAAWBGAME-26MAR01OKLAMIZZ-MIZZ") == "womens"


# ---------------------------------------------------------------------------
# _bet_type_from_ticker
# ---------------------------------------------------------------------------

class TestBetTypeFromTicker:
    def test_spread(self):
        assert _bet_type_from_ticker("KXNCAAMBSPREAD-26JAN10TEXALA-ALA13") == "spread"

    def test_game(self):
        assert _bet_type_from_ticker("KXNCAAWBGAME-26MAR01OKLAMIZZ-MIZZ") == "game"

    def test_total(self):
        assert _bet_type_from_ticker("KXNCAAMBTOTAL-26JAN10-O150") == "total"

    def test_unknown(self):
        assert _bet_type_from_ticker("KXNCAAMB-UNKNOWN") == "other"


# ---------------------------------------------------------------------------
# _result_from_profit
# ---------------------------------------------------------------------------

class TestResultFromProfit:
    def test_win(self):
        assert _result_from_profit(1.50) == "win"

    def test_loss(self):
        assert _result_from_profit(-0.50) == "loss"

    def test_void_zero(self):
        assert _result_from_profit(0.0) == "void"


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_iso_utc(self):
        assert _parse_date("2026-01-10T22:30:00Z") == "2026-01-10"

    def test_iso_offset(self):
        assert _parse_date("2026-03-01T12:00:00+00:00") == "2026-03-01"

    def test_empty(self):
        assert _parse_date("") == ""

    def test_fallback_prefix(self):
        assert _parse_date("2026-01-10garbage") == "2026-01-10"


# ---------------------------------------------------------------------------
# _clean_game_title
# ---------------------------------------------------------------------------

class TestCleanGameTitle:
    def test_with_qualifier(self):
        assert _clean_game_title("Oklahoma at Missouri: Winner") == "Oklahoma at Missouri"

    def test_no_qualifier(self):
        assert _clean_game_title("Oklahoma at Missouri Winner?") == "Oklahoma at Missouri Winner?"


# ---------------------------------------------------------------------------
# _reconstruct_line
# ---------------------------------------------------------------------------

class TestReconstructLine:
    def test_yes_spread(self):
        title = "SMC at Pacific: Saint Mary's wins by over 8.5 Points"
        assert _reconstruct_line(title, "YES") == "Saint Mary's wins by over 8.5 Points"

    def test_no_spread_returns_opposite_team(self):
        title = "Duke at UNC: Duke -5.5"
        assert _reconstruct_line(title, "NO") == "UNC +5.5"

    def test_no_spread_home_favored(self):
        title = "Virginia Tech at Virginia: Virginia -3.5"
        assert _reconstruct_line(title, "NO") == "Virginia Tech +3.5"

    def test_no_spread_multi_colon_title(self):
        title = "NCAA: Duke at UNC: Duke -5.5"
        assert _reconstruct_line(title, "NO") == "UNC +5.5"

    def test_game_market_yes(self):
        # YES = home team (second team) on Kalshi game markets
        assert _reconstruct_line("Oklahoma at Missouri Winner?", "YES") == "Missouri ML"

    def test_game_market_no(self):
        # NO = away team (first team) on Kalshi game markets
        assert _reconstruct_line("Oklahoma at Missouri Winner?", "NO") == "Oklahoma ML"

    def test_game_market_colon_yes(self):
        assert _reconstruct_line("Duke vs UNC: Winner", "YES") == "UNC ML"

    def test_game_market_colon_no(self):
        assert _reconstruct_line("Duke vs UNC: Winner", "NO") == "Duke ML"

    def test_game_market_colon_at_yes(self):
        assert _reconstruct_line("Oklahoma at Missouri: Winner", "YES") == "Missouri ML"

    def test_game_market_colon_at_no(self):
        assert _reconstruct_line("Oklahoma at Missouri: Winner", "NO") == "Oklahoma ML"


# ---------------------------------------------------------------------------
# _parse_settlement -- single side
# ---------------------------------------------------------------------------

class TestParseSettlementSingleSide:
    def test_yes_win(self):
        s = {
            "yes_count": 3, "no_count": 0,
            "yes_total_cost": 150, "no_total_cost": 0,
            "revenue": 300, "market_result": "yes",
            "settled_time": "2026-01-10T22:00:00Z",
        }
        entries = _parse_settlement(s)
        assert len(entries) == 1
        e = entries[0]
        assert e["side"] == "YES"
        assert e["wager"] == 1.50
        assert e["payout"] == 3.00
        assert e["profit"] == 1.50
        assert e["result"] == "win"
        assert e["date"] == "2026-01-10"

    def test_no_loss(self):
        s = {
            "yes_count": 0, "no_count": 2,
            "yes_total_cost": 0, "no_total_cost": 100,
            "revenue": 0, "market_result": "yes",
            "settled_time": "2026-01-10T22:00:00Z",
        }
        entries = _parse_settlement(s)
        assert len(entries) == 1
        assert entries[0]["side"] == "NO"
        assert entries[0]["result"] == "loss"
        assert entries[0]["profit"] == -1.00

    def test_empty_position(self):
        s = {"yes_count": 0, "no_count": 0, "yes_total_cost": 0,
             "no_total_cost": 0, "revenue": 0, "settled_time": ""}
        assert _parse_settlement(s) == []


# ---------------------------------------------------------------------------
# _parse_settlement -- dual side (P1 fix: market-outcome revenue)
# ---------------------------------------------------------------------------

class TestParseSettlementDualSide:
    def test_yes_outcome(self):
        """1 YES @ $0.40, 1 NO @ $0.60, market settles YES.
        YES gets $1.00 payout, NO gets $0.00."""
        s = {
            "yes_count": 1, "no_count": 1,
            "yes_total_cost": 40, "no_total_cost": 60,
            "revenue": 100, "market_result": "yes",
            "settled_time": "2026-02-01T12:00:00Z",
        }
        entries = _parse_settlement(s)
        assert len(entries) == 2

        yes_entry = next(e for e in entries if e["side"] == "YES")
        no_entry = next(e for e in entries if e["side"] == "NO")

        assert yes_entry["wager"] == 0.40
        assert yes_entry["payout"] == 1.00
        assert yes_entry["profit"] == 0.60
        assert yes_entry["result"] == "win"

        assert no_entry["wager"] == 0.60
        assert no_entry["payout"] == 0.00
        assert no_entry["profit"] == -0.60
        assert no_entry["result"] == "loss"

    def test_no_outcome(self):
        """1 YES @ $0.40, 1 NO @ $0.60, market settles NO.
        YES gets $0.00, NO gets $1.00."""
        s = {
            "yes_count": 1, "no_count": 1,
            "yes_total_cost": 40, "no_total_cost": 60,
            "revenue": 100, "market_result": "no",
            "settled_time": "2026-02-01T12:00:00Z",
        }
        entries = _parse_settlement(s)
        assert len(entries) == 2

        yes_entry = next(e for e in entries if e["side"] == "YES")
        no_entry = next(e for e in entries if e["side"] == "NO")

        assert yes_entry["payout"] == 0.00
        assert yes_entry["result"] == "loss"

        assert no_entry["payout"] == 1.00
        assert no_entry["profit"] == 0.40
        assert no_entry["result"] == "win"

    def test_multiple_contracts_yes_outcome(self):
        """2 YES @ $0.30 each, 1 NO @ $0.70, market settles YES."""
        s = {
            "yes_count": 2, "no_count": 1,
            "yes_total_cost": 60, "no_total_cost": 70,
            "revenue": 200, "market_result": "yes",
            "settled_time": "2026-02-01T12:00:00Z",
        }
        entries = _parse_settlement(s)
        yes_entry = next(e for e in entries if e["side"] == "YES")
        no_entry = next(e for e in entries if e["side"] == "NO")

        # YES: 2 contracts * $1 = $2.00 payout, cost $0.60
        assert yes_entry["payout"] == 2.00
        assert yes_entry["profit"] == 1.40
        assert yes_entry["result"] == "win"

        # NO: 0 payout, cost $0.70
        assert no_entry["payout"] == 0.00
        assert no_entry["profit"] == -0.70
        assert no_entry["result"] == "loss"


# ---------------------------------------------------------------------------
# _find_pending_kalshi_match
# ---------------------------------------------------------------------------

def _make_pending_row(line: str, wager: float, platform: str = "Kalshi") -> dict:
    return {
        "result": "pending", "platform": platform, "wager": str(wager),
        "line": line, "bet_id": "", "game": "", "bet_type": "spread",
        "date": "2026-01-10", "odds": "", "payout": "", "profit": "", "league": "",
    }


class TestFindPendingKalshiMatch:
    def test_single_match(self):
        rows = [_make_pending_row("SMC -8.5 YES", 1.53)]
        idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26FEB14SMCPAC-SMC8", 1.53,
            "Saint Mary's wins by over 8.5 Points?",
        )
        assert idx == 0

    def test_no_match_wrong_wager(self):
        rows = [_make_pending_row("SMC -8.5 YES", 2.00)]
        idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26FEB14SMCPAC-SMC8", 1.53,
            "Saint Mary's wins by over 8.5 Points?",
        )
        assert idx is None

    def test_no_match_wrong_platform(self):
        rows = [_make_pending_row("SMC -8.5 YES", 1.53, platform="FanDuel")]
        idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26FEB14SMCPAC-SMC8", 1.53,
        )
        assert idx is None

    def test_disambiguate_by_side(self):
        """Two pending bets at the same wager, different sides."""
        rows = [
            _make_pending_row("SMC -8.5 YES", 0.50),
            _make_pending_row("SMC -8.5 NO", 0.50),
        ]
        yes_idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26FEB14SMCPAC-SMC8", 0.50,
            "Saint Mary's wins by over 8.5 Points?", "YES",
        )
        assert yes_idx == 0

        no_idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26FEB14SMCPAC-SMC8", 0.50,
            "Saint Mary's wins by over 8.5 Points?", "NO",
        )
        assert no_idx == 1

    def test_ambiguous_returns_none(self):
        """Two identical pending rows -- cannot disambiguate."""
        rows = [
            _make_pending_row("SMC -8.5 YES", 0.50),
            _make_pending_row("SMC -8.5 YES", 0.50),
        ]
        idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26FEB14SMCPAC-SMC8", 0.50,
            "Saint Mary's wins by over 8.5 Points?", "YES",
        )
        assert idx is None

    def test_side_not_substring_of_team_name(self):
        """NO should not match NORTHWESTERN -- word boundary required."""
        rows = [
            _make_pending_row("Northwestern -8.5 YES", 0.50),
            _make_pending_row("Northwestern -8.5 NO", 0.50),
        ]
        no_idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26JAN11NWRUTG-NW4", 0.50,
            "Northwestern wins by over 8.5 Points?", "NO",
        )
        assert no_idx == 1

        yes_idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26JAN11NWRUTG-NW4", 0.50,
            "Northwestern wins by over 8.5 Points?", "YES",
        )
        assert yes_idx == 0

    def test_disambiguate_by_spread(self):
        """Two pending bets at same wager, different spreads."""
        rows = [
            _make_pending_row("ALA -13.5 YES", 0.47),
            _make_pending_row("HOU -4.5 YES", 0.47),
        ]
        idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26JAN10TEXALA-ALA13", 0.47,
            "Alabama wins by over 13.5 Points?", "YES",
        )
        assert idx == 0

    def test_skips_settled_rows(self):
        rows = [
            {"result": "win", "platform": "Kalshi", "wager": "1.53",
             "line": "SMC -8.5 YES", "bet_id": "", "game": "", "bet_type": "",
             "date": "", "odds": "", "payout": "", "profit": "", "league": ""},
        ]
        idx = _find_pending_kalshi_match(
            rows, "KXNCAAMBSPREAD-26FEB14SMCPAC-SMC8", 1.53,
        )
        assert idx is None


# ---------------------------------------------------------------------------
# _existing_bet_ids
# ---------------------------------------------------------------------------

class TestExistingBetIds:
    def test_extracts_ids(self):
        rows = [
            {"bet_id": "TICKER-A"},
            {"bet_id": ""},
            {"bet_id": "TICKER-B"},
        ]
        assert _existing_bet_ids(rows) == {"TICKER-A", "TICKER-B"}

    def test_empty(self):
        assert _existing_bet_ids([]) == set()


# ---------------------------------------------------------------------------
# settle_to_csv -- sync cursor must not advance on API errors
# ---------------------------------------------------------------------------

class TestSettleToCsvErrorHandling:
    def test_sync_cursor_not_advanced_on_fetch_error(self, tmp_path, monkeypatch):
        """When get_settlements raises, last_sync_ts must not advance."""
        sync_file = tmp_path / ".kalshi_sync_state.json"
        old_ts = 1700000000
        sync_file.write_text(json.dumps({"last_sync_ts": old_ts}))

        monkeypatch.setattr("settle_kalshi.SYNC_STATE_FILE", str(sync_file))
        monkeypatch.setattr("settle_kalshi.BETTING_HISTORY", str(tmp_path / "bets.csv"))

        mock_client = MagicMock()
        mock_client.private_key = "key"
        mock_client.api_key = "key"
        mock_client.get_settlements.side_effect = RuntimeError("API error")

        with patch("settle_kalshi.KalshiClient", return_value=mock_client):
            result = settle_to_csv()

        assert result["error"] == "API error"
        # Sync cursor must still be at the old value
        with open(sync_file) as f:
            assert json.load(f)["last_sync_ts"] == old_ts
