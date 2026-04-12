"""Tests for settle_polymarket position parsing and CSV writing."""

import csv
import os
import pytest

from settle_polymarket import _parse_position, _existing_bet_ids, settle_to_csv


class TestParsePosition:
    """Tests for _parse_position against the documented closed-positions schema."""

    def _pos(self, **overrides):
        base = {
            "conditionId": "0xabc123",
            "asset": "98765432109876543210",
            "title": "New York Yankees vs. Boston Red Sox",
            "outcome": "YES",
            "avgPrice": 0.45,
            "totalBought": 10.0,
            "realizedPnl": 5.50,
            "timestamp": 1744300800,  # 2025-04-10 16:00:00 UTC
            "side": "YES",
        }
        base.update(overrides)
        return base

    def test_winning_position(self):
        row = _parse_position(self._pos(realizedPnl=5.50, totalBought=10.0))
        assert row["result"] == "win"
        assert row["profit"] == 5.50
        assert row["wager"] == 10.0
        assert row["payout"] == 15.50

    def test_losing_position(self):
        row = _parse_position(self._pos(realizedPnl=-10.0, totalBought=10.0))
        assert row["result"] == "loss"
        assert row["profit"] == -10.0
        assert row["wager"] == 10.0
        assert row["payout"] == 0.0

    def test_breakeven_position(self):
        row = _parse_position(self._pos(realizedPnl=0.0, totalBought=5.0))
        assert row["result"] == "push"
        assert row["profit"] == 0.0

    def test_platform_is_polymarket(self):
        row = _parse_position(self._pos())
        assert row["platform"] == "Polymarket"

    def test_epoch_timestamp_parsed(self):
        row = _parse_position(self._pos(timestamp=1744300800))
        assert row["date"] == "2025-04-10"

    def test_epoch_string_timestamp_parsed(self):
        row = _parse_position(self._pos(timestamp="1744300800"))
        assert row["date"] == "2025-04-10"

    def test_iso_timestamp_parsed(self):
        row = _parse_position(self._pos(timestamp="2025-04-10T16:00:00Z"))
        assert row["date"] == "2025-04-10"

    def test_missing_timestamp_gives_empty_date(self):
        row = _parse_position(self._pos(timestamp=None, created_at=None, date=None))
        assert row["date"] == ""

    def test_bet_type_game_from_title(self):
        row = _parse_position(self._pos(title="Yankees vs Red Sox"))
        assert row["bet_type"] == "game"

    def test_bet_type_spread_from_title(self):
        row = _parse_position(self._pos(title="Spread: Red Sox (-1.5)"))
        assert row["bet_type"] == "spread"

    def test_bet_type_total_from_title(self):
        row = _parse_position(self._pos(title="Yankees vs Red Sox: O/U 8.5"))
        assert row["bet_type"] == "total"

    def test_bet_id_includes_outcome_and_asset(self):
        row = _parse_position(self._pos(
            conditionId="0xabc",
            outcome="YES",
            asset="98765432109876543210",
        ))
        assert row["bet_id"] == "poly_0xabc_YES_76543210"

    def test_bet_id_without_asset(self):
        row = _parse_position(self._pos(
            conditionId="0xdef",
            outcome="NO",
            asset="",
        ))
        assert row["bet_id"] == "poly_0xdef_NO"

    def test_distinct_bet_ids_for_both_sides(self):
        yes_row = _parse_position(self._pos(
            conditionId="0xabc", outcome="YES", asset="111111111122222222"
        ))
        no_row = _parse_position(self._pos(
            conditionId="0xabc", outcome="NO", asset="333333333344444444"
        ))
        assert yes_row["bet_id"] != no_row["bet_id"]

    def test_missing_pnl_fields_default_to_zero(self):
        row = _parse_position(self._pos(realizedPnl=None, totalBought=None))
        assert row["profit"] == 0.0
        assert row["wager"] == 0.0

    def test_string_numeric_fields_parsed(self):
        row = _parse_position(self._pos(realizedPnl="3.25", totalBought="8.00"))
        assert row["profit"] == 3.25
        assert row["wager"] == 8.0


class TestExistingBetIds:
    """Test dedup ID loading from CSV."""

    def test_reads_bet_ids_from_csv(self, tmp_path):
        csv_path = tmp_path / "history.csv"
        csv_path.write_text(
            "date,platform,game,bet_type,line,odds,wager,result,payout,profit,bet_id,league\n"
            "2025-04-10,Polymarket,Test,game,Test YES,n/a,10,win,15,5,poly_abc_YES,mlb\n"
            "2025-04-10,Kalshi,Other,spread,Other -1.5,n/a,5,loss,0,-5,kalshi_xyz,mlb\n"
        )
        import settle_polymarket
        orig = settle_polymarket.BETTING_HISTORY
        settle_polymarket.BETTING_HISTORY = str(csv_path)
        try:
            ids = _existing_bet_ids()
            assert "poly_abc_YES" in ids
            assert "kalshi_xyz" in ids
        finally:
            settle_polymarket.BETTING_HISTORY = orig

    def test_empty_when_no_file(self, tmp_path):
        import settle_polymarket
        orig = settle_polymarket.BETTING_HISTORY
        settle_polymarket.BETTING_HISTORY = str(tmp_path / "nonexistent.csv")
        try:
            assert _existing_bet_ids() == set()
        finally:
            settle_polymarket.BETTING_HISTORY = orig


class TestSettleToCsv:
    """Integration tests for the full settlement flow."""

    def test_skips_without_proxy(self, monkeypatch):
        monkeypatch.delenv("POLYMARKET_PROXY", raising=False)
        monkeypatch.delenv("POLYMARKET_WALLET_ADDRESS", raising=False)
        result = settle_to_csv(wallet_address="0xtest")
        assert result["settled"] == 0

    def test_skips_without_wallet(self, monkeypatch):
        monkeypatch.setenv("POLYMARKET_PROXY", "socks5h://127.0.0.1:8080")
        monkeypatch.delenv("POLYMARKET_WALLET_ADDRESS", raising=False)
        result = settle_to_csv()
        assert result["settled"] == 0

    def test_writes_new_positions_to_csv(self, monkeypatch, tmp_path):
        csv_path = tmp_path / "history.csv"

        import settle_polymarket
        orig_hist = settle_polymarket.BETTING_HISTORY
        orig_sync = settle_polymarket.SYNC_STATE_FILE
        settle_polymarket.BETTING_HISTORY = str(csv_path)
        settle_polymarket.SYNC_STATE_FILE = str(tmp_path / ".sync.json")

        monkeypatch.setenv("POLYMARKET_PROXY", "socks5h://127.0.0.1:8080")

        fake_positions = [
            {
                "conditionId": "0xabc",
                "asset": "1111111122222222",
                "title": "Yankees vs Red Sox",
                "outcome": "YES",
                "totalBought": 10.0,
                "realizedPnl": 5.0,
                "timestamp": 1744300800,
                "side": "YES",
            },
        ]
        monkeypatch.setattr(
            "settle_polymarket.PolymarketClient.get_closed_positions",
            lambda self, addr: fake_positions,
        )

        try:
            result = settle_to_csv(wallet_address="0xtest")
            assert result["settled"] == 1
            assert result["skipped"] == 0

            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["platform"] == "Polymarket"
            assert rows[0]["profit"] == "5.0"
            assert rows[0]["result"] == "win"
        finally:
            settle_polymarket.BETTING_HISTORY = orig_hist
            settle_polymarket.SYNC_STATE_FILE = orig_sync

    def test_skips_already_recorded_positions(self, monkeypatch, tmp_path):
        csv_path = tmp_path / "history.csv"
        csv_path.write_text(
            "date,platform,game,bet_type,line,odds,wager,result,payout,profit,bet_id,league\n"
            "2025-04-10,Polymarket,Yankees vs Red Sox,game,Yankees vs Red Sox YES,n/a,10.0,win,15.0,5.0,poly_0xabc_YES_22222222,mlb\n"
        )

        import settle_polymarket
        orig_hist = settle_polymarket.BETTING_HISTORY
        orig_sync = settle_polymarket.SYNC_STATE_FILE
        settle_polymarket.BETTING_HISTORY = str(csv_path)
        settle_polymarket.SYNC_STATE_FILE = str(tmp_path / ".sync.json")

        monkeypatch.setenv("POLYMARKET_PROXY", "socks5h://127.0.0.1:8080")

        fake_positions = [
            {
                "conditionId": "0xabc",
                "asset": "1111111122222222",
                "title": "Yankees vs Red Sox",
                "outcome": "YES",
                "totalBought": 10.0,
                "realizedPnl": 5.0,
                "timestamp": 1744300800,
                "side": "YES",
            },
        ]
        monkeypatch.setattr(
            "settle_polymarket.PolymarketClient.get_closed_positions",
            lambda self, addr: fake_positions,
        )

        try:
            result = settle_to_csv(wallet_address="0xtest")
            assert result["settled"] == 0
            assert result["skipped"] == 1
        finally:
            settle_polymarket.BETTING_HISTORY = orig_hist
            settle_polymarket.SYNC_STATE_FILE = orig_sync
