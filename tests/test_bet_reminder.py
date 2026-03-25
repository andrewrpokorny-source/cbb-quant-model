"""Tests for the automatic bet logging reminder feature."""

import asyncio
import os
import csv
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Set up a temporary project directory with archive CSVs and betting history."""
    monkeypatch.setattr("telegram_bot.BASE_DIR", str(tmp_path))
    monkeypatch.setattr("telegram_bot.BETTING_HISTORY", str(tmp_path / "betting_history.csv"))
    return tmp_path


def _write_predictions_csv(path, rows):
    """Write a predictions archive CSV with the given rows (list of dicts)."""
    fieldnames = [
        "Bet_Type", "Date/Time", "Matchup", "Spread", "Pick", "Conf",
        "Raw Odds", "Rest", "Kalshi_Side", "Kalshi_Price", "Kalshi_Fee",
        "Kalshi_Title", "Edge", "Edge_Pct", "Rating", "Units",
        "Home_Matched", "Away_Matched", "Kalshi_Ticker", "Breakeven_Spread",
        "Std_Edge", "Std_Edge_Pct", "Std_Rating", "Std_Units",
        "Kalshi_Yes", "Kalshi_No", "Kalshi_Yes_Team", "Picked_Team",
        "Win_Model_Home_Prob", "Win_Model_Variant",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full_row = {k: "" for k in fieldnames}
            full_row.update(row)
            writer.writerow(full_row)


def _write_betting_history(path, rows):
    """Write a betting_history.csv with the given rows (list of dicts)."""
    fieldnames = ["date", "platform", "game", "bet_type", "line", "odds",
                  "wager", "result", "payout", "profit", "bet_id", "league"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full_row = {k: "" for k in fieldnames}
            full_row.update(row)
            writer.writerow(full_row)


class TestFindUnloggedStrongGames:
    """TDD Cycle 1: Finds STRONG games from prediction archives."""

    def test_returns_strong_game_from_archive(self, tmp_project):
        """A STRONG Std_Rating game on the target date should be returned."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 23)
        archive_path = tmp_project / "predictions_20260323.csv"
        _write_predictions_csv(archive_path, [
            {
                "Bet_Type": "spread",
                "Date/Time": "03/23 07:00 PM",
                "Matchup": "Wichita State Shockers @ Tulsa Golden Hurricane",
                "Pick": "Wichita State Shockers +4.5",
                "Std_Rating": "STRONG",
                "Std_Edge_Pct": "+9.2%",
                "Std_Units": "1.5",
                "Rating": "",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 1
        assert "Wichita State" in result[0]["matchup"]

    def test_returns_strong_kalshi_rating_game(self, tmp_project):
        """A STRONG Kalshi Rating game (but not Std) should also be returned."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 23)
        archive_path = tmp_project / "predictions_20260323.csv"
        _write_predictions_csv(archive_path, [
            {
                "Bet_Type": "game",
                "Date/Time": "03/23 10:10 PM",
                "Matchup": "Tennessee Volunteers @ Iowa State Cyclones",
                "Pick": "Tennessee Volunteers ML NO",
                "Rating": "STRONG",
                "Std_Rating": "PASS",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 1
        assert "Tennessee" in result[0]["matchup"]

    def test_ignores_non_strong_games(self, tmp_project):
        """GOOD/MARGINAL/PASS games should not be returned."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 23)
        archive_path = tmp_project / "predictions_20260323.csv"
        _write_predictions_csv(archive_path, [
            {
                "Bet_Type": "spread",
                "Date/Time": "03/23 07:00 PM",
                "Matchup": "Texas Longhorns @ Purdue Boilermakers",
                "Pick": "Texas Longhorns +6.5",
                "Std_Rating": "GOOD",
                "Rating": "GOOD",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 0

    def test_deduplicates_by_matchup(self, tmp_project):
        """Same matchup as both spread and game bet types should return only once."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 27)
        archive_path = tmp_project / "predictions_20260323.csv"
        _write_predictions_csv(archive_path, [
            {
                "Bet_Type": "game",
                "Date/Time": "03/27 07:10 PM",
                "Matchup": "St. John's Red Storm @ Duke Blue Devils",
                "Pick": "Duke Blue Devils ML YES",
                "Rating": "STRONG",
                "Std_Rating": "GOOD",
            },
            {
                "Bet_Type": "spread",
                "Date/Time": "03/27 07:10 PM",
                "Matchup": "St. John's Red Storm @ Duke Blue Devils",
                "Pick": "Duke Blue Devils -6.5",
                "Rating": "STRONG",
                "Std_Rating": "GOOD",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 1
        assert "Duke" in result[0]["matchup"]

    def test_excludes_logged_fanduel_game(self, tmp_project):
        """A STRONG game with a FanDuel entry in betting_history should NOT be returned."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 23)
        archive_path = tmp_project / "predictions_20260323.csv"
        _write_predictions_csv(archive_path, [
            {
                "Bet_Type": "spread",
                "Date/Time": "03/23 07:00 PM",
                "Matchup": "Wichita State Shockers @ Tulsa Golden Hurricane",
                "Pick": "Wichita State Shockers +4.5",
                "Std_Rating": "STRONG",
                "Std_Edge_Pct": "+9.2%",
                "Std_Units": "1.5",
            },
        ])
        _write_betting_history(tmp_project / "betting_history.csv", [
            {
                "date": "2026-03-23",
                "platform": "FanDuel",
                "game": "Wichita State vs Tulsa",
                "bet_type": "spread",
                "line": "Wichita State +4.5",
                "odds": "-110",
                "wager": "1.5",
                "result": "pending",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 0

    def test_excludes_logged_draftkings_game(self, tmp_project):
        """A STRONG game with a DraftKings entry should NOT be returned."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 23)
        archive_path = tmp_project / "predictions_20260323.csv"
        _write_predictions_csv(archive_path, [
            {
                "Bet_Type": "spread",
                "Date/Time": "03/23 07:00 PM",
                "Matchup": "Wichita State Shockers @ Tulsa Golden Hurricane",
                "Pick": "Wichita State Shockers +4.5",
                "Std_Rating": "STRONG",
            },
        ])
        _write_betting_history(tmp_project / "betting_history.csv", [
            {
                "date": "2026-03-23",
                "platform": "DraftKings",
                "game": "Wichita State vs Tulsa",
                "bet_type": "spread",
                "line": "Wichita State +4.5",
                "odds": "-110",
                "wager": "1.0",
                "result": "win",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 0

    def test_does_not_exclude_kalshi_only_logged(self, tmp_project):
        """A STRONG game logged only on Kalshi should still be returned (asking about FD/DK)."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 23)
        archive_path = tmp_project / "predictions_20260323.csv"
        _write_predictions_csv(archive_path, [
            {
                "Bet_Type": "spread",
                "Date/Time": "03/23 07:00 PM",
                "Matchup": "Wichita State Shockers @ Tulsa Golden Hurricane",
                "Pick": "Wichita State Shockers +4.5",
                "Std_Rating": "STRONG",
            },
        ])
        _write_betting_history(tmp_project / "betting_history.csv", [
            {
                "date": "2026-03-23",
                "platform": "Kalshi",
                "game": "Wichita State vs Tulsa",
                "bet_type": "spread",
                "line": "Wichita State +4.5 YES",
                "odds": "N/A",
                "wager": "0.5",
                "result": "pending",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 1

    def test_mixed_logged_and_unlogged(self, tmp_project):
        """With two STRONG games, one logged and one not, only the unlogged one is returned."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 23)
        archive_path = tmp_project / "predictions_20260323.csv"
        _write_predictions_csv(archive_path, [
            {
                "Bet_Type": "spread",
                "Date/Time": "03/23 07:00 PM",
                "Matchup": "Wichita State Shockers @ Tulsa Golden Hurricane",
                "Pick": "Wichita State Shockers +4.5",
                "Std_Rating": "STRONG",
            },
            {
                "Bet_Type": "spread",
                "Date/Time": "03/23 09:00 PM",
                "Matchup": "Arizona Wildcats @ Houston Cougars",
                "Pick": "Arizona Wildcats -7.5",
                "Std_Rating": "STRONG",
            },
        ])
        _write_betting_history(tmp_project / "betting_history.csv", [
            {
                "date": "2026-03-23",
                "platform": "FanDuel",
                "game": "Wichita State vs Tulsa",
                "bet_type": "spread",
                "line": "Wichita State +4.5",
                "odds": "-110",
                "wager": "1.5",
                "result": "pending",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 1
        assert "Arizona" in result[0]["matchup"]

    def test_handles_both_leagues(self, tmp_project):
        """STRONG games from both men's and women's archives should be returned with correct league."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 23)

        # Men's archive
        _write_predictions_csv(tmp_project / "predictions_20260323.csv", [
            {
                "Bet_Type": "spread",
                "Date/Time": "03/23 07:00 PM",
                "Matchup": "Wichita State Shockers @ Tulsa Golden Hurricane",
                "Pick": "Wichita State Shockers +4.5",
                "Std_Rating": "STRONG",
            },
        ])

        # Women's archive
        _write_predictions_csv(tmp_project / "predictions_wbb_20260323.csv", [
            {
                "Bet_Type": "spread",
                "Date/Time": "03/23 08:00 PM",
                "Matchup": "UConn Huskies @ South Carolina Gamecocks",
                "Pick": "UConn Huskies +3.5",
                "Std_Rating": "STRONG",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 2
        leagues = {r["league"] for r in result}
        assert leagues == {"mens", "womens"}

    def test_ignores_games_on_different_date(self, tmp_project):
        """STRONG games on other dates should not be returned."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 23)
        archive_path = tmp_project / "predictions_20260323.csv"
        _write_predictions_csv(archive_path, [
            {
                "Bet_Type": "spread",
                "Date/Time": "03/24 07:00 PM",
                "Matchup": "Arizona Wildcats @ Houston Cougars",
                "Pick": "Arizona Wildcats -7.5",
                "Std_Rating": "STRONG",
            },
        ])

        result = find_unlogged_strong_games(target)

        assert len(result) == 0


class TestReminderCallback:
    """TDD Cycle 5: _reminder_check_unlogged sends messages for unlogged games."""

    def test_sends_message_for_unlogged_games(self, monkeypatch):
        """Callback should send a message listing unlogged STRONG games."""
        from telegram_bot import _reminder_check_unlogged

        unlogged = [
            {
                "matchup": "Wichita State Shockers @ Tulsa Golden Hurricane",
                "pick": "Wichita State Shockers +4.5",
                "std_edge_pct": "+9.2%",
                "std_units": 1.5,
                "league": "mens",
                "bet_type": "spread",
            },
        ]
        monkeypatch.setattr("telegram_bot.find_unlogged_strong_games", lambda d: unlogged)
        monkeypatch.setattr("telegram_bot.ALLOWED_USER_IDS", {12345})

        mock_bot = AsyncMock()
        context = MagicMock()
        context.bot = mock_bot

        asyncio.run(_reminder_check_unlogged(context))

        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args
        assert call_kwargs[1]["chat_id"] == 12345
        assert "Wichita State" in call_kwargs[1]["text"]

    def test_silent_when_no_unlogged_games(self, monkeypatch):
        """Callback should not send any message if all games are logged."""
        from telegram_bot import _reminder_check_unlogged

        monkeypatch.setattr("telegram_bot.find_unlogged_strong_games", lambda d: [])
        monkeypatch.setattr("telegram_bot.ALLOWED_USER_IDS", {12345})

        mock_bot = AsyncMock()
        context = MagicMock()
        context.bot = mock_bot

        asyncio.run(_reminder_check_unlogged(context))

        mock_bot.send_message.assert_not_called()

    def test_sends_to_all_allowed_users(self, monkeypatch):
        """Callback should message every user in ALLOWED_USER_IDS."""
        from telegram_bot import _reminder_check_unlogged

        unlogged = [
            {
                "matchup": "Duke Blue Devils @ UNC Tar Heels",
                "pick": "Duke Blue Devils -6.5",
                "std_edge_pct": "+10.1%",
                "std_units": 2.0,
                "league": "mens",
                "bet_type": "spread",
            },
        ]
        monkeypatch.setattr("telegram_bot.find_unlogged_strong_games", lambda d: unlogged)
        monkeypatch.setattr("telegram_bot.ALLOWED_USER_IDS", {111, 222})

        mock_bot = AsyncMock()
        context = MagicMock()
        context.bot = mock_bot

        asyncio.run(_reminder_check_unlogged(context))

        assert mock_bot.send_message.call_count == 2
        chat_ids = {call[1]["chat_id"] for call in mock_bot.send_message.call_args_list}
        assert chat_ids == {111, 222}


class TestJobRegistration:
    """TDD Cycle 6: Daily job is registered in main()."""

    def test_registers_daily_job(self, monkeypatch):
        """main() should register a daily job on the job queue."""
        from telegram_bot import _register_scheduled_jobs

        mock_job_queue = MagicMock()
        _register_scheduled_jobs(mock_job_queue)

        mock_job_queue.run_daily.assert_called_once()
        call_kwargs = mock_job_queue.run_daily.call_args
        # First positional arg should be the callback
        from telegram_bot import _reminder_check_unlogged
        assert call_kwargs[0][0] is _reminder_check_unlogged or call_kwargs[1].get("callback") is _reminder_check_unlogged

    def test_schedules_at_6am_eastern(self, monkeypatch):
        """The daily job should be scheduled at 6:00 AM Eastern."""
        from telegram_bot import _register_scheduled_jobs
        import pytz

        mock_job_queue = MagicMock()
        _register_scheduled_jobs(mock_job_queue)

        call_kwargs = mock_job_queue.run_daily.call_args
        scheduled_time = call_kwargs[1].get("time") or call_kwargs[0][1]
        assert scheduled_time.hour == 6
        assert scheduled_time.minute == 0
        eastern = pytz.timezone("US/Eastern")
        assert scheduled_time.tzinfo is not None


class TestIntegrationRealData:
    """TDD Cycle 7: Integration test against real prediction archives."""

    def test_finds_unlogged_strong_game_from_real_archive(self):
        """Real archive predictions_20260323.csv has a STRONG Wichita State game on 03/24
        with no corresponding FanDuel/DraftKings entry in betting_history.csv."""
        from telegram_bot import find_unlogged_strong_games

        target = date(2026, 3, 24)
        result = find_unlogged_strong_games(target)

        matchups = [r["matchup"] for r in result]
        assert any("Wichita State" in m for m in matchups), (
            f"Expected Wichita State STRONG game on 03/24 to be unlogged. Got: {matchups}"
        )
