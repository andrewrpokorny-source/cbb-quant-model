"""Tests for live position matching -- ensures positions only match their actual game."""

import pytest

from dashboard_helpers import (
    extract_ticker_date,
    extract_ticker_teams,
    position_matches_game,
    utc_date_to_eastern,
)


class TestExtractTickerTeams:
    """Extract both team abbreviations from a Kalshi ticker."""

    def test_mlb_game_ticker(self):
        assert extract_ticker_teams("KXMLBGAME-26APR061845STLWSH-STL") == ("STL", "WSH")

    def test_mlb_game_ticker_short_abbrs(self):
        assert extract_ticker_teams("KXMLBGAME-26APR041910TBMIN-TB") == ("TB", "MIN")

    def test_mlb_game_ticker_three_letter(self):
        assert extract_ticker_teams("KXMLBGAME-26APR051510PHICOL-COL") == ("PHI", "COL")

    def test_cbb_game_ticker(self):
        # CBB tickers don't encode both teams in a parseable way
        # so we can only extract the YES team from the suffix
        away, home = extract_ticker_teams("KXNCAAMBGAME-26APR04ILLCONN-ILL")
        assert "ILL" in (away, home)

    def test_malformed_ticker_returns_empty(self):
        assert extract_ticker_teams("GARBAGE") == ("", "")

    def test_single_segment_ticker(self):
        assert extract_ticker_teams("KXMLBGAME") == ("", "")


class TestPositionMatchesGame:
    """Verify that positions only match their actual live game."""

    def _game(self, home_abbr, away_abbr, league="mlb", game_date=None):
        g = {
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_name": f"{home_abbr} Team",
            "away_name": f"{away_abbr} Team",
            "league": league,
        }
        if game_date:
            g["game_date"] = game_date
        return g

    def test_mlb_position_matches_correct_game(self):
        """STL @ WSH position should match STL vs WSH game."""
        game = self._game("WSH", "STL")
        assert position_matches_game(
            "KXMLBGAME-26APR061845STLWSH-STL", game, "mlb"
        )

    def test_mlb_position_does_not_match_wrong_opponent(self):
        """STL @ WSH position should NOT match STL @ DET game."""
        game = self._game("DET", "STL")
        assert not position_matches_game(
            "KXMLBGAME-26APR061845STLWSH-STL", game, "mlb"
        )

    def test_mlb_position_does_not_match_wrong_opponent_reverse(self):
        """DET @ MIN position should NOT match STL @ DET game."""
        game = self._game("DET", "STL")
        assert not position_matches_game(
            "KXMLBGAME-26APR061940DETMIN-DET", game, "mlb"
        )

    def test_mlb_position_matches_home_team(self):
        """DET @ MIN position should match DET @ MIN game."""
        game = self._game("MIN", "DET")
        assert position_matches_game(
            "KXMLBGAME-26APR061940DETMIN-DET", game, "mlb"
        )

    def test_cbb_position_matches_by_yes_abbr(self):
        """CBB tickers can't extract both teams -- fall back to YES abbr match."""
        game = self._game("CONN", "ILL", league="mens")
        assert position_matches_game(
            "KXNCAAMBGAME-26APR04ILLCONN-ILL", game, "mens"
        )

    def test_cross_league_no_match(self):
        """MLB position should not match CBB game."""
        game = self._game("WSH", "STL", league="mens")
        assert not position_matches_game(
            "KXMLBGAME-26APR061845STLWSH-STL", game, "mlb"
        )

    def test_no_side_position_matches(self):
        """NO-side position should also verify opponent."""
        game = self._game("WSH", "STL")
        # YES team is STL, but we hold NO (betting on WSH)
        assert position_matches_game(
            "KXMLBGAME-26APR061845STLWSH-STL", game, "mlb"
        )

    def test_doubleheader_same_teams_matches(self):
        """Both doubleheader games have same teams -- both should match."""
        game = self._game("KC", "MIL")
        assert position_matches_game(
            "KXMLBGAME-26APR041610MILKC-MIL", game, "mlb"
        )
        assert position_matches_game(
            "KXMLBGAME-26APR041910MILKC-KC", game, "mlb"
        )

    def test_stale_position_does_not_match_next_day(self):
        """Apr 6 LAD/TOR position must NOT match Apr 7 LAD/TOR game."""
        game = self._game("TOR", "LAD", game_date="2026-04-07")
        assert not position_matches_game(
            "KXMLBGAME-26APR061907LADTOR-TOR", game, "mlb"
        )

    def test_same_day_position_still_matches(self):
        """Apr 7 LAD/TOR position should match Apr 7 LAD/TOR game."""
        game = self._game("TOR", "LAD", game_date="2026-04-07")
        assert position_matches_game(
            "KXMLBGAME-26APR071907LADTOR-TOR", game, "mlb"
        )

    def test_no_game_date_still_matches(self):
        """If game has no date (legacy), fall back to team-only matching."""
        game = self._game("TOR", "LAD")
        assert position_matches_game(
            "KXMLBGAME-26APR071907LADTOR-TOR", game, "mlb"
        )

    def test_stale_cbb_position_does_not_match(self):
        """CBB position from wrong date should not match."""
        game = self._game("CONN", "ILL", league="mens", game_date="2026-04-05")
        assert not position_matches_game(
            "KXNCAAMBGAME-26APR04ILLCONN-ILL", game, "mens"
        )


class TestExtractTickerDate:
    """Extract game date from a Kalshi ticker."""

    def test_mlb_ticker_date(self):
        assert extract_ticker_date("KXMLBGAME-26APR061845STLWSH-STL") == "2026-04-06"

    def test_mlb_ticker_different_month(self):
        assert extract_ticker_date("KXMLBGAME-26MAY151910NYYLAD-NYY") == "2026-05-15"

    def test_cbb_ticker_date(self):
        assert extract_ticker_date("KXNCAAMBGAME-26APR04ILLCONN-ILL") == "2026-04-04"

    def test_malformed_ticker_returns_empty(self):
        assert extract_ticker_date("GARBAGE") == ""

    def test_short_middle_returns_empty(self):
        assert extract_ticker_date("KXMLBGAME-26AP-STL") == ""


class TestUtcDateToEastern:
    """ESPN dates are UTC -- must convert to Eastern for ticker comparison."""

    def test_daytime_game_same_date(self):
        """1 PM ET game = 5 PM UTC, same date in both."""
        assert utc_date_to_eastern("2026-04-07T17:00Z") == "2026-04-07"

    def test_evening_game_same_date(self):
        """7 PM ET game = 11 PM UTC, same date in both."""
        assert utc_date_to_eastern("2026-04-07T23:00Z") == "2026-04-07"

    def test_late_night_west_coast_game(self):
        """10:10 PM ET = 2:10 AM UTC next day. Must return ET date, not UTC."""
        assert utc_date_to_eastern("2026-04-08T02:10Z") == "2026-04-07"

    def test_past_midnight_et(self):
        """Extra innings ending at 12:30 AM ET = 4:30 AM UTC. Still prior ET date."""
        assert utc_date_to_eastern("2026-04-08T04:30Z") == "2026-04-08"

    def test_full_iso_format(self):
        """Handle full ISO format with seconds."""
        assert utc_date_to_eastern("2026-04-07T23:10:00Z") == "2026-04-07"

    def test_empty_string(self):
        assert utc_date_to_eastern("") == ""

    def test_invalid_string(self):
        assert utc_date_to_eastern("not-a-date") == ""

    def test_date_only_no_time(self):
        """Plain date string without time should pass through."""
        assert utc_date_to_eastern("2026-04-07") == "2026-04-07"

    def test_dst_transition(self):
        """During EDT (summer), ET = UTC-4."""
        # 3:30 AM UTC on Apr 7 = 11:30 PM ET on Apr 6
        assert utc_date_to_eastern("2026-04-07T03:30Z") == "2026-04-06"
