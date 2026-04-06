"""Tests for live position matching -- ensures positions only match their actual game."""

import pytest

from dashboard_helpers import extract_ticker_teams, position_matches_game


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

    def _game(self, home_abbr, away_abbr, league="mlb"):
        return {
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_name": f"{home_abbr} Team",
            "away_name": f"{away_abbr} Team",
            "league": league,
        }

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
