"""
Unit tests for settle_bets.py

Tests the critical bet parsing, result determination, and payout calculation logic.
"""

import pytest
from settle_bets import parse_bet_line, determine_bet_result, calculate_payout


class TestParseBetLine:
    """Tests for parse_bet_line function."""

    def test_standard_spread_positive(self):
        """Standard underdog spread."""
        result = parse_bet_line("Providence +15.5")
        assert result == {"team": "Providence", "spread": 15.5, "side": None}

    def test_standard_spread_negative(self):
        """Standard favorite spread."""
        result = parse_bet_line("UConn -15.5")
        assert result == {"team": "UConn", "spread": -15.5, "side": None}

    def test_kalshi_yes(self):
        """Kalshi YES bet."""
        result = parse_bet_line("Furman -10.5 YES")
        assert result == {"team": "Furman", "spread": -10.5, "side": "YES"}

    def test_kalshi_no(self):
        """Kalshi NO bet."""
        result = parse_bet_line("UConn -15.5 NO")
        assert result == {"team": "UConn", "spread": -15.5, "side": "NO"}

    def test_multi_word_team(self):
        """Team name with multiple words."""
        result = parse_bet_line("NC State +7.5")
        assert result == {"team": "NC State", "spread": 7.5, "side": None}

    def test_integer_spread(self):
        """Integer spread (no decimal)."""
        result = parse_bet_line("Duke -3")
        assert result == {"team": "Duke", "spread": -3.0, "side": None}

    def test_embedded_newlines(self):
        """OCR garbled text with newlines - should extract valid spread line."""
        result = parse_bet_line("junk text\nmore junk\nDuke +5.5")
        assert result is not None
        assert result["team"] == "Duke"
        assert result["spread"] == 5.5

    def test_invalid_format(self):
        """Invalid format should return None."""
        result = parse_bet_line("invalid")
        assert result is None

    def test_empty_string(self):
        """Empty string should return None."""
        result = parse_bet_line("")
        assert result is None

    def test_team_with_apostrophe(self):
        """Team name with apostrophe."""
        result = parse_bet_line("St. John's +4.5")
        assert result == {"team": "St. John's", "spread": 4.5, "side": None}


class TestDetermineBetResult:
    """Tests for determine_bet_result function."""

    def test_standard_spread_win(self):
        """Standard spread bet - win."""
        parsed = {"team": "Providence", "spread": 15.5, "side": None}
        game = {
            "home_score": 85,
            "away_score": 75,
            "home_name": "UConn",
            "away_name": "Providence",
        }
        # Providence (away) with +15.5, loses by 10 -> 75 + 15.5 - 85 = 5.5 > 0 -> win
        result = determine_bet_result(parsed, game, "Providence vs UConn")
        assert result == "win"

    def test_standard_spread_loss(self):
        """Standard spread bet - loss."""
        parsed = {"team": "Providence", "spread": 5.5, "side": None}
        game = {
            "home_score": 85,
            "away_score": 75,
            "home_name": "UConn",
            "away_name": "Providence",
        }
        # Providence (away) with +5.5, loses by 10 -> 75 + 5.5 - 85 = -4.5 < 0 -> loss
        result = determine_bet_result(parsed, game, "Providence vs UConn")
        assert result == "loss"

    def test_standard_spread_push(self):
        """Standard spread bet - push (void)."""
        parsed = {"team": "Providence", "spread": 10.0, "side": None}
        game = {
            "home_score": 85,
            "away_score": 75,
            "home_name": "UConn",
            "away_name": "Providence",
        }
        # Providence (away) with +10, loses by 10 -> 75 + 10 - 85 = 0 -> void
        result = determine_bet_result(parsed, game, "Providence vs UConn")
        assert result == "void"

    def test_favorite_covers(self):
        """Favorite covers the spread - win."""
        parsed = {"team": "Duke", "spread": -5.5, "side": None}
        game = {
            "home_score": 80,
            "away_score": 70,
            "home_name": "Duke",
            "away_name": "UNC",
        }
        # Duke (home) with -5.5, wins by 10 -> 80 + (-5.5) - 70 = 4.5 > 0 -> win
        result = determine_bet_result(parsed, game, "UNC vs Duke")
        assert result == "win"

    def test_favorite_fails_to_cover(self):
        """Favorite fails to cover the spread - loss."""
        parsed = {"team": "Duke", "spread": -15.5, "side": None}
        game = {
            "home_score": 80,
            "away_score": 70,
            "home_name": "Duke",
            "away_name": "UNC",
        }
        # Duke (home) with -15.5, wins by 10 -> 80 + (-15.5) - 70 = -5.5 < 0 -> loss
        result = determine_bet_result(parsed, game, "UNC vs Duke")
        assert result == "loss"

    def test_kalshi_yes_win(self):
        """Kalshi YES bet - team wins by more than threshold."""
        parsed = {"team": "UConn", "spread": -15.5, "side": "YES"}
        game = {
            "home_score": 95,
            "away_score": 75,
            "home_name": "UConn",
            "away_name": "Providence",
        }
        # UConn wins by 20, threshold is 15.5 -> 20 > 15.5 -> YES wins
        result = determine_bet_result(parsed, game, "Providence vs UConn")
        assert result == "win"

    def test_kalshi_yes_loss(self):
        """Kalshi YES bet - team does NOT win by more than threshold."""
        parsed = {"team": "UConn", "spread": -15.5, "side": "YES"}
        game = {
            "home_score": 85,
            "away_score": 75,
            "home_name": "UConn",
            "away_name": "Providence",
        }
        # UConn wins by 10, threshold is 15.5 -> 10 < 15.5 -> YES loses
        result = determine_bet_result(parsed, game, "Providence vs UConn")
        assert result == "loss"

    def test_kalshi_no_win(self):
        """Kalshi NO bet - team does NOT win by more than threshold."""
        parsed = {"team": "UConn", "spread": -15.5, "side": "NO"}
        game = {
            "home_score": 85,
            "away_score": 75,
            "home_name": "UConn",
            "away_name": "Providence",
        }
        # UConn wins by 10, threshold is 15.5 -> 10 < 15.5 -> NO wins
        result = determine_bet_result(parsed, game, "Providence vs UConn")
        assert result == "win"

    def test_kalshi_no_loss(self):
        """Kalshi NO bet - team DOES win by more than threshold."""
        parsed = {"team": "UConn", "spread": -15.5, "side": "NO"}
        game = {
            "home_score": 95,
            "away_score": 75,
            "home_name": "UConn",
            "away_name": "Providence",
        }
        # UConn wins by 20, threshold is 15.5 -> 20 > 15.5 -> NO loses
        result = determine_bet_result(parsed, game, "Providence vs UConn")
        assert result == "loss"

    def test_kalshi_exact_threshold_void(self):
        """Kalshi bet - exact threshold is a push (void)."""
        parsed = {"team": "UConn", "spread": -15.0, "side": "YES"}
        game = {
            "home_score": 90,
            "away_score": 75,
            "home_name": "UConn",
            "away_name": "Providence",
        }
        # UConn wins by 15, threshold is 15 -> exact match -> void
        result = determine_bet_result(parsed, game, "Providence vs UConn")
        assert result == "void"

    def test_unknown_team_returns_none(self):
        """Unknown team should return None."""
        parsed = {"team": "Unknown", "spread": 5.5, "side": None}
        game = {
            "home_score": 80,
            "away_score": 70,
            "home_name": "Duke",
            "away_name": "UNC",
        }
        result = determine_bet_result(parsed, game, "UNC vs Duke")
        assert result is None


class TestCalculatePayout:
    """Tests for calculate_payout function."""

    def test_favorite_win(self):
        """Win with negative odds (favorite)."""
        payout, profit = calculate_payout("-110", 1.10, "win")
        assert payout == 2.10
        assert profit == 1.00

    def test_underdog_win(self):
        """Win with positive odds (underdog)."""
        payout, profit = calculate_payout("+150", 1.00, "win")
        assert payout == 2.50
        assert profit == 1.50

    def test_even_money_negative(self):
        """Win at -100 (even money favorite notation)."""
        payout, profit = calculate_payout("-100", 1.00, "win")
        assert payout == 2.00
        assert profit == 1.00

    def test_even_money_positive(self):
        """Win at +100 (even money underdog notation)."""
        payout, profit = calculate_payout("+100", 1.00, "win")
        assert payout == 2.00
        assert profit == 1.00

    def test_loss(self):
        """Loss should return 0 payout and negative profit."""
        payout, profit = calculate_payout("-110", 1.10, "loss")
        assert payout == 0.00
        assert profit == -1.10

    def test_void(self):
        """Void/push should return wager and 0 profit."""
        payout, profit = calculate_payout("-110", 1.10, "void")
        assert payout == 1.10
        assert profit == 0.00

    def test_kalshi_na_odds(self):
        """Kalshi win with missing odds and no stored payout stays unknown."""
        payout, profit = calculate_payout("n/a", 1.00, "win", platform="Kalshi")
        assert payout == 0.00
        assert profit == 0.00

    def test_kalshi_stored_payout(self):
        """Kalshi win uses stored max payout captured at log time."""
        payout, profit = calculate_payout(
            "n/a",
            1.59,
            "win",
            platform="Kalshi",
            stored_payout=3.00,
        )
        assert payout == 3.00
        assert profit == 1.41

    def test_empty_odds(self):
        """Empty odds string returns zeros for non-Kalshi books."""
        payout, profit = calculate_payout("", 1.00, "win")
        assert payout == 0.00
        assert profit == 0.00

    def test_heavy_favorite(self):
        """Heavy favorite odds (-300)."""
        payout, profit = calculate_payout("-300", 3.00, "win")
        # profit = 3.00 * 100 / 300 = 1.00
        assert profit == 1.00
        assert payout == 4.00

    def test_big_underdog(self):
        """Big underdog odds (+300)."""
        payout, profit = calculate_payout("+300", 1.00, "win")
        # profit = 1.00 * 300 / 100 = 3.00
        assert profit == 3.00
        assert payout == 4.00
