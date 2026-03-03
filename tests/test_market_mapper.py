"""Tests for kalshi.market_mapper team name normalization and matching."""

from kalshi.market_mapper import MarketMapper, normalize_team_name


class TestNormalizeTeamName:
    """Tests for Lady prefix stripping and mascot normalization."""

    def test_strips_lady_single_word_mascot(self):
        assert normalize_team_name("Ole Miss Lady Rebels") == "Ole Miss"

    def test_strips_lady_multi_word_mascot(self):
        assert normalize_team_name("Southern Miss Lady Golden Eagles") == "Southern Miss"

    def test_cowgirls_removed_as_suffix(self):
        assert normalize_team_name("Oklahoma State Cowgirls") == "Oklahoma State"

    def test_ladyjacks_removed_as_suffix(self):
        assert normalize_team_name("Stephen F. Austin Ladyjacks") == "Stephen F. Austin"

    def test_preserves_name_without_lady(self):
        assert normalize_team_name("UConn Huskies") == "UConn"

    def test_standard_mascot_removal(self):
        assert normalize_team_name("Kansas Jayhawks") == "Kansas"


class TestTeamsInRules:
    """Tests for _teams_in_rules abbreviation variant matching."""

    def _check(self, rules, keyword):
        mapper = MarketMapper([])
        return mapper._teams_in_rules(rules, keyword)

    def test_exact_match(self):
        assert self._check("Ohio State wins", "ohio state")

    def test_state_to_st_dot(self):
        assert self._check("Ohio St. wins the game", "ohio state")

    def test_st_dot_to_state(self):
        assert self._check("Ohio State wins the game", "ohio st.")

    def test_saint_to_st_dot(self):
        assert self._check("St. Mary's wins", "saint mary")

    def test_st_dot_to_saint(self):
        assert self._check("Saint John's wins", "st. john")

    def test_no_match_returns_false(self):
        assert not self._check("Duke wins", "kentucky")
