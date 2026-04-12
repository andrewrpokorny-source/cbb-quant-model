"""Tests for polymarket.market_mapper field detection, team matching, and game_time disambiguation."""

from datetime import datetime, timezone

import pytest

from polymarket.market_mapper import PolymarketMarketMapper


# -- Fixtures --

def _make_market(title, date_str=None, market_type=None, outcomes=None,
                 prices=None, token_id=None):
    """Helper to build a market dict matching common Polymarket schemas."""
    m = {"question": title}
    if date_str:
        m["game_start_time"] = date_str
    if market_type:
        m["market_type"] = market_type
    if outcomes:
        m["outcomes"] = outcomes
    if prices:
        m["outcomePrices"] = prices
    if token_id:
        m["clobTokenIds"] = [token_id]
    return m


DUKE_UNC_GAME = _make_market(
    "Will Duke beat UNC?",
    date_str="2026-04-09T23:00:00Z",
    market_type="moneyline",
    outcomes=["Duke", "UNC"],
    prices=["0.55", "0.45"],
    token_id="tok_duke_unc_game",
)

DUKE_UNC_SPREAD = _make_market(
    "Duke -5.5 vs UNC",
    date_str="2026-04-09T23:00:00Z",
    market_type="spread",
    prices=["0.48", "0.52"],
    token_id="tok_duke_unc_spread",
)

DUKE_UNC_TOMORROW = _make_market(
    "Will Duke beat UNC?",
    date_str="2026-04-10T23:00:00Z",
    market_type="moneyline",
    prices=["0.60", "0.40"],
    token_id="tok_duke_unc_tmrw",
)


class TestFieldDetection:
    """Auto-detection of JSON field names from market dicts."""

    def test_detects_question_field(self):
        mapper = PolymarketMarketMapper([{"question": "Test?", "id": "1"}])
        assert mapper._f_title == "question"

    def test_detects_title_field(self):
        mapper = PolymarketMarketMapper([{"title": "Test?", "id": "1"}])
        assert mapper._f_title == "title"

    def test_detects_clob_token_ids(self):
        mapper = PolymarketMarketMapper([{"question": "T", "clobTokenIds": ["a", "b"]}])
        assert mapper._f_token == "clobTokenIds"

    def test_flattens_nested_events(self):
        event = {
            "title": "Duke vs UNC",
            "markets": [
                {"question": "Will Duke win?", "id": "m1"},
                {"question": "Duke spread", "id": "m2"},
            ],
        }
        mapper = PolymarketMarketMapper([event])
        assert len(mapper.markets) == 2
        assert mapper.markets[0].get("_event", {}).get("title") == "Duke vs UNC"


class TestDateParsing:
    """Tests for _get_date with various formats."""

    def test_iso_z_suffix_preserves_utc(self):
        m = {"question": "T", "game_start_time": "2026-04-09T19:00:00Z"}
        mapper = PolymarketMarketMapper([m])
        dt = mapper._get_date(m)
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.hour == 19

    def test_unix_timestamp_returns_utc(self):
        m = {"question": "T", "game_start_time": 1776000000}
        mapper = PolymarketMarketMapper([m])
        dt = mapper._get_date(m)
        assert dt is not None
        assert dt.tzinfo is not None

    def test_naive_iso_returns_naive(self):
        m = {"question": "T", "game_start_time": "2026-04-09T19:00:00"}
        mapper = PolymarketMarketMapper([m])
        dt = mapper._get_date(m)
        assert dt is not None
        assert dt.tzinfo is None
        assert dt.hour == 19

    def test_date_only(self):
        m = {"question": "T", "game_start_time": "2026-04-09"}
        mapper = PolymarketMarketMapper([m])
        dt = mapper._get_date(m)
        assert dt is not None
        assert dt.day == 9


class TestTeamMatching:
    """Tests for _teams_match with keywords and abbreviation variants."""

    def test_both_teams_in_title(self):
        mapper = PolymarketMarketMapper([DUKE_UNC_GAME])
        assert mapper._teams_match(DUKE_UNC_GAME, "duke", "unc")

    def test_partial_match_fails(self):
        mapper = PolymarketMarketMapper([DUKE_UNC_GAME])
        assert not mapper._teams_match(DUKE_UNC_GAME, "duke", "kentucky")

    def test_state_abbreviation_variant(self):
        m = _make_market("Ohio State vs Michigan")
        mapper = PolymarketMarketMapper([m])
        assert mapper._teams_match(m, "ohio st.", "michigan")

    def test_saint_abbreviation_variant(self):
        m = _make_market("St. John's vs Villanova")
        mapper = PolymarketMarketMapper([m])
        assert mapper._teams_match(m, "saint john", "villanova")


class TestFindAllMarketsForGame:
    """Tests for find_all_markets_for_game with date filtering."""

    def test_finds_same_day_markets(self):
        mapper = PolymarketMarketMapper([DUKE_UNC_GAME, DUKE_UNC_SPREAD])
        game_date = datetime(2026, 4, 9)
        matches = mapper.find_all_markets_for_game("Duke", "UNC", game_date)
        assert len(matches) == 2

    def test_excludes_wrong_date(self):
        # Apr 12 is 3 days from Apr 9, beyond the 1-day tolerance
        far_market = _make_market(
            "Will Duke beat UNC?",
            date_str="2026-04-12T23:00:00Z",
            market_type="moneyline",
            token_id="tok_duke_unc_far",
        )
        mapper = PolymarketMarketMapper([DUKE_UNC_GAME, far_market])
        game_date = datetime(2026, 4, 9)
        matches = mapper.find_all_markets_for_game("Duke", "UNC", game_date)
        assert len(matches) == 1
        assert mapper._get_token_id(matches[0]) == "tok_duke_unc_game"

    def test_one_day_tolerance(self):
        mapper = PolymarketMarketMapper([DUKE_UNC_TOMORROW])
        game_date = datetime(2026, 4, 9)
        matches = mapper.find_all_markets_for_game("Duke", "UNC", game_date)
        assert len(matches) == 1  # Apr 10 is within 1 day of Apr 9


class TestFindGameMarket:
    """Tests for find_game_market filtering to moneyline type."""

    def test_finds_game_market_not_spread(self):
        mapper = PolymarketMarketMapper([DUKE_UNC_GAME, DUKE_UNC_SPREAD])
        result = mapper.find_game_market("Duke", "UNC", datetime(2026, 4, 9))
        assert result is not None
        assert mapper._get_token_id(result) == "tok_duke_unc_game"

    def test_returns_none_when_no_game_market(self):
        mapper = PolymarketMarketMapper([DUKE_UNC_SPREAD])
        result = mapper.find_game_market("Duke", "UNC", datetime(2026, 4, 9))
        assert result is None


class TestFindSpreadMarket:
    """Tests for find_spread_market with spread value matching."""

    def test_finds_spread_market(self):
        mapper = PolymarketMarketMapper([DUKE_UNC_GAME, DUKE_UNC_SPREAD])
        result = mapper.find_spread_market("Duke", "UNC", datetime(2026, 4, 9), -5.5)
        assert result is not None
        assert mapper._get_token_id(result) == "tok_duke_unc_spread"


class TestGameTimeDisambiguation:
    """Tests for doubleheader resolution via game_time parameter."""

    @pytest.fixture
    def doubleheader_markets(self):
        game1 = _make_market(
            "Will Yankees beat Red Sox?",
            date_str="2026-07-15T17:10:00Z",
            market_type="moneyline",
            token_id="tok_g1",
        )
        game2 = _make_market(
            "Will Yankees beat Red Sox?",
            date_str="2026-07-15T23:10:00Z",
            market_type="moneyline",
            token_id="tok_g2",
        )
        return [game1, game2]

    def test_without_game_time_returns_first(self, doubleheader_markets):
        mapper = PolymarketMarketMapper(doubleheader_markets)
        result = mapper.find_game_market(
            "Yankees", "Red Sox", datetime(2026, 7, 15),
        )
        assert mapper._get_token_id(result) == "tok_g1"

    def test_game_time_picks_afternoon_game(self, doubleheader_markets):
        mapper = PolymarketMarketMapper(doubleheader_markets)
        result = mapper.find_game_market(
            "Yankees", "Red Sox", datetime(2026, 7, 15),
            game_time="17:10",
        )
        assert mapper._get_token_id(result) == "tok_g1"

    def test_game_time_picks_evening_game(self, doubleheader_markets):
        mapper = PolymarketMarketMapper(doubleheader_markets)
        result = mapper.find_game_market(
            "Yankees", "Red Sox", datetime(2026, 7, 15),
            game_time="23:10",
        )
        assert mapper._get_token_id(result) == "tok_g2"

    def test_game_time_picks_closest(self, doubleheader_markets):
        mapper = PolymarketMarketMapper(doubleheader_markets)
        # 20:00 is closer to 17:10 (170 min) than 23:10 (190 min)
        result = mapper.find_game_market(
            "Yankees", "Red Sox", datetime(2026, 7, 15),
            game_time="20:00",
        )
        assert mapper._get_token_id(result) == "tok_g1"


class TestTimeDistanceUTC:
    """Verify _time_distance normalizes timezone-aware market times to UTC."""

    def test_utc_aware_market_time(self):
        m = _make_market("T", date_str="2026-04-09T19:00:00Z")
        mapper = PolymarketMarketMapper([m])
        dist = mapper._time_distance(m, "19:00")
        assert dist == 0

    def test_naive_market_time_treated_as_utc(self):
        m = _make_market("T", date_str="2026-04-09T19:00:00")
        mapper = PolymarketMarketMapper([m])
        dist = mapper._time_distance(m, "19:00")
        assert dist == 0

    def test_offset_aware_market_time_converted(self):
        # Market time is 15:00 EST = 20:00 UTC
        m = {"question": "T", "game_start_time": "2026-04-09T15:00:00-05:00"}
        mapper = PolymarketMarketMapper([m])
        dist = mapper._time_distance(m, "20:00")
        assert dist == 0


class TestInferYesTeam:
    """Tests for yes-team inference from market title/outcomes."""

    def test_infer_from_beat_pattern(self):
        mapper = PolymarketMarketMapper([DUKE_UNC_GAME])
        result = mapper.infer_yes_team(DUKE_UNC_GAME, "Duke", "UNC")
        assert result == "Duke"

    def test_infer_from_outcomes_list(self):
        m = _make_market("Game winner?", outcomes=["Kansas", "Baylor"])
        mapper = PolymarketMarketMapper([m])
        result = mapper.infer_yes_team(m, "Kansas", "Baylor")
        assert result == "Kansas"

    def test_returns_none_when_ambiguous(self):
        m = _make_market("Some generic title")
        mapper = PolymarketMarketMapper([m])
        result = mapper.infer_yes_team(m, "TeamA", "TeamB")
        assert result is None


class TestGetMarketPrices:
    """Tests for price extraction in 0-100 cent scale."""

    def test_extracts_prices_from_outcome_prices(self):
        mapper = PolymarketMarketMapper([DUKE_UNC_GAME])
        prices = mapper.get_market_prices(DUKE_UNC_GAME)
        assert prices["yes_price"] == 55.0
        assert prices["no_price"] == 45.0
        assert prices["token_id"] == "tok_duke_unc_game"
        assert "Duke" in prices["title"]
