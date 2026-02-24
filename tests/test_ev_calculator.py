"""Tests for betting.ev_calculator edge rating logic."""

from betting.ev_calculator import (
    EdgeRating,
    get_rating,
    VALUE_RATINGS,
    RATING_RANK,
    STRONG_THRESHOLD,
    GOOD_THRESHOLD,
    MARGINAL_THRESHOLD,
)


class TestGetRating:
    """Boundary tests for get_rating()."""

    def test_strong_at_threshold(self):
        assert get_rating(0.08) == EdgeRating.STRONG

    def test_strong_above_threshold(self):
        assert get_rating(0.12) == EdgeRating.STRONG

    def test_good_at_threshold(self):
        assert get_rating(0.04) == EdgeRating.GOOD

    def test_good_between_thresholds(self):
        assert get_rating(0.06) == EdgeRating.GOOD

    def test_good_just_below_strong(self):
        assert get_rating(0.0799) == EdgeRating.GOOD

    def test_marginal_at_threshold(self):
        assert get_rating(0.02) == EdgeRating.MARGINAL

    def test_marginal_between_thresholds(self):
        assert get_rating(0.03) == EdgeRating.MARGINAL

    def test_marginal_just_below_good(self):
        assert get_rating(0.0399) == EdgeRating.MARGINAL

    def test_pass_just_below_marginal(self):
        assert get_rating(0.0199) == EdgeRating.PASS

    def test_pass_zero_edge(self):
        assert get_rating(0.0) == EdgeRating.PASS

    def test_pass_negative_edge(self):
        assert get_rating(-0.05) == EdgeRating.PASS


class TestEdgeRatingValues:
    """Verify enum string values match what CSV consumers expect."""

    def test_strong_value(self):
        assert EdgeRating.STRONG.value == "STRONG"

    def test_good_value(self):
        assert EdgeRating.GOOD.value == "GOOD"

    def test_marginal_value(self):
        assert EdgeRating.MARGINAL.value == "MARGINAL"

    def test_pass_value(self):
        assert EdgeRating.PASS.value == "PASS"


class TestConstants:
    """Verify centralized constants are consistent."""

    def test_value_ratings_includes_strong_and_good(self):
        assert "STRONG" in VALUE_RATINGS
        assert "GOOD" in VALUE_RATINGS

    def test_value_ratings_excludes_marginal_and_pass(self):
        assert "MARGINAL" not in VALUE_RATINGS
        assert "PASS" not in VALUE_RATINGS

    def test_rating_rank_ordering(self):
        assert RATING_RANK["STRONG"] > RATING_RANK["GOOD"]
        assert RATING_RANK["GOOD"] > RATING_RANK["MARGINAL"]
        assert RATING_RANK["MARGINAL"] > RATING_RANK["PASS"]

    def test_thresholds_descending(self):
        assert STRONG_THRESHOLD > GOOD_THRESHOLD > MARGINAL_THRESHOLD > 0
