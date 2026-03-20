"""Tests for recent Kalshi results filtering used by the dashboard."""

from datetime import datetime, timedelta

from dashboard_helpers import filter_recent_kalshi


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _days_ago(n):
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


class TestFilterRecentKalshi:
    """Verify the filter picks up settled Kalshi results within the window."""

    def test_includes_todays_settled_win(self):
        bets = [
            {"platform": "Kalshi", "date": _today(), "result": "win",
             "game": "A at B", "line": "B ML", "profit": "0.41"},
        ]
        assert len(filter_recent_kalshi(bets)) == 1

    def test_includes_todays_settled_loss(self):
        bets = [
            {"platform": "Kalshi", "date": _today(), "result": "loss",
             "game": "C at D", "line": "D ML", "profit": "-0.57"},
        ]
        assert len(filter_recent_kalshi(bets)) == 1

    def test_includes_void(self):
        bets = [
            {"platform": "Kalshi", "date": _today(), "result": "void",
             "game": "E at F", "line": "F ML", "profit": "0.00"},
        ]
        assert len(filter_recent_kalshi(bets)) == 1

    def test_excludes_pending(self):
        bets = [
            {"platform": "Kalshi", "date": _today(), "result": "pending",
             "game": "G at H", "line": "H ML", "profit": ""},
        ]
        assert len(filter_recent_kalshi(bets)) == 0

    def test_excludes_non_kalshi_platform(self):
        bets = [
            {"platform": "FanDuel", "date": _today(), "result": "win",
             "game": "I vs J", "line": "I -3.5", "profit": "1.74"},
        ]
        assert len(filter_recent_kalshi(bets)) == 0

    def test_excludes_results_older_than_window(self):
        bets = [
            {"platform": "Kalshi", "date": _days_ago(8), "result": "win",
             "game": "K at L", "line": "L ML", "profit": "0.30"},
        ]
        assert len(filter_recent_kalshi(bets)) == 0

    def test_includes_result_at_window_boundary(self):
        bets = [
            {"platform": "Kalshi", "date": _days_ago(7), "result": "win",
             "game": "M at N", "line": "N ML", "profit": "0.20"},
        ]
        # 7 days ago is exactly at the cutoff -- should still be included
        assert len(filter_recent_kalshi(bets)) == 1

    def test_sorted_by_date_descending(self):
        bets = [
            {"platform": "Kalshi", "date": _days_ago(3), "result": "win",
             "game": "O at P", "line": "P ML", "profit": "0.10"},
            {"platform": "Kalshi", "date": _today(), "result": "loss",
             "game": "Q at R", "line": "R ML", "profit": "-0.50"},
            {"platform": "Kalshi", "date": _days_ago(1), "result": "win",
             "game": "S at T", "line": "T ML", "profit": "0.40"},
        ]
        result = filter_recent_kalshi(bets)
        assert [r["date"] for r in result] == sorted(
            [r["date"] for r in result], reverse=True
        )

    def test_limited_to_8_results(self):
        bets = [
            {"platform": "Kalshi", "date": _today(), "result": "win",
             "game": f"Team{i} at TeamX", "line": "TeamX ML", "profit": "0.10"}
            for i in range(12)
        ]
        assert len(filter_recent_kalshi(bets)) == 8

    def test_handles_mixed_case_platform(self):
        bets = [
            {"platform": "KALSHI", "date": _today(), "result": "win",
             "game": "U at V", "line": "V ML", "profit": "0.10"},
            {"platform": "kalshi", "date": _today(), "result": "loss",
             "game": "W at X", "line": "X ML", "profit": "-0.30"},
        ]
        assert len(filter_recent_kalshi(bets)) == 2

    def test_handles_whitespace_in_platform_and_result(self):
        bets = [
            {"platform": " Kalshi ", "date": _today(), "result": " win ",
             "game": "Y at Z", "line": "Z ML", "profit": "0.50"},
        ]
        assert len(filter_recent_kalshi(bets)) == 1

    def test_empty_input(self):
        assert filter_recent_kalshi([]) == []

    def test_missing_fields_gracefully_excluded(self):
        bets = [
            {"platform": "Kalshi"},  # missing date and result
            {"date": _today(), "result": "win"},  # missing platform
        ]
        assert len(filter_recent_kalshi(bets)) == 0

    def test_custom_cutoff_days(self):
        bets = [
            {"platform": "Kalshi", "date": _days_ago(5), "result": "win",
             "game": "A at B", "line": "B ML", "profit": "0.10"},
        ]
        assert len(filter_recent_kalshi(bets, cutoff_days=3)) == 0
        assert len(filter_recent_kalshi(bets, cutoff_days=7)) == 1

    def test_custom_limit(self):
        bets = [
            {"platform": "Kalshi", "date": _today(), "result": "win",
             "game": f"T{i} at TX", "line": "TX ML", "profit": "0.10"}
            for i in range(5)
        ]
        assert len(filter_recent_kalshi(bets, limit=3)) == 3

    def test_none_field_values_gracefully_excluded(self):
        bets = [
            {"platform": None, "date": _today(), "result": "win"},
            {"platform": "Kalshi", "date": None, "result": "win"},
            {"platform": "Kalshi", "date": _today(), "result": None},
        ]
        assert len(filter_recent_kalshi(bets)) == 0
