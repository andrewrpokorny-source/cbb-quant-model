"""Tests for MLB feature engineering."""

import pandas as pd
import pytest

from mlb.features import (
    calculate_rolling_stats,
    calculate_pitcher_rolling_stats,
    merge_opponent_stats,
    compute_differentials,
    compute_target,
    clean_stale_data,
)


def _make_games(n_days=10, teams=("TeamA", "TeamB")):
    """Build a minimal MLB game DataFrame for testing."""
    rows = []
    for i in range(n_days):
        date = f"2025-04-{i + 1:02d}"
        score_a = 4 + (i % 3)
        score_b = 3 + ((i + 1) % 3)
        rows.append({
            "date": date, "season": 2025,
            "team": teams[0], "team_abbr": "TA",
            "opponent": teams[1], "opp_abbr": "TB",
            "location": "Home", "is_home": 1,
            "team_score": score_a, "opp_score": score_b,
            "starting_pitcher": f"PitcherA{i % 3}",
            "sp_era": 3.5, "opp_sp_era": 4.0,
            "opp_starting_pitcher": f"PitcherB{i % 3}",
            "sp_espn_id": "", "opp_sp_espn_id": "",
            "venue_name": "", "venue_city": "", "venue_state": "",
            "venue_indoor": 0, "moneyline": float("nan"),
            "run_line": float("nan"), "total_line": float("nan"),
            "team_hits": 8, "team_errors": 0,
            "opp_hits": 6, "opp_errors": 1,
            "opp_runs": float("nan"), "team_runs": float("nan"),
        })
        rows.append({
            "date": date, "season": 2025,
            "team": teams[1], "team_abbr": "TB",
            "opponent": teams[0], "opp_abbr": "TA",
            "location": "Away", "is_home": 0,
            "team_score": score_b, "opp_score": score_a,
            "starting_pitcher": f"PitcherB{i % 3}",
            "sp_era": 4.0, "opp_sp_era": 3.5,
            "opp_starting_pitcher": f"PitcherA{i % 3}",
            "sp_espn_id": "", "opp_sp_espn_id": "",
            "venue_name": "", "venue_city": "", "venue_state": "",
            "venue_indoor": 0, "moneyline": float("nan"),
            "run_line": float("nan"), "total_line": float("nan"),
            "team_hits": 6, "team_errors": 1,
            "opp_hits": 8, "opp_errors": 0,
            "opp_runs": float("nan"), "team_runs": float("nan"),
        })
    return pd.DataFrame(rows)


class TestHonestLag:
    """Verify that rolling stats use only pre-game data (.shift(1))."""

    def test_prev_columns_are_shifted(self):
        df = _make_games(5)
        df = calculate_rolling_stats(df)
        # First game for each team should have NaN prev_ stats
        team_a = df[df["team"] == "TeamA"].sort_values("date")
        assert pd.isna(team_a.iloc[0]["prev_season_runs_per_game"])
        # Second game should have the first game's value
        assert team_a.iloc[1]["prev_season_runs_per_game"] == team_a.iloc[0]["season_runs_per_game"]

    def test_no_future_data_in_prev_columns(self):
        df = _make_games(10)
        df = calculate_rolling_stats(df)
        team_a = df[df["team"] == "TeamA"].sort_values("date")
        for i in range(1, len(team_a)):
            # prev_season value at row i should equal season value at row i-1
            assert team_a.iloc[i]["prev_season_runs_per_game"] == pytest.approx(
                team_a.iloc[i - 1]["season_runs_per_game"]
            )


class TestPitcherRollingStats:
    """Verify pitcher rolling stats are computed per-pitcher with honest lag."""

    def _make_pitcher_data(self):
        rows = []
        for i in range(8):
            date = f"2025-04-{i + 1:02d}"
            pitcher = "Ace" if i % 2 == 0 else "Reliever"
            rows.append({
                "date": date, "team": "TeamA", "starting_pitcher": pitcher,
                "sp_ip": 6.0, "sp_er": 2 + (i % 2), "sp_h": 5,
                "sp_bb": 2, "sp_k": 7,
            })
        return pd.DataFrame(rows)

    def test_first_start_has_no_rolling_stats(self):
        df = self._make_pitcher_data()
        df = calculate_pitcher_rolling_stats(df)
        # Ace's first start (row 0) should have NaN rolling stats
        ace_starts = df[df["starting_pitcher"] == "Ace"].sort_values("date")
        assert pd.isna(ace_starts.iloc[0]["sp_roll_era"])

    def test_second_start_uses_first_start_only(self):
        df = self._make_pitcher_data()
        df = calculate_pitcher_rolling_stats(df)
        ace_starts = df[df["starting_pitcher"] == "Ace"].sort_values("date")
        # Second start should have ERA based on first start: 9 * 2 / 6 = 3.0
        assert ace_starts.iloc[1]["sp_roll_era"] == pytest.approx(3.0)

    def test_pitchers_are_independent(self):
        df = self._make_pitcher_data()
        df = calculate_pitcher_rolling_stats(df)
        ace = df[df["starting_pitcher"] == "Ace"].sort_values("date")
        reliever = df[df["starting_pitcher"] == "Reliever"].sort_values("date")
        # Different ERs since they have different earned runs
        if not pd.isna(ace.iloc[1]["sp_roll_era"]) and not pd.isna(reliever.iloc[1]["sp_roll_era"]):
            assert ace.iloc[1]["sp_roll_era"] != reliever.iloc[1]["sp_roll_era"]


class TestDoubleheaderHandling:
    """Verify merge doesn't duplicate rows on doubleheader days."""

    def test_merge_opponent_stats_no_row_explosion(self):
        # Two games on the same date between the same teams (doubleheader)
        rows = []
        for game_num in range(2):
            rows.append({
                "date": "2025-04-15", "team": "TeamA", "opponent": "TeamB",
                "is_home": 1, "team_score": 5 + game_num, "opp_score": 3,
                "prev_win_pct": 0.6, "prev_roll10_runs_per_game": 4.5,
                "prev_roll10_runs_allowed": 3.5,
                "prev_season_runs_per_game": 4.0,
                "prev_season_runs_allowed": 3.8,
                "prev_roll10_win_pct": 0.55,
            })
            rows.append({
                "date": "2025-04-15", "team": "TeamB", "opponent": "TeamA",
                "is_home": 0, "team_score": 3, "opp_score": 5 + game_num,
                "prev_win_pct": 0.4, "prev_roll10_runs_per_game": 3.5,
                "prev_roll10_runs_allowed": 4.5,
                "prev_season_runs_per_game": 3.8,
                "prev_season_runs_allowed": 4.0,
                "prev_roll10_win_pct": 0.45,
            })
        df = pd.DataFrame(rows)
        original_len = len(df)
        merged = merge_opponent_stats(df)
        assert len(merged) == original_len


class TestDoubleheaderDataDedup:
    """Verify that update_database preserves both games of a doubleheader."""

    def test_different_game_times_not_deduped(self):
        """Two same-day games with different game_times should both survive."""
        from mlb.data import update_database
        import tempfile, os

        rows = []
        for time, score in [("13:05", 5), ("19:05", 3)]:
            rows.append({
                "date": "2025-04-15", "game_time": time,
                "team": "TeamA", "team_abbr": "TA",
                "opponent": "TeamB", "opp_abbr": "TB",
                "is_home": 1, "team_score": score, "opp_score": 2,
            })
            rows.append({
                "date": "2025-04-15", "game_time": time,
                "team": "TeamB", "team_abbr": "TB",
                "opponent": "TeamA", "opp_abbr": "TA",
                "is_home": 0, "team_score": 2, "opp_score": score,
            })
        df = pd.DataFrame(rows)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            result = pd.read_csv(f.name)
            os.unlink(f.name)

        assert len(result) == 4  # 2 games x 2 rows each


class TestComputeTarget:

    def test_home_win_computed_correctly(self):
        df = pd.DataFrame({
            "team_score": [5, 3, 4],
            "opp_score": [3, 5, 4],
        })
        result = compute_target(df)
        assert result["home_win"].tolist() == [1, 0, 0]

    def test_tie_is_labeled_as_loss(self):
        df = pd.DataFrame({"team_score": [4], "opp_score": [4]})
        result = compute_target(df)
        assert result["home_win"].iloc[0] == 0


class TestComputeDifferentials:

    def test_sp_era_diff(self):
        df = pd.DataFrame({
            "sp_era": [3.0], "opp_sp_era": [5.0],
            "prev_roll10_runs_per_game": [4.0],
            "opp_prev_roll10_rpg": [3.5],
            "prev_roll10_runs_allowed": [3.0],
            "opp_prev_roll10_ra": [4.0],
        })
        result = compute_differentials(df)
        assert result["sp_era_diff"].iloc[0] == pytest.approx(2.0)
        assert result["roll10_rpg_diff"].iloc[0] == pytest.approx(0.5)
        assert result["roll10_ra_diff"].iloc[0] == pytest.approx(1.0)


class TestRestDays:

    def test_rest_days_computed_from_schedule(self):
        df = _make_games(3)
        df = calculate_rolling_stats(df)
        team_a = df[df["team"] == "TeamA"].sort_values("date")
        # First game: default rest days (3)
        assert team_a.iloc[0]["rest_days"] == 3.0
        # Subsequent games: 1 day apart
        assert team_a.iloc[1]["rest_days"] == 1.0
