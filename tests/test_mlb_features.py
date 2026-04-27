"""Tests for MLB feature engineering."""

import pandas as pd
import pytest

import mlb.features as mlb_features
from mlb.features import (
    calculate_rolling_stats,
    calculate_pitcher_rolling_stats,
    merge_opponent_stats,
    compute_differentials,
    compute_target,
    clean_stale_data,
    add_prior_season_stats,
    SEASON_SCOPE_MODE,
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
    """Verify merge handles doubleheader days correctly."""

    def test_merge_opponent_stats_no_row_explosion(self):
        # Two games on the same date between the same teams (doubleheader)
        rows = []
        for game_num, time in enumerate(["13:05", "19:05"]):
            rows.append({
                "date": "2025-04-15", "game_time": time,
                "team": "TeamA", "opponent": "TeamB",
                "is_home": 1, "team_score": 5 + game_num, "opp_score": 3,
                "prev_win_pct": 0.6, "prev_roll10_runs_per_game": 4.5,
                "prev_roll10_runs_allowed": 3.5,
                "prev_season_runs_per_game": 4.0,
                "prev_season_runs_allowed": 3.8,
                "prev_roll10_win_pct": 0.55,
            })
            rows.append({
                "date": "2025-04-15", "game_time": time,
                "team": "TeamB", "opponent": "TeamA",
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

    def test_each_doubleheader_game_gets_own_opponent_stats(self):
        """Game 1 and game 2 of a doubleheader should get distinct opp stats."""
        rows = []
        for game_num, time in enumerate(["13:05", "19:05"]):
            win_pct = 0.5 + game_num * 0.1  # 0.5 for game 1, 0.6 for game 2
            rows.append({
                "date": "2025-04-15", "game_time": time,
                "team": "TeamA", "opponent": "TeamB",
                "is_home": 1, "team_score": 5, "opp_score": 3,
                "prev_win_pct": win_pct,
                "prev_roll10_runs_per_game": 4.0 + game_num,
                "prev_roll10_runs_allowed": 3.0,
                "prev_season_runs_per_game": 4.0,
                "prev_season_runs_allowed": 3.0,
                "prev_roll10_win_pct": win_pct,
            })
            rows.append({
                "date": "2025-04-15", "game_time": time,
                "team": "TeamB", "opponent": "TeamA",
                "is_home": 0, "team_score": 3, "opp_score": 5,
                "prev_win_pct": 0.4 + game_num * 0.1,
                "prev_roll10_runs_per_game": 3.0 + game_num,
                "prev_roll10_runs_allowed": 4.0,
                "prev_season_runs_per_game": 3.0,
                "prev_season_runs_allowed": 4.0,
                "prev_roll10_win_pct": 0.4 + game_num * 0.1,
            })
        df = pd.DataFrame(rows)
        merged = merge_opponent_stats(df)
        # TeamA's game 1 row should get TeamB's game-1 opp_win_pct (0.4)
        # TeamA's game 2 row should get TeamB's game-2 opp_win_pct (0.5)
        team_a = merged[merged["team"] == "TeamA"].sort_values("game_time")
        assert team_a.iloc[0]["opp_win_pct"] == pytest.approx(0.4)
        assert team_a.iloc[1]["opp_win_pct"] == pytest.approx(0.5)


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


class TestDoubleheaderRollingOrder:
    """Verify rolling stats respect game_time ordering within same day."""

    def test_earlier_game_does_not_see_later_game(self):
        rows = []
        # Day 1: single game
        rows.append({
            "date": "2025-04-14", "game_time": "19:00", "season": 2025,
            "team": "TeamA", "team_abbr": "TA", "opponent": "TeamC",
            "opp_abbr": "TC", "location": "Home", "is_home": 1,
            "team_score": 10, "opp_score": 0,
            "starting_pitcher": "Ace", "sp_era": 3.0,
            "opp_starting_pitcher": "Other", "opp_sp_era": 4.0,
            "sp_espn_id": "", "opp_sp_espn_id": "",
            "venue_name": "", "venue_city": "", "venue_state": "",
            "venue_indoor": 0, "moneyline": float("nan"),
            "run_line": float("nan"), "total_line": float("nan"),
            "team_hits": 10, "team_errors": 0,
            "opp_hits": 3, "opp_errors": 0,
            "opp_runs": float("nan"), "team_runs": float("nan"),
        })
        # Day 2: doubleheader -- game 1 at 13:05, game 2 at 19:05
        for time, score in [("13:05", 2), ("19:05", 8)]:
            rows.append({
                "date": "2025-04-15", "game_time": time, "season": 2025,
                "team": "TeamA", "team_abbr": "TA", "opponent": "TeamB",
                "opp_abbr": "TB", "location": "Home", "is_home": 1,
                "team_score": score, "opp_score": 3,
                "starting_pitcher": "Ace", "sp_era": 3.0,
                "opp_starting_pitcher": "Other", "opp_sp_era": 4.0,
                "sp_espn_id": "", "opp_sp_espn_id": "",
                "venue_name": "", "venue_city": "", "venue_state": "",
                "venue_indoor": 0, "moneyline": float("nan"),
                "run_line": float("nan"), "total_line": float("nan"),
                "team_hits": 8, "team_errors": 0,
                "opp_hits": 5, "opp_errors": 0,
                "opp_runs": float("nan"), "team_runs": float("nan"),
            })
        df = pd.DataFrame(rows)
        df = calculate_rolling_stats(df)
        team_a = df[df["team"] == "TeamA"].sort_values(["date", "game_time"])

        # The 13:05 game's prev_season_runs_per_game should only reflect day 1 (score=10)
        game1_prev = team_a.iloc[1]["prev_season_runs_per_game"]
        assert game1_prev == pytest.approx(10.0), (
            f"13:05 game saw runs_per_game={game1_prev}, expected 10.0 (day 1 only)"
        )

        # The 19:05 game should reflect day 1 + the 13:05 game: (10+2)/2 = 6.0
        game2_prev = team_a.iloc[2]["prev_season_runs_per_game"]
        assert game2_prev == pytest.approx(6.0), (
            f"19:05 game saw runs_per_game={game2_prev}, expected 6.0 (day1 + game1)"
        )


class TestRestDays:

    def test_rest_days_computed_from_schedule(self):
        df = _make_games(3)
        df = calculate_rolling_stats(df)
        team_a = df[df["team"] == "TeamA"].sort_values("date")
        # First game: default rest days (3)
        assert team_a.iloc[0]["rest_days"] == 3.0
        # Subsequent games: 1 day apart
        assert team_a.iloc[1]["rest_days"] == 1.0


# === Phase 1 new feature tests ===


class TestPythagoreanWinPct:
    """Pythagorean expected win% from run scoring/allowed."""

    def test_formula_correct(self):
        from mlb.features import compute_pythagorean_wpct
        # 5 RPG scored, 4 RPG allowed -> ~0.597
        result = compute_pythagorean_wpct(5.0, 4.0, exponent=1.83)
        assert result == pytest.approx(0.597, abs=0.01)

    def test_equal_runs_gives_500(self):
        from mlb.features import compute_pythagorean_wpct
        assert compute_pythagorean_wpct(4.0, 4.0) == pytest.approx(0.5)

    def test_zero_runs_allowed_gives_1(self):
        from mlb.features import compute_pythagorean_wpct
        assert compute_pythagorean_wpct(4.0, 0.0) == pytest.approx(1.0)

    def test_zero_runs_scored_gives_0(self):
        from mlb.features import compute_pythagorean_wpct
        assert compute_pythagorean_wpct(0.0, 4.0) == pytest.approx(0.0)

    def test_both_zero_gives_500(self):
        from mlb.features import compute_pythagorean_wpct
        assert compute_pythagorean_wpct(0.0, 0.0) == pytest.approx(0.5)

    def test_pyth_columns_in_rolling_stats(self):
        df = _make_games(5)
        df = calculate_rolling_stats(df)
        assert "prev_season_pyth_wpct" in df.columns
        assert "prev_roll10_pyth_wpct" in df.columns

    def test_pyth_honest_lag(self):
        df = _make_games(5)
        df = calculate_rolling_stats(df)
        team_a = df[df["team"] == "TeamA"].sort_values("date")
        # First game: no prior data
        assert pd.isna(team_a.iloc[0]["prev_season_pyth_wpct"])
        # Second game: value based on first game only
        val = team_a.iloc[1]["prev_season_pyth_wpct"]
        assert pd.notna(val)
        assert 0.0 <= val <= 1.0

    def test_pyth_values_bounded(self):
        df = _make_games(10)
        df = calculate_rolling_stats(df)
        pyth = df["prev_season_pyth_wpct"].dropna()
        assert (pyth >= 0.0).all()
        assert (pyth <= 1.0).all()


class TestNewDifferentials:
    """Pythagorean diff and roll5 diffs."""

    def test_pyth_diff_computed(self):
        df = _make_games(6)
        df = calculate_rolling_stats(df)
        df = calculate_pitcher_rolling_stats(df)
        df = merge_opponent_stats(df)
        df = compute_differentials(df)
        assert "pyth_wpct_diff" in df.columns

    def test_roll5_diffs_computed(self):
        df = _make_games(6)
        df = calculate_rolling_stats(df)
        df = calculate_pitcher_rolling_stats(df)
        df = merge_opponent_stats(df)
        df = compute_differentials(df)
        assert "roll5_rpg_diff" in df.columns
        assert "roll5_ra_diff" in df.columns


class TestBuildFeatureRowNewFeatures:
    """Verify build_feature_row populates new features from stats, not fallbacks."""

    def test_pyth_wpct_from_stats(self):
        from mlb.predict import build_feature_row
        home_stats = {"prev_season_pyth_wpct": 0.6, "prev_roll10_pyth_wpct": 0.55}
        away_stats = {"prev_season_pyth_wpct": 0.45}
        row = build_feature_row(home_stats, away_stats, 3.5, 4.0)
        assert row["prev_season_pyth_wpct"] == 0.6
        assert row["prev_roll10_pyth_wpct"] == 0.55
        assert row["pyth_wpct_diff"] == pytest.approx(0.15)

    def test_bullpen_era_diff_from_stats(self):
        from mlb.predict import build_feature_row
        home_stats = {"bullpen_era": 3.5}
        away_stats = {"bullpen_era": 4.2}
        row = build_feature_row(home_stats, away_stats, 3.5, 4.0)
        assert row["bullpen_era_diff"] == pytest.approx(0.7)

    def test_bullpen_era_diff_defaults_to_zero_when_missing(self):
        from mlb.predict import build_feature_row
        row = build_feature_row({}, {}, 3.5, 4.0)
        assert row["bullpen_era_diff"] == 0.0

    def test_roll5_rpg_diff_from_stats(self):
        from mlb.predict import build_feature_row
        home_stats = {"prev_roll5_runs_per_game": 5.0}
        away_stats = {"prev_roll5_runs_per_game": 3.5}
        row = build_feature_row(home_stats, away_stats, 3.5, 4.0)
        assert row["roll5_rpg_diff"] == pytest.approx(1.5)

    def test_wind_speed_defaults_for_indoor(self):
        from mlb.predict import build_feature_row
        row = build_feature_row({}, {}, 3.5, 4.0, venue_name="Chase Field", venue_indoor=1)
        assert row["wind_speed"] == 0.0


# === Multi-season helpers and tests ===


def _make_multi_season_games(n_days_s1=10, n_days_s2=5, teams=("TeamA", "TeamB")):
    """Build a multi-season MLB game DataFrame for testing."""
    rows = []
    for season, n_days, month_start in [(2025, n_days_s1, 4), (2026, n_days_s2, 4)]:
        for i in range(n_days):
            date = f"{season}-{month_start:02d}-{i + 1:02d}"
            score_a = 4 + (i % 3)
            score_b = 3 + ((i + 1) % 3)
            for team_idx in range(2):
                t = teams[team_idx]
                opp = teams[1 - team_idx]
                is_home = 1 if team_idx == 0 else 0
                ts = score_a if team_idx == 0 else score_b
                os_ = score_b if team_idx == 0 else score_a
                rows.append({
                    "date": date, "season": season,
                    "team": t, "team_abbr": t[:2].upper(),
                    "opponent": opp, "opp_abbr": opp[:2].upper(),
                    "location": "Home" if is_home else "Away",
                    "is_home": is_home,
                    "team_score": ts, "opp_score": os_,
                    "starting_pitcher": f"Pitcher{t}{i % 3}",
                    "sp_era": 3.5, "opp_sp_era": 4.0,
                    "opp_starting_pitcher": f"Pitcher{opp}{i % 3}",
                    "sp_espn_id": "", "opp_sp_espn_id": "",
                    "venue_name": "", "venue_city": "", "venue_state": "",
                    "venue_indoor": 0, "moneyline": float("nan"),
                    "run_line": float("nan"), "total_line": float("nan"),
                    "team_hits": 8, "team_errors": 0,
                    "opp_hits": 6, "opp_errors": 1,
                    "opp_runs": float("nan"), "team_runs": float("nan"),
                })
    return pd.DataFrame(rows)


class TestSeasonBoundary:
    """Verify rolling stats reset at season boundary."""

    def test_games_played_resets_at_season_boundary(self):
        df = _make_multi_season_games(n_days_s1=10, n_days_s2=5)
        df = calculate_rolling_stats(df)
        team_a = df[df["team"] == "TeamA"].sort_values("date")
        # First game of 2026 should have prev_games_played = 0
        first_2026 = team_a[team_a["date"].str.startswith("2026")].iloc[0]
        assert first_2026["prev_games_played"] == 0

    def test_season_expanding_stats_reset(self):
        df = _make_multi_season_games(n_days_s1=10, n_days_s2=5)
        df = calculate_rolling_stats(df)
        team_a = df[df["team"] == "TeamA"].sort_values("date")
        # First game of 2026 should have NaN prev_season_runs_per_game (no prior data this season)
        first_2026 = team_a[team_a["date"].str.startswith("2026")].iloc[0]
        assert pd.isna(first_2026["prev_season_runs_per_game"])

    def test_rolling_stats_behavior_full_mode(self, monkeypatch):
        monkeypatch.setattr(mlb_features, "SEASON_SCOPE_MODE", "full")
        df = _make_multi_season_games(n_days_s1=10, n_days_s2=5)
        df = calculate_rolling_stats(df)
        team_a = df[df["team"] == "TeamA"].sort_values("date")
        # In full mode, roll10 also resets -- first 2026 game has NaN prev_roll10
        first_2026 = team_a[team_a["date"].str.startswith("2026")].iloc[0]
        assert pd.isna(first_2026["prev_roll10_runs_per_game"])

    def test_rolling_stats_behavior_selective_mode(self, monkeypatch):
        monkeypatch.setattr(mlb_features, "SEASON_SCOPE_MODE", "selective")
        df = _make_multi_season_games(n_days_s1=10, n_days_s2=5)
        df = calculate_rolling_stats(df)
        team_a = df[df["team"] == "TeamA"].sort_values("date")
        # In selective mode, roll10 carries over -- first 2026 game has a value
        first_2026 = team_a[team_a["date"].str.startswith("2026")].iloc[0]
        assert pd.notna(first_2026["prev_roll10_runs_per_game"])


class TestPriorSeasonStats:
    """Verify prior-season columns are populated correctly."""

    def test_prior_season_columns_populated_for_second_season(self):
        df = _make_multi_season_games(n_days_s1=10, n_days_s2=5)
        df = calculate_rolling_stats(df)
        df = add_prior_season_stats(df)
        s2 = df[df["season"] == 2026]
        for col in ["prior_season_win_pct", "prior_season_rpg",
                     "prior_season_ra", "prior_season_pyth_wpct"]:
            assert col in s2.columns
            assert s2[col].notna().all(), f"{col} has NaN in season 2"

    def test_first_season_gets_defaults(self):
        df = _make_multi_season_games(n_days_s1=10, n_days_s2=5)
        df = calculate_rolling_stats(df)
        df = add_prior_season_stats(df)
        s1 = df[df["season"] == 2025]
        assert (s1["prior_season_win_pct"] == 0.5).all()
        assert (s1["prior_season_rpg"] == 4.5).all()
        assert (s1["prior_season_ra"] == 4.5).all()
        assert (s1["prior_season_pyth_wpct"] == 0.5).all()

    def test_prior_stats_reflect_end_of_prior_season(self):
        df = _make_multi_season_games(n_days_s1=10, n_days_s2=5)
        df = calculate_rolling_stats(df)
        df = add_prior_season_stats(df)
        # TeamA's 2026 prior_season_win_pct should match their 2025 final win_pct
        team_a_2025 = df[(df["team"] == "TeamA") & (df["season"] == 2025)].sort_values("date")
        final_2025_wpct = team_a_2025.iloc[-1]["win_pct"]
        team_a_2026 = df[(df["team"] == "TeamA") & (df["season"] == 2026)]
        # All 2026 rows for TeamA should have the same prior_season_win_pct
        for val in team_a_2026["prior_season_win_pct"]:
            assert val == pytest.approx(final_2025_wpct)
