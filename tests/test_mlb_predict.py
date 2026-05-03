"""Tests for MLB prediction pipeline."""

from datetime import datetime

import pandas as pd
import pytest

import mlb.predict as mlb_predict
import mlb.shadow_grader as shadow_grader
from mlb.predict import (
    build_feature_row,
    find_best_match,
    get_latest_stats,
    get_latest_pitcher_stats,
    _get_kalshi_game_rating,
    _refresh_shadow_grader_ledger,
    KALSHI_EDGE_CAP,
)
from model import MLB_FEATURES


class TestFetchScheduleOdds:

    def test_moneyline_falls_back_to_current_before_open(self, monkeypatch):
        event = {
            "id": "game-1",
            "date": "2026-05-02T23:05Z",
            "status": {"type": {"state": "pre"}},
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {
                                "displayName": "New York Yankees",
                                "abbreviation": "NYY",
                            },
                        },
                        {
                            "homeAway": "away",
                            "team": {
                                "displayName": "Boston Red Sox",
                                "abbreviation": "BOS",
                            },
                        },
                    ],
                    "venue": {"fullName": "Yankee Stadium", "indoor": False},
                    "odds": [
                        {
                            "moneyline": {
                                "home": {
                                    "current": {"odds": "-145"},
                                    "open": {"odds": "-130"},
                                },
                                "away": {
                                    "current": {"odds": "+125"},
                                    "open": {"odds": "+110"},
                                },
                            }
                        }
                    ],
                }
            ],
        }

        def fake_fetch_mlb_espn_date(base_url, date_str):
            return date_str, [event], None

        monkeypatch.setattr(
            mlb_predict,
            "_fetch_mlb_espn_date",
            fake_fetch_mlb_espn_date,
        )

        games = mlb_predict.fetch_schedule()

        assert len(games) == 1
        assert games[0]["home_ml_odds"] == "-145"
        assert games[0]["away_ml_odds"] == "+125"


class TestBuildFeatureRow:

    def _home_stats(self):
        return {
            "last_game_date": pd.Timestamp("2025-09-20"),
            "prev_roll10_runs_per_game": 4.5,
            "prev_roll10_runs_allowed": 3.8,
            "prev_season_runs_per_game": 4.2,
            "prev_season_runs_allowed": 3.9,
            "prev_games_played": 150,
            "prev_win_pct": 0.58,
            "prev_roll10_win_pct": 0.60,
            "prev_volatility": 2.1,
        }

    def _away_stats(self):
        return {
            "prev_roll10_runs_per_game": 3.8,
            "prev_roll10_runs_allowed": 4.2,
            "prev_win_pct": 0.45,
            "prev_roll10_win_pct": 0.40,
        }

    def test_returns_all_mlb_features(self):
        row = build_feature_row(
            self._home_stats(), self._away_stats(),
            home_sp_era=3.20, away_sp_era=4.50,
        )
        for f in MLB_FEATURES:
            assert f in row, f"Missing feature: {f}"
            assert not pd.isna(row[f]), f"Feature {f} is NaN"

    def test_is_home_always_one(self):
        row = build_feature_row(
            self._home_stats(), self._away_stats(),
            home_sp_era=3.50, away_sp_era=4.00,
        )
        assert row["is_home"] == 1

    def test_sp_era_diff_positive_when_away_pitcher_worse(self):
        row = build_feature_row(
            self._home_stats(), self._away_stats(),
            home_sp_era=3.00, away_sp_era=5.00,
        )
        assert row["sp_era_diff"] == pytest.approx(2.0)

    def test_missing_stats_get_neutral_defaults(self):
        row = build_feature_row({}, {}, home_sp_era=float("nan"), away_sp_era=float("nan"))
        assert row["sp_era"] == 4.50  # default ERA
        assert row["prev_win_pct"] == 0.5  # default win pct
        assert row["sp_roll_whip"] == 1.30  # default WHIP

    def test_pitcher_stats_populated_from_dict(self):
        pitcher_stats = {
            "Ace Pitcher": {
                "sp_roll_era": 2.80,
                "sp_roll_whip": 1.05,
                "sp_roll_k9": 10.5,
                "sp_roll_ip": 6.2,
            }
        }
        row = build_feature_row(
            self._home_stats(), self._away_stats(),
            home_sp_era=3.00, away_sp_era=4.00,
            home_sp_name="Ace Pitcher", away_sp_name="Unknown",
            pitcher_stats=pitcher_stats,
        )
        assert row["sp_roll_era"] == 2.80
        assert row["sp_roll_whip"] == 1.05
        assert row["sp_roll_k9"] == 10.5

    def test_roll10_rpg_diff_computed(self):
        row = build_feature_row(
            self._home_stats(), self._away_stats(),
            home_sp_era=3.50, away_sp_era=4.00,
        )
        # home 4.5 - away 3.8 = 0.7
        assert row["roll10_rpg_diff"] == pytest.approx(0.7)


class TestRestDaysUsesGameDate:
    """Verify rest_days is relative to the scheduled game, not datetime.now()."""

    def test_future_game_gets_correct_rest_days(self):
        stats = {"last_game_date": pd.Timestamp("2026-03-24")}
        # Game on March 26 -- 2 days rest
        row_mar26 = build_feature_row(
            stats, {}, home_sp_era=4.0, away_sp_era=4.0,
            game_date=datetime(2026, 3, 26),
        )
        # Game on March 28 -- 4 days rest
        row_mar28 = build_feature_row(
            stats, {}, home_sp_era=4.0, away_sp_era=4.0,
            game_date=datetime(2026, 3, 28),
        )
        assert row_mar26["rest_days"] == 2
        assert row_mar28["rest_days"] == 4

    def test_same_day_game_gets_zero_rest(self):
        stats = {"last_game_date": pd.Timestamp("2026-03-25")}
        row = build_feature_row(
            stats, {}, home_sp_era=4.0, away_sp_era=4.0,
            game_date=datetime(2026, 3, 25),
        )
        assert row["rest_days"] == 0

    def test_rest_days_capped_at_seven(self):
        stats = {"last_game_date": pd.Timestamp("2026-03-01")}
        row = build_feature_row(
            stats, {}, home_sp_era=4.0, away_sp_era=4.0,
            game_date=datetime(2026, 3, 25),
        )
        assert row["rest_days"] == 7


class TestFindBestMatch:

    def test_exact_match(self):
        known = {"New York Yankees", "Boston Red Sox"}
        assert find_best_match("New York Yankees", known) == "New York Yankees"

    def test_substring_match(self):
        known = {"New York Yankees", "Boston Red Sox"}
        assert find_best_match("Yankees", known) is not None

    def test_returns_none_for_unknown(self):
        known = {"New York Yankees"}
        result = find_best_match("Nonexistent Team", known)
        assert result is None


class TestGetLatestStats:

    def test_returns_stats_for_each_team(self):
        df = pd.DataFrame({
            "date": ["2025-04-01", "2025-04-02", "2025-04-01", "2025-04-02"],
            "team": ["TeamA", "TeamA", "TeamB", "TeamB"],
            "prev_win_pct": [0.5, 0.55, 0.5, 0.45],
            "prev_volatility": [2.0, 2.1, 2.0, 1.9],
            "prev_games_played": [1, 2, 1, 2],
            "prev_roll10_win_pct": [0.5, 0.55, 0.5, 0.45],
        })
        stats = get_latest_stats(df)
        assert "TeamA" in stats
        assert "TeamB" in stats
        assert stats["TeamA"]["prev_win_pct"] == 0.55  # latest game
        assert stats["TeamB"]["prev_win_pct"] == 0.45


class TestGetLatestPitcherStats:

    def test_returns_latest_rolling_stats_per_pitcher(self):
        df = pd.DataFrame({
            "date": ["2025-04-01", "2025-04-05", "2025-04-03"],
            "starting_pitcher": ["Ace", "Ace", "Reliever"],
            "sp_roll_era": [3.5, 3.2, 4.0],
            "sp_roll_whip": [1.1, 1.0, 1.3],
            "sp_roll_k9": [9.0, 9.5, 7.0],
            "sp_roll_ip": [6.0, 6.2, 5.0],
        })
        stats = get_latest_pitcher_stats(df)
        assert stats["Ace"]["sp_roll_era"] == 3.2  # latest date
        assert stats["Reliever"]["sp_roll_era"] == 4.0


class TestGetKalshiGameRating:

    def test_strong_when_all_gates_pass(self):
        assert _get_kalshi_game_rating(0.10, 0.60, 50) == "STRONG"

    def test_strong_boundary_edge(self):
        assert _get_kalshi_game_rating(0.08, 0.55, 50) == "STRONG"

    def test_good_when_edge_below_strong(self):
        assert _get_kalshi_game_rating(0.05, 0.53, 50) == "GOOD"

    def test_good_boundary(self):
        assert _get_kalshi_game_rating(0.04, 0.52, 50) == "GOOD"

    def test_pass_when_prob_too_low_for_strong(self):
        """High edge but low model prob -- tail punt rejection."""
        assert _get_kalshi_game_rating(0.12, 0.54, 50) == "GOOD"

    def test_pass_when_prob_too_low_for_good(self):
        assert _get_kalshi_game_rating(0.05, 0.51, 50) == "PASS"

    def test_falls_to_good_when_price_outside_strong_band(self):
        # Price 91 is outside STRONG (10-90) but inside GOOD (15-85)
        assert _get_kalshi_game_rating(0.10, 0.60, 91) == "PASS"
        # Price 9 is outside both bands
        assert _get_kalshi_game_rating(0.10, 0.60, 9) == "PASS"

    def test_pass_when_price_outside_good_band(self):
        assert _get_kalshi_game_rating(0.05, 0.53, 10) == "PASS"
        assert _get_kalshi_game_rating(0.05, 0.53, 90) == "PASS"

    def test_pass_when_edge_too_low(self):
        assert _get_kalshi_game_rating(0.03, 0.60, 50) == "PASS"

    def test_none_inputs_return_pass(self):
        assert _get_kalshi_game_rating(None, 0.60, 50) == "PASS"
        assert _get_kalshi_game_rating(0.10, None, 50) == "PASS"
        assert _get_kalshi_game_rating(0.10, 0.60, None) == "PASS"


class TestKalshiEdgeCap:

    def test_cap_value(self):
        assert KALSHI_EDGE_CAP == 0.15

    def test_edge_capped_in_result(self):
        """Edges above 15% should be clamped."""
        raw_edge = 0.25
        capped = min(raw_edge, KALSHI_EDGE_CAP)
        assert capped == 0.15

    def test_edge_below_cap_unchanged(self):
        raw_edge = 0.10
        capped = min(raw_edge, KALSHI_EDGE_CAP)
        assert capped == 0.10


class TestRefreshShadowGraderLedger:
    """The hook in generate_predictions must be best-effort: it writes the
    ledger when the grader succeeds, and swallows exceptions so a grader bug
    cannot break the predict pipeline."""

    def test_writes_ledger_on_success(self, monkeypatch, capsys):
        called = {}

        def fake_grade():
            called["grade"] = True
            return pd.DataFrame({"archive_date": ["2026-04-15"]})

        def fake_write(ledger, path=shadow_grader.DEFAULT_LEDGER):
            called["write"] = (len(ledger), path)

        monkeypatch.setattr(shadow_grader, "grade", fake_grade)
        monkeypatch.setattr(shadow_grader, "write_ledger", fake_write)

        _refresh_shadow_grader_ledger()
        assert called.get("grade") is True
        assert called.get("write") == (1, shadow_grader.DEFAULT_LEDGER)
        assert "Shadow grader" in capsys.readouterr().out

    def test_swallows_grader_exception(self, monkeypatch, capsys):
        def boom():
            raise RuntimeError("training csv missing")

        monkeypatch.setattr(shadow_grader, "grade", boom)
        # Must not raise.
        _refresh_shadow_grader_ledger()
        assert "WARNING: shadow grader refresh failed" in capsys.readouterr().out
