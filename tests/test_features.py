import pandas as pd

from features import calculate_advanced_stats, clean_stale_data, merge_archived_market_spreads, merge_opponent_stats


def test_clean_stale_data_preserves_raw_source_columns():
    df = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "team": ["Home"],
            "opponent": ["Away"],
            "location": ["Home"],
            "team_score": [80],
            "opp_score": [70],
            "spread": [-4.5],
            "is_home": [1],
            "is_neutral": [0],
            "venue_city": [""],
            "venue_state": [""],
            "possessions": [69.5],
            "team_eFG": [0.54],
            "team_TO": [15.2],
            "team_ORB": [28.1],
            "prev_season_eFG": [0.5],
            "diff_eFG": [0.04],
            "ats_win": [1],
        }
    )

    result = clean_stale_data(df)

    assert "possessions" in result.columns
    assert "team_eFG" in result.columns
    assert "team_TO" in result.columns
    assert "team_ORB" in result.columns
    assert "prev_season_eFG" not in result.columns
    assert "diff_eFG" not in result.columns
    assert "ats_win" not in result.columns


def test_calculate_advanced_stats_uses_real_inputs_without_fabrication():
    df = pd.DataFrame(
        {
            "team_score": [84],
            "possessions": [70.0],
            "team_eFG": [0.55],
            "team_TO": [14.0],
            "team_ORB": [29.0],
        }
    )

    result, stat_cols = calculate_advanced_stats(df)

    assert stat_cols == ["eFG", "to", "orb", "poss", "off_rating"]
    assert result.loc[0, "eFG"] == 0.55
    assert result.loc[0, "to"] == 14.0
    assert result.loc[0, "orb"] == 29.0
    assert result.loc[0, "poss"] == 70.0
    assert result.loc[0, "off_rating"] == 120.0
    assert "fga" not in result.columns
    assert "fgm" not in result.columns
    assert "TS" not in result.columns


def test_merge_opponent_stats_zero_fills_unavailable_advanced_deltas():
    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01"],
            "team": ["Home", "Away"],
            "opponent": ["Away", "Home"],
            "prev_win_pct": [0.8, 0.3],
        }
    )

    result = merge_opponent_stats(df)

    home_row = result[result["team"] == "Home"].iloc[0]
    assert home_row["opp_win_pct"] == 0.3
    assert home_row["diff_eFG"] == 0.0
    assert home_row["diff_Rebound"] == 0.0
    assert home_row["diff_TO"] == 0.0
    assert home_row["momentum_gap"] == 0.0


def test_merge_archived_market_spreads_fills_womens_zero_spreads(tmp_path):
    archive_file = tmp_path / "odds_history.csv"
    pd.DataFrame(
        [
            {
                "captured_at": "2026-03-06T12:00:00+00:00",
                "league": "womens",
                "date": "2026-03-06",
                "home_team": "Home",
                "away_team": "Away",
                "spread": -4.5,
                "total_line": pd.NA,
                "book": "Manual",
                "provider": "manual_override",
                "source": "prediction_run",
                "raw_line": "Manual -4.5",
                "has_market_spread": True,
            }
        ]
    ).to_csv(archive_file, index=False)

    df = pd.DataFrame(
        [
            {"date": "2026-03-06", "team": "Home", "opponent": "Away", "is_home": 1, "spread": 0.0},
            {"date": "2026-03-06", "team": "Away", "opponent": "Home", "is_home": 0, "spread": 0.0},
        ]
    )

    result = merge_archived_market_spreads(df, "womens", {"odds_archive_file": str(archive_file)})

    home = result[result["is_home"] == 1].iloc[0]
    away = result[result["is_home"] == 0].iloc[0]
    assert home["spread"] == -4.5
    assert away["spread"] == 4.5
