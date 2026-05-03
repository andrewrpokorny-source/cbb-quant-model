"""Tests for the MLB shadow grader."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from mlb import shadow_grader


PREDICTION_BASE_COLS = [
    "Bet_Type",
    "Date/Time",
    "Matchup",
    "Pick",
    "Prob_Home",
    "Prob_Away",
    "Conf",
    "Std_Odds",
]

SHADOW_COLS = list(shadow_grader.REQUIRED_SHADOW_COLS) + [
    "MarketV2_Prob_Away",
    "MarketV2_Edge_vs_Market",
]


def _prediction_row(
    *,
    matchup="Boston Red Sox @ New York Yankees",
    pick="New York Yankees",
    prob_home=0.62,
    std_odds="-150",
    shadow_status="ok",
    shadow_prob_home=0.62,
    shadow_pick="New York Yankees",
    shadow_market_home=0.55,
    agrees=True,
    date="2026-04-15 19:05",
):
    return {
        "Bet_Type": "game",
        "Date/Time": date,
        "Matchup": matchup,
        "Pick": pick,
        "Prob_Home": prob_home,
        "Prob_Away": 1.0 - prob_home,
        "Conf": max(prob_home, 1.0 - prob_home),
        "Std_Odds": std_odds,
        "MarketV2_Status": shadow_status,
        "MarketV2_Prob_Home": shadow_prob_home,
        "MarketV2_Prob_Away": 1.0 - shadow_prob_home,
        "MarketV2_Pick": shadow_pick,
        "MarketV2_Conf": max(shadow_prob_home, 1.0 - shadow_prob_home),
        "MarketV2_Market_NoVig_Home": shadow_market_home,
        "MarketV2_Edge_vs_Market": 0.05,
        "MarketV2_Agrees_With_Production": agrees,
    }


def _write_archive(tmp_path, date_str, rows):
    df = pd.DataFrame(rows)
    path = os.path.join(tmp_path, f"predictions_mlb_{date_str.replace('-', '')}.csv")
    df.to_csv(path, index=False)
    return path


def _write_outcomes(tmp_path, games):
    """Each game: dict(date, home, away, home_won, game_time='19:05')."""
    rows = []
    for g in games:
        rows.append(
            {
                "date": g["date"],
                "game_time": g.get("game_time", "19:05"),
                "team": g["home"],
                "opponent": g["away"],
                "is_home": 1,
                "team_score": 5 if g["home_won"] else 2,
                "opp_score": 2 if g["home_won"] else 5,
                "home_win": int(g["home_won"]),
            }
        )
        rows.append(
            {
                "date": g["date"],
                "game_time": g.get("game_time", "19:05"),
                "team": g["away"],
                "opponent": g["home"],
                "is_home": 0,
                "team_score": 2 if g["home_won"] else 5,
                "opp_score": 5 if g["home_won"] else 2,
                "home_win": int(g["home_won"]),
            }
        )
    path = os.path.join(tmp_path, "outcomes.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_skips_archive_without_shadow_columns(tmp_path):
    legacy = pd.DataFrame(
        [{
            "Bet_Type": "game",
            "Date/Time": "2026-04-12 19:05",
            "Matchup": "Boston Red Sox @ New York Yankees",
            "Pick": "New York Yankees",
            "Prob_Home": 0.55,
            "Prob_Away": 0.45,
            "Conf": 0.55,
            "Std_Odds": "-130",
        }]
    )
    legacy_path = os.path.join(tmp_path, "predictions_mlb_20260412.csv")
    legacy.to_csv(legacy_path, index=False)

    outcomes_path = _write_outcomes(
        tmp_path,
        [{"date": "2026-04-12", "home": "New York Yankees",
          "away": "Boston Red Sox", "home_won": True}],
    )

    ledger = shadow_grader.grade(str(tmp_path), outcomes_path)
    assert ledger.empty


def test_skips_rows_with_non_ok_status(tmp_path):
    rows = [
        _prediction_row(shadow_status="odds_missing"),
        _prediction_row(
            matchup="Houston Astros @ Boston Red Sox",
            pick="Boston Red Sox",
            shadow_status="model_missing",
            shadow_pick="",
        ),
    ]
    archive = _write_archive(tmp_path, "2026-04-15", rows)
    outcomes = _write_outcomes(
        tmp_path,
        [
            {"date": "2026-04-15", "home": "New York Yankees",
             "away": "Boston Red Sox", "home_won": True},
            {"date": "2026-04-15", "home": "Boston Red Sox",
             "away": "Houston Astros", "home_won": False},
        ],
    )

    ledger = shadow_grader.grade(os.path.dirname(archive), outcomes)
    assert ledger.empty


def test_grades_shadow_disagrees_and_wins(tmp_path):
    """The cell that matters for promotion: shadow disagrees with production
    and shadow's pick wins."""
    rows = [
        _prediction_row(
            matchup="New York Mets @ Los Angeles Angels",
            pick="New York Mets",            # production picks away
            prob_home=0.45,
            shadow_pick="Los Angeles Angels",  # shadow picks home
            shadow_prob_home=0.55,
            shadow_market_home=0.51,
            agrees=False,
            std_odds="+120",
        ),
    ]
    archive = _write_archive(tmp_path, "2026-04-15", rows)
    outcomes = _write_outcomes(
        tmp_path,
        [{"date": "2026-04-15", "home": "Los Angeles Angels",
          "away": "New York Mets", "home_won": True}],
    )

    ledger = shadow_grader.grade(os.path.dirname(archive), outcomes)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["outcome_status"] == "graded"
    assert not row["production_correct"]
    assert row["shadow_correct"]
    assert row["market_correct"]
    assert not row["agrees_with_production"]
    # Std_Odds is for production side (away). Shadow disagrees, so its ROI
    # is intentionally None and roi_data_missing is True.
    assert pd.isna(row["shadow_roi_units"])
    assert row["roi_data_missing"]
    # Production lost at +120 -> -1U.
    assert row["production_roi_units"] == pytest.approx(-1.0)


def test_grades_agree_case_with_brier_targets(tmp_path):
    rows = [
        _prediction_row(
            matchup="Boston Red Sox @ New York Yankees",
            pick="New York Yankees",
            prob_home=0.62,
            shadow_pick="New York Yankees",
            shadow_prob_home=0.58,
            shadow_market_home=0.55,
            agrees=True,
            std_odds="-140",
        ),
    ]
    archive = _write_archive(tmp_path, "2026-04-15", rows)
    outcomes = _write_outcomes(
        tmp_path,
        [{"date": "2026-04-15", "home": "New York Yankees",
          "away": "Boston Red Sox", "home_won": True}],
    )

    ledger = shadow_grader.grade(os.path.dirname(archive), outcomes)
    row = ledger.iloc[0]

    assert row["production_brier"] == pytest.approx((0.62 - 1) ** 2)
    assert row["shadow_brier"] == pytest.approx((0.58 - 1) ** 2)
    assert row["market_brier"] == pytest.approx((0.55 - 1) ** 2)
    assert row["production_correct"]
    assert row["shadow_correct"]
    assert row["market_correct"]
    # -140 -> +0.7143U on win.
    assert row["production_roi_units"] == pytest.approx(100.0 / 140.0)
    assert row["shadow_roi_units"] == pytest.approx(100.0 / 140.0)
    assert row["market_roi_units"] == pytest.approx(100.0 / 140.0)
    assert not row["roi_data_missing"]


def test_idempotent_through_csv_roundtrip(tmp_path):
    rows = [_prediction_row()]
    archive = _write_archive(tmp_path, "2026-04-15", rows)
    outcomes = _write_outcomes(
        tmp_path,
        [{"date": "2026-04-15", "home": "New York Yankees",
          "away": "Boston Red Sox", "home_won": True}],
    )

    ledger_path = os.path.join(tmp_path, "ledger.csv")
    first = shadow_grader.grade(os.path.dirname(archive), outcomes)
    shadow_grader.write_ledger(first, ledger_path)
    persisted = pd.read_csv(ledger_path)

    second = shadow_grader.grade(os.path.dirname(archive), outcomes)
    shadow_grader.write_ledger(second, ledger_path)
    persisted_again = pd.read_csv(ledger_path)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(persisted, persisted_again)


def test_doubleheader_disambiguates_by_game_time(tmp_path):
    """Two games same date / same teams should each pick up their own outcome."""
    rows = [
        _prediction_row(
            matchup="Boston Red Sox @ New York Yankees",
            pick="New York Yankees",
            shadow_pick="New York Yankees",
            shadow_prob_home=0.62,
            shadow_market_home=0.55,
            agrees=True,
            std_odds="-130",
            date="2026-04-15 17:05",
        ),
        _prediction_row(
            matchup="Boston Red Sox @ New York Yankees",
            pick="New York Yankees",
            shadow_pick="New York Yankees",
            shadow_prob_home=0.55,
            shadow_market_home=0.50,
            agrees=True,
            std_odds="-115",
            date="2026-04-15 20:35",
        ),
    ]
    archive = _write_archive(tmp_path, "2026-04-15", rows)
    outcomes = _write_outcomes(
        tmp_path,
        [
            {"date": "2026-04-15", "home": "New York Yankees",
             "away": "Boston Red Sox", "home_won": True,
             "game_time": "17:05"},
            {"date": "2026-04-15", "home": "New York Yankees",
             "away": "Boston Red Sox", "home_won": False,
             "game_time": "20:35"},
        ],
    )

    ledger = shadow_grader.grade(os.path.dirname(archive), outcomes)
    assert len(ledger) == 2

    by_time = ledger.set_index("game_time")
    assert by_time.loc["17:05"]["home_won"] == 1
    assert by_time.loc["17:05"]["production_correct"]
    assert by_time.loc["20:35"]["home_won"] == 0
    assert not by_time.loc["20:35"]["production_correct"]


def test_corrupted_archive_does_not_abort_run(tmp_path):
    """A zero-byte archive (e.g. interrupted predict run) should be skipped."""
    rows = [_prediction_row()]
    good = _write_archive(tmp_path, "2026-04-15", rows)
    bad = os.path.join(tmp_path, "predictions_mlb_20260416.csv")
    open(bad, "w").close()  # zero bytes

    outcomes = _write_outcomes(
        tmp_path,
        [{"date": "2026-04-15", "home": "New York Yankees",
          "away": "Boston Red Sox", "home_won": True}],
    )

    ledger = shadow_grader.grade(os.path.dirname(good), outcomes)
    assert len(ledger) == 1
    assert ledger.iloc[0]["outcome_status"] == "graded"


def test_shadow_disagrees_and_loses(tmp_path):
    """Counterpart to the disagrees-and-wins case: penalize wrong shadow flip."""
    rows = [
        _prediction_row(
            matchup="New York Mets @ Los Angeles Angels",
            pick="New York Mets",
            prob_home=0.45,
            shadow_pick="Los Angeles Angels",
            shadow_prob_home=0.55,
            shadow_market_home=0.51,
            agrees=False,
            std_odds="+120",
        ),
    ]
    archive = _write_archive(tmp_path, "2026-04-15", rows)
    outcomes = _write_outcomes(
        tmp_path,
        # Production was right this time -- away (Mets) won.
        [{"date": "2026-04-15", "home": "Los Angeles Angels",
          "away": "New York Mets", "home_won": False}],
    )

    ledger = shadow_grader.grade(os.path.dirname(archive), outcomes)
    row = ledger.iloc[0]
    assert row["production_correct"]
    assert not row["shadow_correct"]
    assert not row["market_correct"]
    # Production won at +120 -> +1.2U.
    assert row["production_roi_units"] == pytest.approx(120 / 100)
    # Shadow disagrees with production: ROI is intentionally None.
    assert pd.isna(row["shadow_roi_units"])
    assert row["roi_data_missing"]


def test_outcome_pending_when_game_not_in_training_data(tmp_path):
    rows = [_prediction_row(date="2026-05-02 19:05")]
    archive = _write_archive(tmp_path, "2026-05-02", rows)
    outcomes = _write_outcomes(
        tmp_path,
        # No game on 2026-05-02 in training data -> outcome pending.
        [{"date": "2026-04-15", "home": "New York Yankees",
          "away": "Boston Red Sox", "home_won": True}],
    )

    ledger = shadow_grader.grade(os.path.dirname(archive), outcomes)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["outcome_status"] == "outcome_pending"
    assert pd.isna(row["home_won"])
    assert pd.isna(row["production_brier"])
    assert pd.isna(row["shadow_brier"])
    assert pd.isna(row["market_brier"])
    assert row["roi_data_missing"]


def test_team_name_normalization_via_find_best_match(tmp_path, monkeypatch):
    """ESPN-style 'Athletics' should normalize to training data 'Oakland Athletics'."""
    monkeypatch.setattr(shadow_grader, "find_best_match",
                        lambda name, known: "Oakland Athletics" if name == "Athletics" else (
                            name if name in known else None))

    rows = [
        _prediction_row(
            matchup="Cleveland Guardians @ Athletics",
            pick="Athletics",
            shadow_pick="Athletics",
            shadow_prob_home=0.58,
            shadow_market_home=0.55,
            prob_home=0.6,
        ),
    ]
    archive = _write_archive(tmp_path, "2026-04-15", rows)
    outcomes = _write_outcomes(
        tmp_path,
        [{"date": "2026-04-15", "home": "Oakland Athletics",
          "away": "Cleveland Guardians", "home_won": True}],
    )

    ledger = shadow_grader.grade(os.path.dirname(archive), outcomes)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["home_team"] == "Oakland Athletics"
    assert row["outcome_status"] == "graded"
    assert row["production_correct"]


def test_aggregate_report_contains_paired_brier_deltas(tmp_path):
    rows = [
        _prediction_row(
            matchup="Boston Red Sox @ New York Yankees",
            pick="New York Yankees",
            prob_home=0.62,
            shadow_prob_home=0.58,
            shadow_pick="New York Yankees",
            shadow_market_home=0.55,
            agrees=True,
            std_odds="-140",
        ),
        _prediction_row(
            matchup="Cleveland Guardians @ Detroit Tigers",
            pick="Cleveland Guardians",
            prob_home=0.45,
            shadow_pick="Detroit Tigers",
            shadow_prob_home=0.52,
            shadow_market_home=0.51,
            agrees=False,
            std_odds="+115",
            date="2026-04-15 13:10",
        ),
    ]
    archive = _write_archive(tmp_path, "2026-04-15", rows)
    outcomes = _write_outcomes(
        tmp_path,
        [
            {"date": "2026-04-15", "home": "New York Yankees",
             "away": "Boston Red Sox", "home_won": True},
            {"date": "2026-04-15", "home": "Detroit Tigers",
             "away": "Cleveland Guardians", "home_won": True},
        ],
    )

    ledger = shadow_grader.grade(os.path.dirname(archive), outcomes)
    report = shadow_grader.aggregate_report(ledger)

    assert report["n_graded"] == 2
    assert report["agreement_rate"] == pytest.approx(0.5)
    assert "shadow_minus_market_brier" in report
    assert "production_minus_market_brier" in report
    assert "shadow_minus_production_brier" in report
    formatted = shadow_grader.format_report(report)
    assert "Brier deltas" in formatted
    assert "Agreement" in formatted
