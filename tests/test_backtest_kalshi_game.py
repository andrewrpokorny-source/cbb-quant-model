"""Tests for the Kalshi GAME historical backtest helpers."""

from datetime import datetime

import pandas as pd
import pytest

from backtest_kalshi_game import (
    calculate_kalshi_contract_outcome,
    compare_actual_bets_to_archived_predictions,
    load_actual_betting_history_rows,
    load_actual_betting_history_results,
    result_from_market_result,
    resolve_backtest_results,
    select_latest_snapshot_per_game,
    summarize_backtest,
)


def _sample_snapshots():
    return pd.DataFrame(
        [
            {
                "captured_at": pd.Timestamp("2026-03-08T13:00:00Z"),
                "league": "mens",
                "game_date": "2026-03-08",
                "game_datetime": pd.Timestamp("2026-03-08T19:00:00Z"),
                "matchup": "Providence Friars @ UConn Huskies",
                "home_team": "UConn Huskies",
                "away_team": "Providence Friars",
                "pick": "UConn Huskies ML YES",
                "picked_team": "UConn Huskies",
                "kalshi_side": "YES",
                "kalshi_ticker": "KXNCAAMBGAME-EXAMPLE",
                "kalshi_price": 58.0,
                "kalshi_fee": 1.7,
                "edge": 0.023,
                "edge_pct": 2.3,
                "rating": "MARGINAL",
                "conf": 0.61,
            },
            {
                "captured_at": pd.Timestamp("2026-03-08T15:00:00Z"),
                "league": "mens",
                "game_date": "2026-03-08",
                "game_datetime": pd.Timestamp("2026-03-08T19:00:00Z"),
                "matchup": "Providence Friars @ UConn Huskies",
                "home_team": "UConn Huskies",
                "away_team": "Providence Friars",
                "pick": "UConn Huskies ML NO",
                "picked_team": "Providence Friars",
                "kalshi_side": "NO",
                "kalshi_ticker": "KXNCAAMBGAME-EXAMPLE",
                "kalshi_price": 36.0,
                "kalshi_fee": 1.6,
                "edge": 0.055,
                "edge_pct": 5.5,
                "rating": "GOOD",
                "conf": 0.60,
            },
        ]
    )


def test_select_latest_snapshot_per_game_keeps_latest():
    latest = select_latest_snapshot_per_game(_sample_snapshots())
    assert len(latest) == 1
    assert latest.iloc[0]["kalshi_side"] == "NO"


def test_calculate_kalshi_contract_outcome_handles_win_and_loss():
    stake, payout, profit = calculate_kalshi_contract_outcome(40, "win")
    assert stake == pytest.approx(0.4168, rel=1e-4)
    assert payout == 1.0
    assert profit == pytest.approx(0.5832, rel=1e-4)

    stake, payout, profit = calculate_kalshi_contract_outcome(40, "loss")
    assert stake == pytest.approx(0.4168, rel=1e-4)
    assert payout == 0.0
    assert profit == pytest.approx(-0.4168, rel=1e-4)


def test_result_from_market_result_maps_yes_no():
    assert result_from_market_result("YES", "yes") == "win"
    assert result_from_market_result("NO", "yes") == "loss"
    assert result_from_market_result("YES", "void") is None


def test_resolve_backtest_results_grades_game_pick():
    snapshots = pd.DataFrame(
        [
            {
                "captured_at": pd.Timestamp("2026-03-08T13:00:00Z"),
                "league": "mens",
                "game_date": "2026-03-08",
                "game_datetime": pd.Timestamp("2026-03-08T19:00:00Z"),
                "matchup": "Providence Friars @ UConn Huskies",
                "home_team": "UConn Huskies",
                "away_team": "Providence Friars",
                "pick": "UConn Huskies ML NO",
                "picked_team": "Providence Friars",
                "kalshi_side": "NO",
                "kalshi_ticker": "KXNCAAMBGAME-EXAMPLE",
                "kalshi_price": 36.0,
                "kalshi_fee": 1.6,
                "edge": 0.055,
                "edge_pct": 5.5,
                "rating": "GOOD",
                "conf": 0.60,
            }
        ]
    )

    def _fetcher(date_obj: datetime, league: str = "mens"):
        assert league == "mens"
        assert date_obj.date().isoformat() == "2026-03-08"
        return {
            ("UConn Huskies", "Providence Friars"): {
                "home_score": 80,
                "away_score": 70,
                "home_name": "UConn Huskies",
                "away_name": "Providence Friars",
            }
        }

    results = resolve_backtest_results(snapshots, league="mens", score_fetcher=_fetcher)
    assert len(results) == 1
    row = results.iloc[0]
    assert row["result"] == "loss"
    assert row["price_bucket"] == "25-39"
    assert row["edge_bucket"] == "GOOD"
    assert row["profit"] < 0


def test_resolve_backtest_results_prefers_kalshi_market_result():
    snapshots = pd.DataFrame(
        [
            {
                "captured_at": pd.Timestamp("2026-03-08T13:00:00Z"),
                "league": "mens",
                "game_date": "2026-03-08",
                "game_datetime": pd.Timestamp("2026-03-08T19:00:00Z"),
                "matchup": "Providence Friars @ UConn Huskies",
                "home_team": "UConn Huskies",
                "away_team": "Providence Friars",
                "pick": "UConn Huskies ML NO",
                "picked_team": "Providence Friars",
                "kalshi_side": "NO",
                "kalshi_ticker": "KXNCAAMBGAME-EXAMPLE",
                "kalshi_price": 36.0,
                "kalshi_fee": 1.6,
                "edge": 0.055,
                "edge_pct": 5.5,
                "rating": "GOOD",
                "conf": 0.60,
            }
        ]
    )

    def _fetcher(*args, **kwargs):
        raise AssertionError("ESPN fallback should not be used when Kalshi market_result resolves")

    def _resolver(ticker: str):
        assert ticker == "KXNCAAMBGAME-EXAMPLE"
        return {"market_result": "no"}

    results = resolve_backtest_results(
        snapshots,
        league="mens",
        score_fetcher=_fetcher,
        market_resolver=_resolver,
    )
    assert len(results) == 1
    assert results.iloc[0]["result"] == "win"


def test_summarize_backtest_builds_group_tables():
    results = pd.DataFrame(
        [
            {
                "league": "mens",
                "result": "win",
                "profit": 0.5,
                "stake": 0.4,
                "edge_pct": 5.0,
                "kalshi_price": 40.0,
                "conf": 0.6,
                "edge_bucket": "GOOD",
                "price_bucket": "40-59",
                "rating": "GOOD",
            },
            {
                "league": "mens",
                "result": "loss",
                "profit": -0.3,
                "stake": 0.3,
                "edge_pct": 2.5,
                "kalshi_price": 30.0,
                "conf": 0.55,
                "edge_bucket": "MARGINAL",
                "price_bucket": "25-39",
                "rating": "MARGINAL",
            },
        ]
    )

    summary = summarize_backtest(results)
    assert set(summary) == {"overall", "by_edge", "by_price", "by_rating"}
    assert int(summary["overall"].iloc[0]["bets"]) == 2
    assert "GOOD" in summary["by_edge"]["edge_bucket"].tolist()


def test_load_actual_betting_history_results(tmp_path):
    csv_path = tmp_path / "betting_history.csv"
    csv_path.write_text(
        "date,platform,game,bet_type,line,odds,wager,result,payout,profit,bet_id,league\n"
        "2026-03-02,Kalshi,Oklahoma at Missouri Winner?,game,Missouri ML,,0.11,loss,0.0,-0.11,KXNCAAWBGAME-26MAR01OKLAMIZZ-MIZZ,womens\n"
        "2026-03-04,Kalshi,North Texas at Wichita St. Winner?,game,Wichita St. ML,,1.20,win,8.0,6.80,KXNCAAWBGAME-26MAR03UNTWICH-WICH,womens\n"
    )

    results = load_actual_betting_history_results(str(csv_path), league="womens")
    assert len(results) == 2
    assert results.iloc[0]["rating"] == "actual_bet"
    assert results.iloc[0]["price_bucket"] == "unknown"
    assert results.iloc[1]["profit"] == pytest.approx(6.8)


def test_load_actual_betting_history_rows_filters_kalshi_game(tmp_path):
    csv_path = tmp_path / "betting_history.csv"
    csv_path.write_text(
        "date,platform,game,bet_type,line,odds,wager,result,payout,profit,bet_id,league\n"
        "2026-03-02,Kalshi,Game A,game,Team A ML,,0.11,loss,0.0,-0.11,TICKER1,mens\n"
        "2026-03-02,FanDuel,Game B,spread,Team B -3.5,-110,1.10,win,2.10,1.00,BET2,mens\n"
    )
    rows = load_actual_betting_history_rows(str(csv_path), league="mens")
    assert len(rows) == 1
    assert rows.iloc[0]["bet_id"] == "TICKER1"


def test_compare_actual_bets_to_archived_predictions_matches_on_ticker(tmp_path):
    csv_path = tmp_path / "betting_history.csv"
    csv_path.write_text(
        "date,platform,game,bet_type,line,odds,wager,result,payout,profit,bet_id,league\n"
        "2026-03-10,Kalshi,Providence Friars @ UConn Huskies,game,UConn Huskies ML YES,,0.58,win,1.0,0.42,TICKER1,mens\n"
    )
    archived = pd.DataFrame(
        [
            {
                "captured_at": pd.Timestamp("2026-03-09T15:00:00Z"),
                "league": "mens",
                "game_date": "2026-03-10",
                "game_datetime": pd.Timestamp("2026-03-10T20:00:00Z"),
                "matchup": "Providence Friars @ UConn Huskies",
                "home_team": "UConn Huskies",
                "away_team": "Providence Friars",
                "pick": "UConn Huskies ML YES",
                "picked_team": "UConn Huskies",
                "kalshi_side": "YES",
                "kalshi_ticker": "TICKER1",
                "kalshi_price": 58.0,
                "kalshi_fee": 1.7,
                "edge": 0.08,
                "edge_pct": 8.0,
                "rating": "STRONG",
                "conf": 0.67,
            }
        ]
    )

    comparisons = compare_actual_bets_to_archived_predictions(
        archived,
        str(csv_path),
        league="mens",
    )
    assert len(comparisons) == 1
    row = comparisons.iloc[0]
    assert row["rating"] == "STRONG"
    assert row["result"] == "win"
    assert row["edge_bucket"] == "STRONG"
