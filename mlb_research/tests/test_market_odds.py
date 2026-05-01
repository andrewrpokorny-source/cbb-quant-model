import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mlb_research.market_odds import (
    add_market_odds_features,
    american_odds_profit,
    american_odds_to_implied_prob,
    market_coverage_summary,
    no_vig_probability,
)


def test_american_odds_to_implied_prob():
    assert american_odds_to_implied_prob("-150") == pytest.approx(0.6)
    assert american_odds_to_implied_prob("+130") == pytest.approx(100 / 230)
    assert american_odds_to_implied_prob("EVEN") == pytest.approx(0.5)
    assert math.isnan(american_odds_to_implied_prob(""))


def test_american_odds_profit():
    assert american_odds_profit("-150") == pytest.approx(100 / 150)
    assert american_odds_profit("+130") == pytest.approx(1.3)
    assert american_odds_profit("EVEN") == pytest.approx(1.0)
    assert math.isnan(american_odds_profit("n/a"))


def test_no_vig_probability_normalizes_overround():
    favorite = american_odds_to_implied_prob("-150")
    underdog = american_odds_to_implied_prob("+130")
    assert favorite + underdog > 1.0
    assert no_vig_probability(favorite, underdog) == pytest.approx(
        favorite / (favorite + underdog)
    )


def test_add_market_odds_features_pairs_doubleheaders_by_time():
    df = pd.DataFrame(
        [
            {
                "date": "2025-04-15",
                "game_time": "13:05",
                "team": "A",
                "opponent": "B",
                "is_home": 1,
                "moneyline": -150,
            },
            {
                "date": "2025-04-15",
                "game_time": "13:05",
                "team": "B",
                "opponent": "A",
                "is_home": 0,
                "moneyline": 130,
            },
            {
                "date": "2025-04-15",
                "game_time": "19:05",
                "team": "A",
                "opponent": "B",
                "is_home": 1,
                "moneyline": 110,
            },
            {
                "date": "2025-04-15",
                "game_time": "19:05",
                "team": "B",
                "opponent": "A",
                "is_home": 0,
                "moneyline": -120,
            },
        ]
    )

    out = add_market_odds_features(df)
    early_home = out[(out["team"] == "A") & (out["game_time"] == "13:05")].iloc[0]
    late_home = out[(out["team"] == "A") & (out["game_time"] == "19:05")].iloc[0]

    assert early_home["opp_moneyline"] == pytest.approx(130)
    assert late_home["opp_moneyline"] == pytest.approx(-120)
    assert early_home["market_home_no_vig_prob"] > 0.5
    assert late_home["market_home_no_vig_prob"] < 0.5


def test_market_coverage_summary_counts_complete_pairs():
    df = pd.DataFrame(
        [
            {
                "date": "2025-04-15",
                "game_time": "13:05",
                "team": "A",
                "opponent": "B",
                "is_home": 1,
                "moneyline": -150,
            },
            {
                "date": "2025-04-15",
                "game_time": "13:05",
                "team": "B",
                "opponent": "A",
                "is_home": 0,
                "moneyline": 130,
            },
            {
                "date": "2025-04-16",
                "game_time": "13:05",
                "team": "C",
                "opponent": "D",
                "is_home": 1,
                "moneyline": float("nan"),
            },
            {
                "date": "2025-04-16",
                "game_time": "13:05",
                "team": "D",
                "opponent": "C",
                "is_home": 0,
                "moneyline": float("nan"),
            },
        ]
    )

    summary = market_coverage_summary(df)
    assert summary["rows"] == 4
    assert summary["complete_no_vig_rows"] == 2
    assert summary["complete_no_vig_share"] == pytest.approx(0.5)
    assert summary["home_complete_no_vig_rows"] == 1
