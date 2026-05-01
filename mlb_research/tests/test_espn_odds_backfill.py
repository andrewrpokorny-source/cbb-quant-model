import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mlb_research" / "anchor_v2"))

from backfill_espn_odds import (  # noqa: E402
    _extract_moneyline,
    _select_odds_item,
    enrich_training_rows,
)


def test_extract_moneyline_prefers_close_then_current_then_open():
    team_odds = {
        "moneyLine": 120,
        "open": {"moneyLine": {"american": "+110"}},
        "current": {"moneyLine": {"american": "+125"}},
        "close": {"moneyLine": {"american": "+130"}},
    }
    assert _extract_moneyline(team_odds, ["close", "current", "open"]) == (130.0, "close")


def test_extract_moneyline_falls_back_to_current():
    team_odds = {
        "open": {"moneyLine": {"american": "+110"}},
        "current": {"moneyLine": {"american": "+125"}},
    }
    assert _extract_moneyline(team_odds, ["close", "current", "open"]) == (125.0, "current")


def test_extract_moneyline_uses_top_level_only_when_requested():
    team_odds = {"moneyLine": -118}
    assert _extract_moneyline(team_odds, ["close"]) == (None, None)
    assert _extract_moneyline(team_odds, ["close", "top_level"]) == (-118.0, "top_level")


def test_select_odds_item_filters_provider_and_requires_both_sides():
    items = [
        {
            "provider": {"name": "Other Book"},
            "homeTeamOdds": {"close": {"moneyLine": {"american": "-150"}}},
            "awayTeamOdds": {"close": {"moneyLine": {"american": "+130"}}},
        },
        {
            "provider": {"name": "ESPN BET"},
            "homeTeamOdds": {"close": {"moneyLine": {"american": "-145"}}},
            "awayTeamOdds": {"close": {"moneyLine": {"american": "+125"}}},
        },
    ]
    item, home, away, home_basis, away_basis = _select_odds_item(
        items, "ESPN BET", ["close"]
    )
    assert item["provider"]["name"] == "ESPN BET"
    assert home == -145.0
    assert away == 125.0
    assert home_basis == "close"
    assert away_basis == "close"


def test_enrich_training_rows_fills_home_and_away_moneylines():
    source = pd.DataFrame(
        [
            {
                "date": "2025-04-15",
                "game_time": "23:05",
                "team_abbr": "NYY",
                "opp_abbr": "TOR",
                "moneyline": float("nan"),
                "run_line": float("nan"),
                "total_line": float("nan"),
            },
            {
                "date": "2025-04-15",
                "game_time": "23:05",
                "team_abbr": "TOR",
                "opp_abbr": "NYY",
                "moneyline": float("nan"),
                "run_line": float("nan"),
                "total_line": float("nan"),
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "date": "2025-04-15",
                "game_time": "23:05",
                "home_abbr": "TOR",
                "away_abbr": "NYY",
                "home_moneyline": 125.0,
                "away_moneyline": -145.0,
                "home_run_line": 1.5,
                "away_run_line": -1.5,
                "total_line": 7.5,
            }
        ]
    )

    enriched = enrich_training_rows(source, odds)
    away = enriched[enriched["team_abbr"] == "NYY"].iloc[0]
    home = enriched[enriched["team_abbr"] == "TOR"].iloc[0]

    assert away["moneyline"] == pytest.approx(-145.0)
    assert away["run_line"] == pytest.approx(-1.5)
    assert home["moneyline"] == pytest.approx(125.0)
    assert home["run_line"] == pytest.approx(1.5)
    assert home["total_line"] == pytest.approx(7.5)
