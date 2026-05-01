"""Market-odds helpers for MLB research anchors.

The production prediction path already reads same-day market prices, but the
frozen research anchor has historically carried empty moneyline columns. This
module is the shared, deterministic piece needed before a market-aware anchor
can be frozen: parse American odds, pair home/away rows, remove sportsbook vig,
and expose per-row market probabilities as normal tabular features.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


MARKET_FEATURE_COLUMNS = [
    "team_moneyline",
    "opp_moneyline",
    "market_team_implied_prob",
    "market_opp_implied_prob",
    "market_overround",
    "market_team_no_vig_prob",
    "market_home_no_vig_prob",
]


def american_odds_to_implied_prob(value: Any) -> float:
    """Convert American odds to raw implied probability.

    Returns NaN for empty, zero, or unparseable values so callers can use normal
    pandas missing-value semantics.
    """
    if value is None:
        return float("nan")
    if isinstance(value, str):
        cleaned = value.strip().upper()
        if not cleaned:
            return float("nan")
        if cleaned == "EVEN":
            cleaned = "100"
        cleaned = cleaned.replace("+", "")
    else:
        cleaned = value
    try:
        odds = float(cleaned)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(odds) or odds == 0:
        return float("nan")
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def american_odds_profit(value: Any, stake: float = 1.0) -> float:
    """Return profit on a winning American-odds bet for a given stake."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        cleaned = value.strip().upper()
        if not cleaned:
            return float("nan")
        if cleaned == "EVEN":
            cleaned = "100"
        cleaned = cleaned.replace("+", "")
    else:
        cleaned = value
    try:
        odds = float(cleaned)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(odds) or odds == 0:
        return float("nan")
    if odds < 0:
        return stake * 100.0 / abs(odds)
    return stake * odds / 100.0


def no_vig_probability(team_implied: Any, opponent_implied: Any) -> float:
    """Normalize two raw implied probabilities to remove overround."""
    try:
        team = float(team_implied)
        opponent = float(opponent_implied)
    except (TypeError, ValueError):
        return float("nan")
    total = team + opponent
    if not math.isfinite(team) or not math.isfinite(opponent) or total <= 0:
        return float("nan")
    return team / total


def _game_keys(df: pd.DataFrame) -> pd.Series:
    teams = df["team"].astype(str)
    opponents = df["opponent"].astype(str)
    low = np.where(teams <= opponents, teams, opponents)
    high = np.where(teams <= opponents, opponents, teams)
    game_time = df["game_time"].fillna("").astype(str) if "game_time" in df else ""
    return (
        df["date"].astype(str)
        + "|"
        + pd.Series(game_time, index=df.index).astype(str)
        + "|"
        + pd.Series(low, index=df.index).astype(str)
        + "|"
        + pd.Series(high, index=df.index).astype(str)
    )


def add_market_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add paired moneyline and no-vig market probability columns.

    Input rows must follow the MLB pipeline convention: two rows per game, one
    per team, with the row's own American moneyline in ``moneyline``. Games are
    paired by date, game_time, and unordered team/opponent names, which keeps
    doubleheaders separated when game_time is present.
    """
    required = {"date", "team", "opponent", "is_home", "moneyline"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required market-odds columns: {missing}")

    out = df.copy()
    out["team_moneyline"] = pd.to_numeric(out["moneyline"], errors="coerce")
    out["opp_moneyline"] = np.nan
    out["_market_game_key"] = _game_keys(out)

    for _, idx in out.groupby("_market_game_key").groups.items():
        idx = list(idx)
        if len(idx) != 2:
            continue
        left, right = idx
        out.at[left, "opp_moneyline"] = out.at[right, "team_moneyline"]
        out.at[right, "opp_moneyline"] = out.at[left, "team_moneyline"]

    out["market_team_implied_prob"] = out["team_moneyline"].map(american_odds_to_implied_prob)
    out["market_opp_implied_prob"] = out["opp_moneyline"].map(american_odds_to_implied_prob)
    out["market_overround"] = (
        out["market_team_implied_prob"] + out["market_opp_implied_prob"] - 1.0
    )
    out["market_team_no_vig_prob"] = [
        no_vig_probability(team, opp)
        for team, opp in zip(out["market_team_implied_prob"], out["market_opp_implied_prob"])
    ]
    out["market_home_no_vig_prob"] = np.where(
        out["is_home"].astype(int) == 1,
        out["market_team_no_vig_prob"],
        1.0 - out["market_team_no_vig_prob"],
    )
    out.loc[out["market_team_no_vig_prob"].isna(), "market_home_no_vig_prob"] = np.nan
    return out.drop(columns=["_market_game_key"])


def market_coverage_summary(df: pd.DataFrame) -> dict:
    """Return row-level coverage counts for market-derived columns."""
    enriched = add_market_odds_features(df)
    rows = int(len(enriched))
    complete_market = enriched["market_team_no_vig_prob"].notna()
    home_complete = complete_market & (enriched["is_home"].astype(int) == 1)
    return {
        "rows": rows,
        "team_moneyline_rows": int(enriched["team_moneyline"].notna().sum()),
        "paired_moneyline_rows": int(enriched["opp_moneyline"].notna().sum()),
        "complete_no_vig_rows": int(complete_market.sum()),
        "complete_no_vig_share": float(complete_market.mean()) if rows else 0.0,
        "home_rows": int((enriched["is_home"].astype(int) == 1).sum()),
        "home_complete_no_vig_rows": int(home_complete.sum()),
        "home_complete_no_vig_share": (
            float(home_complete.sum() / max(1, (enriched["is_home"].astype(int) == 1).sum()))
        ),
    }
