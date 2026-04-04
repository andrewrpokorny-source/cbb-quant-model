"""Edge calculation and bet rating logic."""

import math
from enum import Enum


class EdgeRating(Enum):
    """Bet recommendation tiers based on edge size."""

    STRONG = "STRONG"      # >=8% edge
    GOOD = "GOOD"          # >=4% and <8% edge
    MARGINAL = "MARGINAL"  # >=2% and <4% edge
    PASS = "PASS"          # <2% edge


# Rating thresholds (edge percentage)
STRONG_THRESHOLD = 0.08    # 8%
GOOD_THRESHOLD = 0.04      # 4%
MARGINAL_THRESHOLD = 0.02  # 2%

# Ratings considered actionable value bets
VALUE_RATINGS = ("STRONG", "GOOD")

# Rank ordering for sorting/display (higher = better)
RATING_RANK = {"STRONG": 3, "GOOD": 2, "MARGINAL": 1, "PASS": 0}

# Kalshi taker fee: 0.07 * P * (1-P) per contract, max ~1.75c at P=50c.
# https://kalshi.com/docs/kalshi-fee-schedule.pdf
KALSHI_TAKER_FEE_COEFF = 0.07


def kalshi_fee_cents(price_cents: float) -> float:
    """Kalshi taker fee in cents for a single contract.

    Fee = 0.07 * P * (1-P) * 100, where P = price_cents / 100.

    Kalshi rounds up per-trade (ceil to nearest cent), but we return
    the raw per-contract amount since rounding applies to the total
    trade, not individual contracts.
    """
    p = price_cents / 100.0
    return KALSHI_TAKER_FEE_COEFF * p * (1.0 - p) * 100.0


def kalshi_implied_prob(price_cents: float) -> float:
    """Convert Kalshi ask price to fee-adjusted implied probability.

    The true cost of a contract is ask + taker fee, so the breakeven
    probability is (ask + fee) / 100.

    Args:
        price_cents: Kalshi ask price on 0-100 scale (e.g., 40 = 40 cents)

    Returns:
        Fee-adjusted implied probability (0-1 scale)
    """
    p = price_cents / 100.0
    fee = KALSHI_TAKER_FEE_COEFF * p * (1.0 - p)
    return p + fee


def calculate_edge(model_prob: float, market_implied_prob: float) -> float:
    """
    Calculate betting edge.

    Edge = Model probability - Market implied probability

    For Kalshi markets, pass fee-adjusted implied probability from
    kalshi_implied_prob() so fees are properly deducted from edge.

    Args:
        model_prob: Model's predicted probability (0-1)
        market_implied_prob: Market's implied probability (0-1),
            fee-adjusted for Kalshi via kalshi_implied_prob()

    Returns:
        Edge as decimal (e.g., 0.06 = 6% edge)
    """
    return model_prob - market_implied_prob


def get_rating(edge: float) -> EdgeRating:
    """
    Get recommendation tier based on edge.

    Args:
        edge: Edge as decimal (e.g., 0.06 = 6%)

    Returns:
        EdgeRating enum value
    """
    if edge >= STRONG_THRESHOLD:
        return EdgeRating.STRONG
    if edge >= GOOD_THRESHOLD:
        return EdgeRating.GOOD
    if edge >= MARGINAL_THRESHOLD:
        return EdgeRating.MARGINAL
    return EdgeRating.PASS


def american_odds_to_implied_prob(odds_str):
    """Convert American odds string to implied probability (no-vig single side).

    Args:
        odds_str: American odds like "-150", "+130", "-110", or "+100".

    Returns:
        Implied probability as a float (0-1), or None if unparseable.

    Examples:
        "-110" -> 110/210 = 0.5238
        "-150" -> 150/250 = 0.6000
        "+130" -> 100/230 = 0.4348
    """
    try:
        odds = float(str(odds_str).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if odds == 0 or math.isnan(odds):
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def calculate_ev(
    model_prob: float,
    payout_if_win: float = 1.0,
    cost_if_loss: float = 1.1,
) -> float:
    """
    Calculate expected value of a bet.

    EV = (prob_win * payout) - (prob_loss * cost)

    Args:
        model_prob: Probability of winning (0-1)
        payout_if_win: Amount won on a win (default 1.0 = even money)
        cost_if_loss: Amount lost on a loss (default 1.1 = -110 juice)

    Returns:
        Expected value per unit wagered
    """
    prob_loss = 1 - model_prob
    return (model_prob * payout_if_win) - (prob_loss * cost_if_loss)


def analyze_bet(model_prob: float, kalshi_yes_price: float) -> dict:
    """
    Complete bet analysis with Kalshi taker fees included.

    Args:
        model_prob: Model's predicted probability (0-1)
        kalshi_yes_price: Kalshi Yes price (0-100 scale)

    Returns:
        Dict with edge, edge_pct, rating, implied_prob, model_prob, ev
    """
    implied_prob = kalshi_implied_prob(kalshi_yes_price)
    edge = calculate_edge(model_prob, implied_prob)
    rating = get_rating(edge)

    # Effective cost includes taker fee
    effective_cost = kalshi_yes_price + kalshi_fee_cents(kalshi_yes_price)
    payout = (100 - effective_cost) / effective_cost if effective_cost > 0 else 0
    ev = calculate_ev(model_prob, payout_if_win=payout, cost_if_loss=1.0)

    return {
        "edge": edge,
        "edge_pct": edge * 100,
        "rating": rating,
        "implied_prob": implied_prob,
        "model_prob": model_prob,
        "ev": ev,
    }
