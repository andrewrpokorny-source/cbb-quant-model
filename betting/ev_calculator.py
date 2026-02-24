"""Edge calculation and bet rating logic."""

from enum import Enum


class EdgeRating(Enum):
    """Bet recommendation tiers based on edge size."""

    STRONG = "STRONG"    # 8%+ edge
    GOOD = "GOOD"        # 4-8% edge
    MARGINAL = "MARGINAL"  # 2-4% edge
    PASS = "PASS"        # <2% edge


# Rating thresholds (edge percentage)
STRONG_THRESHOLD = 0.08    # 8%
GOOD_THRESHOLD = 0.04      # 4%
MARGINAL_THRESHOLD = 0.02  # 2%

# Ratings considered actionable value bets
VALUE_RATINGS = ("STRONG", "GOOD")

# Rank ordering for sorting/display (higher = better)
RATING_RANK = {"STRONG": 3, "GOOD": 2, "MARGINAL": 1, "PASS": 0}


def calculate_edge(model_prob: float, market_implied_prob: float) -> float:
    """
    Calculate betting edge.

    Edge = Model probability - Market implied probability

    Kalshi fees are already captured in the bid-ask spread,
    so no separate fee adjustment is needed.

    Args:
        model_prob: Model's predicted probability (0-1)
        market_implied_prob: Market's implied probability (0-1)

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
    Complete bet analysis.

    Kalshi fees are already captured in the bid-ask spread,
    so no separate fee adjustment is needed.

    Args:
        model_prob: Model's predicted probability (0-1)
        kalshi_yes_price: Kalshi Yes price (0-100 scale)

    Returns:
        Dict with edge, rating, implied_prob, ev
    """
    implied_prob = kalshi_yes_price / 100.0
    edge = calculate_edge(model_prob, implied_prob)
    rating = get_rating(edge)

    # Kalshi pays 100 cents for a winning contract that cost X cents
    payout = (100 - kalshi_yes_price) / kalshi_yes_price if kalshi_yes_price > 0 else 0
    ev = calculate_ev(model_prob, payout_if_win=payout, cost_if_loss=1.0)

    return {
        "edge": edge,
        "edge_pct": edge * 100,
        "rating": rating,
        "implied_prob": implied_prob,
        "model_prob": model_prob,
        "ev": ev,
    }
