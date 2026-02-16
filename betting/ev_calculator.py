"""Edge calculation and bet rating logic."""

from enum import Enum
from typing import Tuple


class EdgeRating(Enum):
    """Bet recommendation: bet or pass. Kelly handles sizing."""

    STRONG = "STRONG"  # 5%+ edge -- bet, let Kelly size it
    PASS = "PASS"  # <5% edge


# Rating threshold (edge percentage)
STRONG_THRESHOLD = 0.05  # 5%

# Kalshi fee constant (approximately 3.5% of payout at 50/50)
KALSHI_FEE_RATE = 0.035


def calculate_kalshi_fee(price: float) -> float:
    """
    Calculate Kalshi trading fee as a fraction of contract value.

    Fee formula: 0.035 * price * (1 - price)
    - At 50 cents: fee = 0.035 * 0.5 * 0.5 = 0.875% of contract
    - Fee is highest at 50/50, lower at extremes

    Args:
        price: Contract price as decimal (0-1), e.g., 0.52 for 52 cents

    Returns:
        Fee as fraction of potential payout
    """
    return KALSHI_FEE_RATE * price * (1 - price)


def calculate_edge(
    model_prob: float,
    market_implied_prob: float,
    include_fees: bool = True,
) -> float:
    """
    Calculate betting edge, optionally accounting for Kalshi fees.

    Edge = Model probability - Market implied probability - Fee adjustment

    Args:
        model_prob: Model's predicted probability (0-1)
        market_implied_prob: Market's implied probability (0-1)
        include_fees: Whether to subtract Kalshi fees (default True)

    Returns:
        Edge as decimal (e.g., 0.06 = 6% edge)
    """
    raw_edge = model_prob - market_implied_prob

    if include_fees:
        # Fee reduces effective edge
        # Fee is applied to payout, so we approximate impact on probability
        fee = calculate_kalshi_fee(market_implied_prob)
        # Effective edge is reduced by fee rate
        # (fee eats into your winnings, reducing EV)
        return raw_edge - fee

    return raw_edge


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


def analyze_bet(
    model_prob: float,
    kalshi_yes_price: float,
    include_fees: bool = True,
) -> dict:
    """
    Complete bet analysis including Kalshi fees.

    Args:
        model_prob: Model's predicted probability (0-1)
        kalshi_yes_price: Kalshi Yes price (0-100 scale)
        include_fees: Whether to account for Kalshi fees (default True)

    Returns:
        Dict with edge, rating, implied_prob, ev, fee
    """
    # Convert Kalshi price to probability
    implied_prob = kalshi_yes_price / 100.0

    # Calculate fee
    fee = calculate_kalshi_fee(implied_prob) if include_fees else 0

    # Calculate edge (includes fee by default)
    edge = calculate_edge(model_prob, implied_prob, include_fees=include_fees)

    # Get rating
    rating = get_rating(edge)

    # Calculate EV (assuming Kalshi standard payouts)
    # Kalshi pays 100 cents for a winning contract that cost X cents
    # Subtract fee from payout
    gross_payout = (100 - kalshi_yes_price) / kalshi_yes_price if kalshi_yes_price > 0 else 0
    net_payout = gross_payout * (1 - KALSHI_FEE_RATE) if include_fees else gross_payout
    ev = calculate_ev(model_prob, payout_if_win=net_payout, cost_if_loss=1.0)

    return {
        "edge": edge,
        "edge_pct": edge * 100,
        "rating": rating,
        "implied_prob": implied_prob,
        "model_prob": model_prob,
        "ev": ev,
        "fee_pct": fee * 100,
    }
