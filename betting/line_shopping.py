"""Line shopping recommendations for spread betting."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.stats import norm

from .kelly import recommended_units

# Standard -110 odds implied probability
STANDARD_IMPLIED_PROB = 0.5238


@dataclass
class SpreadAnalysis:
    """Analysis for a single spread value."""
    spread: float
    model_prob: float
    edge: float
    kelly_units: float
    is_market: bool = False


@dataclass
class LineShoppingResult:
    """Complete line shopping analysis for a bet."""
    picked_team: str
    market_spread: float
    breakeven_spread: Optional[float]
    recommendations: List[SpreadAnalysis]


def calculate_line_shopping(
    margin_model,
    sigma: float,
    base_features: dict,
    market_spread: float,
    picked_team: str,
    is_home_pick: bool,
) -> LineShoppingResult:
    """
    Calculate line shopping recommendations using margin model + norm.cdf.

    Uses P(home covers) = norm.cdf((predicted_margin + home_spread) / sigma)
    which is inherently monotonic -- no PCHIP smoothing needed.

    Args:
        margin_model: Trained margin regression model
        sigma: Standard deviation of training residuals
        base_features: Feature dict for margin model (no spread needed)
        market_spread: Current market spread (from picked team's perspective)
        picked_team: Name of the team we're betting on
        is_home_pick: True if picked team is home team

    Returns:
        LineShoppingResult with breakeven spread and recommendations ladder
    """
    from model_margin import predict_margin

    # Predict margin once (fixed for this matchup)
    predicted_margin = predict_margin(margin_model, base_features)

    recommendations = []

    # Generate spreads: market +/- 2 points in 0.5 increments
    spread_range = np.arange(market_spread - 2.0, market_spread + 2.5, 0.5)

    for spread_val in spread_range:
        # Convert picked-team spread to home-team spread
        if is_home_pick:
            home_spread = spread_val
        else:
            home_spread = -spread_val

        # P(home covers) = norm.cdf((predicted_margin + home_spread) / sigma)
        home_cover_prob = norm.cdf((predicted_margin + home_spread) / sigma)

        # Convert to P(picked team covers)
        if is_home_pick:
            model_prob = home_cover_prob
        else:
            model_prob = 1 - home_cover_prob

        # Calculate edge vs standard -110 odds
        edge = model_prob - STANDARD_IMPLIED_PROB

        # Get Kelly units (0 if negative edge)
        units = recommended_units(edge, STANDARD_IMPLIED_PROB)

        # Check if this is the market spread
        is_market = abs(spread_val - market_spread) < 0.01

        recommendations.append(SpreadAnalysis(
            spread=spread_val,
            model_prob=model_prob,
            edge=edge,
            kelly_units=units,
            is_market=is_market,
        ))

    # Find breakeven spread via interpolation
    breakeven = find_breakeven_spread(recommendations)

    return LineShoppingResult(
        picked_team=picked_team,
        market_spread=market_spread,
        breakeven_spread=breakeven,
        recommendations=recommendations,
    )


def find_breakeven_spread(recommendations: List[SpreadAnalysis]) -> Optional[float]:
    """
    Find the spread where edge = 0 via linear interpolation.

    Returns None if all spreads have positive edge (very favorable)
    or if no interpolation is possible.
    """
    # Sort by spread value
    sorted_recs = sorted(recommendations, key=lambda x: x.spread)

    # Look for sign change in edge
    for i in range(len(sorted_recs) - 1):
        curr = sorted_recs[i]
        next_rec = sorted_recs[i + 1]

        # Check for sign change (positive to negative or vice versa)
        if curr.edge * next_rec.edge < 0:
            # Linear interpolation: find where edge crosses zero
            # edge = 0 at: spread = curr.spread + (next.spread - curr.spread) * (0 - curr.edge) / (next.edge - curr.edge)
            denom = next_rec.edge - curr.edge
            if abs(denom) > 0.0001:
                t = -curr.edge / denom
                breakeven = curr.spread + t * (next_rec.spread - curr.spread)
                # Round to nearest 0.5
                return round(breakeven * 2) / 2

    # If all edges are positive, the breakeven is beyond our range
    if all(r.edge > 0 for r in sorted_recs):
        return None

    # If all edges are negative, return the most favorable spread we checked
    return None


def format_spread(spread: float) -> str:
    """Format spread for display with +/- sign."""
    if spread > 0:
        return f"+{spread:.1f}"
    else:
        return f"{spread:.1f}"


def format_line_shopping_text(result: LineShoppingResult) -> str:
    """Format line shopping result as text for display."""
    lines = []

    # Header with breakeven
    if result.breakeven_spread is not None:
        lines.append(f"Breakeven: {result.picked_team} {format_spread(result.breakeven_spread)}")
    else:
        lines.append("Breakeven: Beyond range (all positive edge)")

    lines.append("")
    lines.append("Spread    Model %    Edge      Units")

    for rec in result.recommendations:
        spread_str = format_spread(rec.spread)
        model_pct = f"{rec.model_prob:.1%}"
        edge_str = f"{rec.edge * 100:+.1f}%"

        if rec.kelly_units > 0:
            units_str = f"{rec.kelly_units:.1f}U"
        else:
            units_str = "PASS"

        marker = ""
        if rec.is_market:
            marker = " [MARKET]"
        elif result.breakeven_spread is not None and abs(rec.spread - result.breakeven_spread) < 0.3:
            marker = " [BREAKEVEN]"

        lines.append(f"{spread_str:8} {model_pct:9} {edge_str:9} {units_str:5}{marker}")

    return "\n".join(lines)
