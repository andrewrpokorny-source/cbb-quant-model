"""Betting calculations: edge, EV, and Kelly sizing."""

from .ev_calculator import (
    calculate_edge,
    calculate_kalshi_fee,
    get_rating,
    EdgeRating,
    analyze_bet,
)
from .kelly import kelly_fraction, recommended_units
from .line_shopping import (
    SpreadAnalysis,
    LineShoppingResult,
    calculate_line_shopping,
    find_breakeven_spread,
    format_line_shopping_text,
    STANDARD_IMPLIED_PROB,
)

__all__ = [
    "calculate_edge",
    "calculate_kalshi_fee",
    "get_rating",
    "EdgeRating",
    "analyze_bet",
    "kelly_fraction",
    "recommended_units",
    "SpreadAnalysis",
    "LineShoppingResult",
    "calculate_line_shopping",
    "find_breakeven_spread",
    "format_line_shopping_text",
    "STANDARD_IMPLIED_PROB",
]
