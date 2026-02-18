"""Kelly criterion bet sizing."""


def kelly_fraction(
    edge: float,
    implied_prob: float,
    fraction: float = 0.25,
) -> float:
    """
    Calculate Kelly fraction for bet sizing.

    Full Kelly = edge / (1 - implied_prob)

    Derived from standard Kelly f* = (bp - q) / b where edge = p - implied_prob.

    We use fractional Kelly (default 1/4 Kelly) to reduce variance.

    Args:
        edge: Betting edge as decimal (e.g., 0.06 = 6%)
        implied_prob: Market implied probability (0-1)
        fraction: Kelly fraction to use (default 0.25 = quarter Kelly)

    Returns:
        Fraction of bankroll to wager (0-1)
    """
    if edge <= 0 or implied_prob <= 0 or implied_prob >= 1:
        return 0.0

    # Full Kelly = edge / (1 - implied_prob)
    # Equivalent to (bp - q) / b where b = (1-ip)/ip, p = ip + edge
    full_kelly = edge / (1 - implied_prob)

    # Apply fractional Kelly
    kelly = full_kelly * fraction

    # Never recommend more than 10% of bankroll
    return min(max(kelly, 0.0), 0.10)


def recommended_units(
    edge: float,
    implied_prob: float,
    fraction: float = 0.25,
    max_units: float = 3.0,
    unit_scale: float = 30,
) -> float:
    """
    Calculate recommended bet size in units.

    Args:
        edge: Betting edge as decimal
        implied_prob: Market implied probability (0-1)
        fraction: Kelly fraction (default 0.25)
        max_units: Maximum units to recommend (default 3.0)
        unit_scale: Multiplier to convert kelly fraction to units (default 30)

    Returns:
        Recommended units to bet (0 to max_units)
    """
    kelly = kelly_fraction(edge, implied_prob, fraction)

    # Convert to units
    units = kelly * unit_scale

    # Round to nearest 0.5 units
    units = round(units * 2) / 2

    # Cap at max
    return min(units, max_units)


def format_units(units: float) -> str:
    """Format units for display."""
    if units <= 0:
        return "PASS"
    elif units < 1:
        return f"{units:.1f}U"
    else:
        return f"{units:.1f}U"
