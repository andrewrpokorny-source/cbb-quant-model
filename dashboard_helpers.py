"""Helpers for the Streamlit dashboard that can be imported without Streamlit."""

from datetime import datetime, timedelta


def filter_recent_kalshi(bets, *, cutoff_days=7, limit=8):
    """Return recent settled Kalshi results, sorted newest-first.

    Args:
        bets: list of dicts (as read by csv.DictReader from betting_history.csv)
        cutoff_days: only include results from the last N days
        limit: max number of results to return
    """
    cutoff = (datetime.now() - timedelta(days=cutoff_days)).strftime("%Y-%m-%d")
    filtered = [
        r for r in bets
        if (r.get("platform") or "").strip().upper() == "KALSHI"
        and (r.get("date") or "") >= cutoff
        and (r.get("result") or "").strip().lower() in ("win", "loss", "void")
    ]
    filtered.sort(key=lambda r: r.get("date") or "", reverse=True)
    return filtered[:limit]
