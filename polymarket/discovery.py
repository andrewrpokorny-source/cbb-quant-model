"""Schema discovery utility for the Polymarket sports CLI.

Run once to inspect the raw JSON output and confirm field names
before tuning the market mapper.

Usage:
    python -m polymarket.discovery
"""

import json
import sys

from .client import PolymarketClient


def discover_schema():
    client = PolymarketClient()

    if not client.proxy_url:
        print("POLYMARKET_PROXY not set. Set it to your SOCKS5 proxy URL.")
        print("Example: export POLYMARKET_PROXY=socks5://127.0.0.1:8080")
        sys.exit(1)

    print("=== Geoblock check ===")
    print(json.dumps(client.check_geoblock(), indent=2))
    print()

    print("=== CLOB API health ===")
    print(f"OK: {client.is_ok()}")
    print()

    print("=== Sports market types ===")
    types = client.get_sports_market_types()
    print(json.dumps(types[:5], indent=2))
    print()

    print("=== NCAAB markets (first 3) ===")
    ncaab = client.get_sports_markets("NCAAB", limit=3)
    print(json.dumps(ncaab[:3], indent=2))
    print()

    print("=== MLB markets (first 3) ===")
    mlb = client.get_sports_markets("MLB", limit=3)
    print(json.dumps(mlb[:3], indent=2))
    print()

    print("=== NCAA basketball search (first 3) ===")
    search = client.search_markets("NCAA basketball", limit=3)
    print(json.dumps(search[:3], indent=2))
    print()

    print("=== MLB search (first 3) ===")
    search_mlb = client.search_markets("MLB", limit=3)
    print(json.dumps(search_mlb[:3], indent=2))
    print()

    print("=== NCAAB teams (first 5) ===")
    teams = client.get_sports_teams("NCAAB", limit=5)
    print(json.dumps(teams[:5], indent=2))
    print()

    # Print discovered field names from the first NCAAB market
    if ncaab:
        print("=== Detected NCAAB market fields ===")
        for key in sorted(ncaab[0].keys()):
            val = ncaab[0][key]
            val_preview = str(val)[:80]
            print(f"  {key}: {type(val).__name__} = {val_preview}")


if __name__ == "__main__":
    discover_schema()
