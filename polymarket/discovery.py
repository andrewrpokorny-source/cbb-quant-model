"""Schema discovery utility for the Polymarket Gamma API.

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
        print("Example: export POLYMARKET_PROXY=socks5h://127.0.0.1:8080")
        sys.exit(1)

    print("=== Geoblock check ===")
    print(json.dumps(client.check_geoblock(), indent=2))
    print()

    print("=== CLOB API health ===")
    print(f"OK: {client.is_ok()}")
    print()

    print("=== Sports metadata ===")
    sports = client.get_sports()
    for s in sports[:10]:
        print(f"  id={s.get('id')} sport={s.get('sport')} series={s.get('series')}")
    print(f"  ... ({len(sports)} total)")
    print()

    print("=== MLB game markets (first 3) ===")
    mlb = client.get_sports_game_markets("MLB", limit=10)
    for m in mlb[:3]:
        print(json.dumps({k: v for k, v in m.items() if k != "description"}, indent=2, default=str)[:500])
        print()
    print(f"  ... ({len(mlb)} total)")
    print()

    print("=== NCAAB game markets (first 3) ===")
    ncaab = client.get_sports_game_markets("NCAAB", limit=10)
    for m in ncaab[:3]:
        print(json.dumps({k: v for k, v in m.items() if k != "description"}, indent=2, default=str)[:500])
        print()
    print(f"  ... ({len(ncaab)} total)")
    print()

    # Print discovered field names from the first market
    sample = mlb[0] if mlb else (ncaab[0] if ncaab else None)
    if sample:
        print("=== Detected market fields ===")
        for key in sorted(sample.keys()):
            val = sample[key]
            val_preview = str(val)[:80]
            print(f"  {key}: {type(val).__name__} = {val_preview}")


if __name__ == "__main__":
    discover_schema()
