# MLB Market Anchor V2

Track 2 is blocked on source data, not model code. The current
`data/mlb_training_data_processed.csv` has the `moneyline`, `run_line`, and
`total_line` columns, but moneyline coverage is 0%, so the existing frozen
anchor cannot answer market-aware research questions.

Use the diagnostic command before freezing any market-aware anchor:

```bash
uv run python mlb_research/anchor_v2/build_market_anchor.py --diagnose-only
```

The freeze command intentionally fails unless paired moneyline coverage clears
the configured threshold:

```bash
uv run python mlb_research/anchor_v2/build_market_anchor.py --min-coverage 0.95
```

When historical odds are backfilled, this builder adds:

- `team_moneyline`
- `opp_moneyline`
- `market_team_implied_prob`
- `market_opp_implied_prob`
- `market_overround`
- `market_team_no_vig_prob`
- `market_home_no_vig_prob`

The next real research run should use `market_home_no_vig_prob` as a feature
and `roi_mode="moneyline"` in configs so ROI is scored at actual prices rather
than flat -110.

## Odds Source Requirement

Yes, Track 2 needs an odds source before it can produce model conclusions.
The minimum acceptable source is historical game-level MLB moneylines for both
sides of each game, keyed closely enough to join on date, teams, and preferably
game time for doubleheaders. Closing lines are preferred; a fixed pre-game
snapshot time is acceptable if it is consistent and documented.

The builder is intentionally vendor-neutral. It expects a processed MLB source
CSV with paired `moneyline` values. The original source remains
`data/mlb_training_data_processed.csv`; the frozen market-v2 run uses the
backfilled/enriched source `data/mlb_training_data_with_espn_odds.csv`.

## ESPN Core Odds Backfill

The regular ESPN scoreboard endpoint is not sufficient for completed games:
historical scoreboard responses often omit the `odds` block. ESPN's core
competition odds endpoint does retain historical provider odds by event ID.

Use the backfill helper to discover event IDs from the scoreboard and fetch the
core odds endpoint:

```bash
uv run python mlb_research/anchor_v2/backfill_espn_odds.py \
  --start-date 2025-03-27 \
  --end-date 2026-04-02 \
  --basis-order close \
  --odds-output data/mlb_espn_odds_backfill.csv \
  --enriched-output data/mlb_training_data_with_espn_odds.csv
```

Observed sample behavior from the May 1, 2026 source check:

- `2025-04-15`: 15/15 games had ESPN BET closing moneylines.
- `2025-07-01`: 14/15 games had closing moneylines.
- `2026-04-01`: 15/15 games had DraftKings closing moneylines.
- `2026-05-01`: 15/15 games had current DraftKings moneylines, but no closing
  moneylines yet because games had not completed.

For final research, prefer a completed historical window and `close` prices.
For same-day/live production, use `current` prices from the prediction-time
snapshot rather than closing prices.

## Frozen Market Anchor

The frozen market-v2 branch keeps generated artifacts under this directory and
uses a separate `results.tsv` from the original no-market research ledger.

```bash
uv run python mlb_research/anchor_v2/build_market_anchor.py \
  --source data/mlb_training_data_with_espn_odds.csv \
  --output mlb_research/anchor_v2/mlb_market_frozen.csv \
  --manifest mlb_research/anchor_v2/market_anchor_manifest.json \
  --min-coverage 0.95 \
  --force
```

For model evals against the market anchor, set:

```bash
MLB_RESEARCH_FROZEN_CSV=mlb_research/anchor_v2/mlb_market_frozen.csv
MLB_RESEARCH_ANCHOR_MANIFEST=mlb_research/anchor_v2/market_anchor_manifest.json
```
