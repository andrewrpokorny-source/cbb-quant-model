# Market Anchor V2 Freeze

Generated from `develop` after PR #84 was merged at `28f4ed8`.

## Frozen Artifacts

- Odds backfill: `data/mlb_espn_odds_backfill.csv`
- Enriched source: `data/mlb_training_data_with_espn_odds.csv`
- Frozen anchor: `mlb_research/anchor_v2/mlb_market_frozen.csv`
- Manifest: `mlb_research/anchor_v2/market_anchor_manifest.json`
- Anchor sha256: `c7613525d02009d3bc4807954a6aa502907ce17039b2c48869b2aa4993e34996`

Strict closing-odds coverage:

- Games: 2,566
- Complete paired moneyline games: 2,535
- Game coverage: 98.79%
- Source rows: 5,130
- Enriched rows with moneyline: 5,068
- Row coverage: 98.79%

The 2,566 / 2,535 counts are game-level odds-backfill counts. The manifest's
`home_rows=2565` and `home_complete_no_vig_rows=2534` are home-side rows present
in the enriched training anchor, so their denominator is one game lower.

Window sizes:

- Optimizer: 3,200 rows from 2025-04-01 through 2025-07-31
- Monitor 2025 tail: 1,610 rows from 2025-08-01 through 2025-10-31
- Monitor 2026: 190 rows from 2026-03-25 through 2026-04-02

## Initial Read

The market feature is useful, but the direct market-only baseline is still a
required yardstick. On the same top-13 row mask, direct no-vig market probability
scores optimizer Brier `0.242321` and loses `-74.90U` at actual prices. The best
initial model candidate, `lgbm_top13_market`, improves optimizer Brier to
`0.239482` and actual-priced optimizer ROI to `+51.47U`.

The production 28-feature model keeps its old Brier shape, but actual moneyline
pricing changes the interpretation of ROI: optimizer ROI is only `+1.37U`, not
the much larger flat -110 result from the legacy ledger. That confirms Track 2
is measuring a different and more realistic question.

The first shortlist after this freeze:

- `lgbm_top13_market`: best optimizer Brier and ROI among model candidates.
- `lgbm_stumps_market`: slightly weaker optimizer result, slightly stronger
  2025-tail ROI.
- `baseline_moneyline`: useful control for separating model changes from ROI
  scoring changes.

Do not promote from this branch solely on one season-plus-window snapshot. The
next research step is to repeat the shortlist with either a second historical
season or an independent odds source/snapshot policy, then check whether the
same top-13 plus market feature edge survives.
