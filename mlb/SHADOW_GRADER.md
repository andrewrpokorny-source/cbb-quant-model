# MLB Shadow Grader

Grades the MLB market-v2 shadow model against actual game outcomes. The shadow
model emits `MarketV2_*` columns alongside production picks (see PR #87) but
does not drive bets. This grader is the live-evidence loop that lets us decide
whether to promote shadow to primary.

## Run

```bash
uv run python -m mlb.shadow_grader
```

Default behavior:

- Reads dated archives at `data/predictions_mlb_*.csv` (any with `MarketV2_*`
  columns; older archives are silently skipped).
- Joins outcomes from `data/mlb_training_data_processed.csv`
  (filtered to `is_home == 1`, dropping NaN `home_win`).
- Writes the per-game ledger to `data/mlb_shadow_grader_ledger.csv` (gitignored).
- Prints an aggregate report.

Useful flags:

- `--since YYYY-MM-DD` -- only grade archives on/after this date
- `--archive-dir PATH` / `--data-file PATH` / `--ledger PATH` -- override paths
- `--no-write` -- compute and report without persisting the ledger
- `--no-report` -- skip printing the aggregate report

The Streamlit MLB tab also surfaces the report (collapsed expander) when the
ledger file exists.

## Scoring conventions

- **Brier** is home-side throughout: `(prob_home - I[home_won])^2`. Production
  uses `Prob_Home`, shadow uses `MarketV2_Prob_Home`, market-only uses
  `MarketV2_Market_NoVig_Home`. Identical scoring lets the three deltas be
  compared directly.
- **Accuracy** is computed on each model's own pick.
- **ROI** is unit-stake at the production-pick moneyline (`Std_Odds`). Win
  payout is computed from the American line; loss is -1U.

## ROI gap

`Std_Odds` in the prediction archive only carries the moneyline for the
production pick's side. When shadow disagrees with production (or the
market-only pick disagrees), the off-side moneyline is unknown, so the ROI
cells are `None` and the row is flagged `roi_data_missing = True`. Such rows
are still graded for Brier/accuracy but are excluded from the ROI aggregate.

To fix this and unlock full-coverage ROI, persist both home and away
moneylines from `mlb/predict.py` (e.g. `Std_Odds_Home`, `Std_Odds_Away`) into
the archive.

## Ledger columns

| Column | Meaning |
|---|---|
| `archive_date` | Date the predictions archive was generated for. |
| `game_time` | HH:MM portion of the prediction's `Date/Time`. |
| `home_team` / `away_team` | Normalized team names (training-data form). |
| `matchup` | Original `Away @ Home` string from the archive. |
| `production_pick` / `shadow_pick` / `market_pick` | The three picks (ESPN display names). |
| `agrees_with_production` | Copied from `MarketV2_Agrees_With_Production`. |
| `outcome_status` | `graded` once outcome is in training data; `outcome_pending` otherwise. |
| `home_won` | 1 if home won, 0 if home lost, NaN while pending. |
| `production_correct` / `shadow_correct` / `market_correct` | Per-model correctness. |
| `production_brier` / `shadow_brier` / `market_brier` | Home-side Brier. |
| `production_prob_home` / `shadow_prob_home` / `market_prob_home` | Inputs to Brier. |
| `std_odds` | American moneyline at prediction time (production-pick side). |
| `production_roi_units` / `shadow_roi_units` / `market_roi_units` | Per-game ROI. |
| `roi_data_missing` | True when off-side moneyline isn't known or shadow/market disagrees with production. |
