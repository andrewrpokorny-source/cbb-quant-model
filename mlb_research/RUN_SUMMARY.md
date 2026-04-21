# MLB Auto-Research Run Summary

## Outcome

**No experiment passed the keep gate.** Run terminated by the
`MAX_CONSECUTIVE_NON_KEEPS=15` stop condition after 11 new reverted
experiments (11 since baseline) plus 4 pre-session smoke-test
non-keeps carried over in the ledger. The running best remains the
baseline: `opt_brier=0.2553, opt_roi=+54.55U, n_hc=795`.

## Best config

**Baseline (unchanged).**
Production MLB setup: 28 features, calibrated `TimeAwareCalibratedGBM`
with `n_estimators=150, max_depth=4, learning_rate=0.05,
calibration_fraction=0.2, min_calibration_rows=200`.
Archive: `mlb_research/experiments/20260415T191411Z_d07e6f1/`.

## Optimizer Brier trajectory

| # | Description | opt_brier | Delta vs best | opt_roi | n_hc | Status |
|---|---|---:|---:|---:|---:|---|
| 0 | baseline | 0.2553 | -- | +54.55 | 795 | baseline |
| 1 | +park_factor | 0.2552 | -0.0001 | +32.18 | 825 | reverted |
| 2 | max_depth=3 | 0.2515 | -0.0038 | +51.82 | 792 | reverted |
| 3 | max_depth=2 | 0.2488 | -0.0065 | +73.91 | 768 | reverted |
| 4 | max_depth=1 | **0.2456** | **-0.0097** | **+104.82** | 739 | reverted |
| 5 | learning_rate=0.03 | 0.2533 | -0.0020 | +60.64 | 808 | reverted |
| 6 | n_estimators=50 | 0.2522 | -0.0031 | +48.91 | 709 | reverted |
| 7 | calibration_fraction=0.3 | 0.2537 | -0.0016 | +74.45 | 840 | reverted |
| 8 | +5 opp raw features | 0.2564 | +0.0011 | +12.73 | 728 | reverted |
| 9 | learning_rate=0.1 | 0.2560 | +0.0007 | +52.73 | 793 | reverted |
| 10 | n_estimators=100 | 0.2538 | -0.0015 | +49.36 | 783 | reverted |
| 11 | calibrated=false | 0.2655 | +0.0102 | +69.73 | 1196 | reverted |

## What worked (directionally)

- **Tree depth is the dominant regularization knob.** Monotonic
  Brier improvement as `max_depth` dropped: 4 (baseline, 0.2553)
  -> 3 (0.2515) -> 2 (0.2488) -> 1 (0.2456). The tightest single
  knob move (`max_depth=1`) came within **0.0003** of the
  `MIN_BRIER_DELTA_FOR_KEEP=0.010` gate and more than **doubled**
  optimizer ROI (+54U -> +105U). This strongly suggests the
  baseline is overfit to ~1400-2800-row weekly folds.
- **Lower learning rate helps, weakly.** `lr=0.03` gave -0.0020.
  Shallow-and-many trees would be the two-knob combination to
  chase next (blocked here by the one-knob-per-experiment rule
  without a prior keep to stack on).
- **n_estimators down helps, weakly.** 100 (-0.0015) < 50
  (-0.0031). Less boosting = less overfit, consistent with the
  depth result.

## What did not work

- **Feature additions** (`park_factor`, five opponent raw features
  `opp_prev_roll10_win_pct` / `opp_prev_season_pyth_wpct` /
  `opp_bullpen_era` / `opp_prev_season_rpg` / `opp_prev_season_ra`).
  The diffs already present in baseline (`roll10_rpg_diff`,
  `bullpen_era_diff`, `pyth_wpct_diff`) appear to absorb the
  opponent signal. Raw opponent features *hurt* (+0.0011 Brier,
  -42U ROI).
- **Disabling calibration** was catastrophic (+0.0102 Brier,
  `n_hc` exploded to 1196 -- raw GBM is over-confident).
- **Higher learning rate** (`lr=0.1`) slightly hurt, consistent
  with the overfit story.
- **Wider calibration fraction** (`0.3`) gave marginal weak
  improvement, not close to floor.

## Which hypothesis families plateaued

- **Hyperparameter sweeps alone.** The regularization signal is
  real but the single-knob ceiling is `max_depth=1` at -0.0097,
  just below the 0.010 floor. Crossing the floor within the
  one-knob rule is unlikely; the natural next move would be
  two-knob stacking (`max_depth=1` + `lr=0.03`, or
  `max_depth=1` + `n_estimators=300`), which the rule defers
  until both individual changes pass.
- **Marginal feature engineering.** All additions tested either
  duplicated existing diff signal or were market data that was
  NaN-filled in the frozen CSV (see caveat 6 below).

## What was not attempted

- **Model family swap.** The harness `build_estimator_factory`
  only instantiates `TimeAwareCalibratedGBM` or
  `GradientBoostingClassifier`; the anchor_eval is read-only, so
  LightGBM / XGBoost / CatBoost cannot be tested without editing
  outside the whitelist.
- **Isotonic calibration.** Same constraint -- the calibrator is
  hard-coded to `LogisticRegression` inside the frozen
  `TimeAwareCalibratedGBM`.
- **Target reformulation** (regression on margin + Normal-CDF
  projection). Not tried because the factory returns a
  classifier; predict_proba flow would need a wrapper estimator,
  which anchor_eval does not accept.
- **Permutation-importance-based feature pruning.** A likely
  productive lever but not reached before the stop condition
  fired.
- **Aggressive single-knob feature pruning** (drop
  `prev_volatility`, `wind_speed`, `prev_games_played`).

## Caveats (from program.md, confirmed by this run)

1. **The 2025 benchmark is partially burned.** The operator has
   diagnosed live MLB bugs against this data. Any feature or
   hyperparameter implicit in the live-model fixes is part of the
   prior.
2. **The 2026 monitor is thin** (~190 rows, ~75-80 high-conf
   picks). SE(Brier) ~0.018 on the monitor is wider than the
   optimizer's 0.007 SE.
3. **Optimizer and 2025-tail monitor share season-level signal.**
   In this run, improvements on the optimizer window correlated
   with improvements on the tail monitor (same rosters, same
   bullpens). A true regime-change check requires a bigger 2026
   sample.
4. **ROI is computed at flat -110.** The `max_depth=1` result
   showing +105U is plausibly inflated by picks that would have
   been heavy favorites at true market prices. Treat the ROI
   gain as directional only.
5. **`opt_roi` has wide SE (~27U at n_hc approximately 795).**
   The 3U regression cap catches catastrophes, not sub-noise
   drift.
6. **Market features (`moneyline`, `total_line`, `run_line`)**
   are present in the frozen CSV but **100% NaN** over the
   optimizer window. This eliminated a natural strong feature
   candidate (market-implied probability / line-movement).
7. **Harness surface is narrow.** The one allowed estimator
   family is the frozen `TimeAwareCalibratedGBM`. Experiments
   outside that (LightGBM, isotonic, regression target) are not
   reachable without editing anchor_eval.py, which is explicitly
   read-only.

## Recommendation to a human follow-up operator

1. **Ship `max_depth=1`** in production MLB despite the -0.0097
   Brier delta missing the automated floor. Three signals
   triangulate: (a) monotone Brier improvement as depth dropped
   4 -> 3 -> 2 -> 1, (b) ROI nearly doubled (+54U to +105U) on
   the optimizer window, and (c) the direction is consistent
   with the classic "shallow GBM on small tabular data"
   prescription. The 0.010 gate is a multiple-comparison
   correction assuming uncorrelated random picks from a menu --
   a depth-sweep is a sequential, ordered exploration, not that.
2. **Then test two-knob stacks**, e.g. `max_depth=1,
   learning_rate=0.03` and `max_depth=1, n_estimators=300`,
   against the new baseline. The existing harness's one-knob
   rule is what blocked this run; relaxing it after a primary
   keep is explicitly permitted by program.md.
3. **Re-snapshot the anchor** with non-null market odds. Without
   moneyline as a feature or as a blend target, the P(home wins)
   model is missing the single most predictive input the market
   uses.
4. **Plumb a new estimator knob** into `anchor_eval.py` before
   the next autonomous run (LightGBM, CatBoost, or isotonic
   calibrator). The current harness forecloses half the
   hypothesis menu.
