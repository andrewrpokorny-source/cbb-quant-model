# MLB Auto-Research Run Summary

## Outcome

Run 3 produced no automated keeps. It stopped at the
MAX_CONSECUTIVE_NON_KEEPS=15 guard after 15 reverted rows following the most
recent kept row.

The active optimizer best remains the Run 2 LightGBM max_depth=1 stumps row:
opt_brier=0.241967, opt_roi=+124.64U, n_hc=1143. The later human override row
for calibrated sklearn depth=1 remains a kept ledger row, but it is not the
optimizer best because its opt_brier is 0.245639.

Run 3's best candidate was LightGBM stumps margin-CDF with the Run 2 top-13
feature prune: opt_brier=0.240305, opt_roi=+139.73U, n_hc=895. It was reverted
because the marginal Brier improvement versus the active best was only 0.0017,
well below the 0.010 keep floor.

## Best Config

Automated best retained by the ledger:

- Archive: `mlb_research/experiments/20260422T154250Z_7fee946/`
- Model family: LightGBM
- Target: `home_win`
- Calibration: false
- Features: production-default 28-feature set
- Hyperparams: n_estimators=150, max_depth=1, learning_rate=0.05,
  random_state=42, subsample=0.8, colsample_bytree=0.8

Best Run 3 evidence row, not kept:

- Archive: `mlb_research/experiments/20260427T210059Z_606d3d3/`
- Model family: LightGBM
- Target: `margin`, converted to P(home wins) through Normal CDF
- Calibration: false
- Features: Run 2 top-13 prune
- Hyperparams: n_estimators=150, max_depth=1, learning_rate=0.05,
  random_state=42, subsample=0.8, colsample_bytree=0.8,
  calibration_fraction=0.2, min_calibration_rows=200

## Optimizer Brier Trajectory (Run 3)

Running best at start and end of Run 3 stayed at 0.241967.

| Row | Description | opt_brier | Delta vs active best | opt_roi | n_hc | Status |
|---:|---|---:|---:|---:|---:|---|
| 32 | LGBM stumps isotonic calibration | 0.248681 | +0.006714 | +87.82 | 1050 | reverted |
| 33 | LGBM stumps sigmoid calibration | 0.245424 | +0.003457 | +111.18 | 725 | reverted |
| 34 | XGBoost stumps sigmoid calibration | 0.245131 | +0.003164 | +102.00 | 780 | reverted |
| 35 | XGBoost stumps isotonic calibration | 0.249674 | +0.007707 | +119.00 | 952 | reverted |
| 36 | LGBM top-13 prune isotonic calibration | 0.247488 | +0.005521 | +131.36 | 1079 | reverted |
| 37 | LGBM top-13 prune sigmoid calibration | 0.243903 | +0.001936 | +124.45 | 727 | reverted |
| 38 | LGBM top-5 prune isotonic calibration | 0.247078 | +0.005111 | +141.00 | 867 | reverted |
| 39 | sklearn GBM margin CDF target | 0.248909 | +0.006942 | +111.64 | 1093 | reverted |
| 40 | LGBM stumps margin CDF target | 0.241162 | -0.000805 | +138.00 | 891 | reverted |
| 41 | XGBoost stumps margin CDF target | 0.241359 | -0.000608 | +136.45 | 925 | reverted |
| 42 | LGBM depth-2 margin CDF target | 0.241021 | -0.000946 | +146.91 | 1010 | reverted |
| 43 | LGBM margin CDF top-13 prune | 0.240305 | -0.001662 | +139.73 | 895 | reverted |
| 44 | LGBM margin CDF top-5 prune | 0.240795 | -0.001172 | +132.45 | 929 | reverted |
| 45 | LGBM depth-2 margin CDF residual fraction 0.3 | 0.240980 | -0.000987 | +108.18 | 1001 | reverted |
| 46 | LGBM depth-2 margin CDF residual floor 100 | 0.242600 | +0.000633 | +145.91 | 1074 | reverted |

## What Worked

- Margin-CDF was the only new Run 3 surface that consistently approached the
  active best. LightGBM and XGBoost stumps on the margin target both landed
  around 0.2412 to 0.2414, and LightGBM depth=2 reached 0.2410.
- Margin-CDF plus feature pruning was the strongest combined signal. The
  top-13 margin run reached 0.240305 with better ROI than the active kept
  config. This reinforces Run 2's pruning evidence, but the marginal gain is
  still too small for the autonomous gate.
- The fallback-share gate did useful work as a diagnostic. Most margin rows
  had only one std-of-y fallback fold out of 17. Lowering the residual-row
  floor removed that fallback, but Brier worsened, so the fallback fold was
  not the limiting issue.

## What Did Not Work

- Calibrated LightGBM/XGBoost stumps did not beat the uncalibrated LightGBM
  stumps best. Sigmoid was less bad than isotonic, but both moved Brier in the
  wrong direction or collapsed the high-confidence pick pool.
- Adding calibration on top of the Run 2 pruning signal did not unlock the
  pruned models. The best calibrated prune row was still 0.243903.
- sklearn GBM margin-CDF was not competitive at 0.248909. The useful margin
  signal appears tied to shallow LightGBM/XGBoost regularization.
- Residual-sigma holdout tweaks did not create a keep. A larger 30% holdout
  reduced ROI sharply, while a lower residual-row floor removed fallback usage
  but worsened Brier to 0.242600.

## Interpretation

Run 3 strengthens the same conclusion as Run 2: the current autonomous gate has
already found the one large single-change improvement available on this frozen
surface, namely shallow LightGBM. The remaining improvements cluster around
0.001 to 0.002 Brier, which is directionally interesting but below the 0.010
single-run floor.

The best research idea is no longer "try another single knob." It is a human
decision about whether repeated sub-floor evidence is enough to promote a
combined production candidate outside the autonomous rule. The most defensible
candidate for that is LightGBM margin-CDF plus top-13 pruning, with the caveat
that the monitor windows must be reviewed by a human because autonomous
decisions deliberately ignored them.

## Known Caveats

1. The 2025 benchmark is partially burned. The operator has previously
   diagnosed live-model issues against this data, so a strong-looking result
   may confirm prior beliefs rather than discover new alpha.
2. The 2026 monitor is thin. Its Brier standard error is larger than the
   optimizer noise floor, so it can veto disasters but cannot confirm subtle
   0.001 to 0.002 Brier gains.
3. Optimizer and 2025-tail monitor share season-level signal. A feature that
   overfits 2025 roster or bullpen quality can still look acceptable on both.
4. ROI is flat -110. Real MLB moneylines range widely, so ROI here is a
   directional tie-breaker, not a literal betting-return estimate.
5. opt_roi has wide standard error. The 3U regression cap catches major
   failures, not small ROI drift.
6. Market features remain unusable in the frozen anchor because moneyline,
   total_line, and run_line are effectively empty. The model is missing the
   highest-signal input available to a sports win-probability model.

## Human Follow-Up

1. Keep the automated production recommendation at LightGBM max_depth=1
   stumps unless a human explicitly promotes a combined candidate.
2. Review the monitor windows for `20260427T210059Z_606d3d3` before any
   override. If they do not veto it, LightGBM margin-CDF plus top-13 pruning is
   the best Run 3 candidate despite failing the autonomous floor.
3. For the next run, do not spend more budget on calibration variants of
   shallow LGBM/XGBoost. The ledger now has enough negative evidence there.
4. Re-freeze only deliberately, ideally with non-null market odds. Without
   market-implied probability, the current feature surface is probably near
   its practical ceiling.
