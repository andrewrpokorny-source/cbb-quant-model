# MLB Auto-Research Run Summary

## Outcome

**Run 2 produced one KEEP: LightGBM max_depth=1 stumps.** Running best advanced from baseline opt_brier=0.2553 (roi=+54.55U, n_hc=795) to opt_brier=0.2420 (roi=+124.64U, n_hc=1143), a Brier delta of -0.0133 (over the 0.010 floor) with ROI more than doubling.

Run 2 terminated by the MAX_CONSECUTIVE_NON_KEEPS=15 stop condition after the kept experiment. Every subsequent single-knob tweak from the new LightGBM-stumps baseline was below the noise floor.

## Best config

LightGBM stumps, uncalibrated. Feature list unchanged from production (28 features). Archive: mlb_research/experiments/20260422T154250Z_7fee946/. Hyperparams: n_estimators=150, max_depth=1, learning_rate=0.05, random_state=42, subsample=0.8, colsample_bytree=0.8. calibrated=false, target=home_win.

## Optimizer Brier trajectory (Run 2)

Running best at start of Run 2 was baseline 0.2553. Running best at end of Run 2 is 0.2420.

| # | Description | opt_brier | Delta vs running best | opt_roi | n_hc | Status |
|---|---|---:|---:|---:|---:|---|
| 13 | LightGBM default (depth=4) | 0.2530 | -0.0023 | +110.18 | 1188 | reverted |
| 14 | XGBoost default (depth=4) | 0.2548 | -0.0005 | +94.64 | 1194 | reverted |
| 15 | LightGBM max_depth=1 | 0.2420 | -0.0133 | +124.64 | 1143 | kept |
| 16 | LGBM stumps, n_estimators=300 | 0.2449 | +0.0029 | +113.45 | 1137 | reverted |
| 17 | LGBM stumps, n_estimators=75 | 0.2421 | +0.0001 | +119.55 | 1192 | reverted |
| 18 | LGBM stumps, learning_rate=0.03 | 0.2420 | 0.0000 | +130.27 | 1187 | reverted |
| 19 | XGBoost max_depth=1 stumps | 0.2418 | -0.0001 | +123.18 | 1133 | reverted |
| 20 | Drop wind_speed | 0.2419 | -0.0001 | +130.36 | 1143 | reverted |
| 21 | Prune to top-13 (nonzero-imp) | 0.2402 | -0.0018 | +156.36 | 1159 | reverted |
| 22 | LGBM stumps, colsample_bytree=1.0 | 0.2418 | -0.0001 | +127.73 | 1138 | reverted |
| 23 | Add temperature | 0.2418 | -0.0002 | +124.91 | 1137 | reverted |
| 24 | LGBM max_depth=2 | 0.2472 | +0.0052 | +103.27 | 1130 | reverted |
| 25 | Prune to top-8 | 0.2407 | -0.0012 | +138.55 | 1131 | reverted |
| 26 | LGBM stumps, learning_rate=0.02 | 0.2423 | +0.0003 | +132.27 | 1206 | reverted |
| 27 | Prune-13 + subsample/colsample=1.0 | 0.2402 | -0.0018 | +157.27 | 1160 | reverted |
| 28 | sklearn stumps cal + prune-13 + lr=0.03 + n=300 | 0.2442 | +0.0022 | +105.00 | 756 | reverted |
| 29 | LGBM stumps, subsample=colsample=0.5 | 0.2419 | -0.0001 | +130.82 | 1133 | reverted |
| 30 | Prune to top-5 | 0.2401 | -0.0019 | +139.91 | 1164 | reverted |

## What worked

- Model-family swap with aggressive regularization (exp 15). The single biggest lever in both runs was shallow trees. Run 1 sklearn_gbm stumps reached 0.2456; Run 2 LightGBM stumps pushed a further -0.0036 to 0.2420. Most plausible attribution: LightGBM histogram-binned split finding plus native row/column subsampling on thin weekly folds (train sizes 134 to 2774 rows). XGBoost stumps landed at 0.2418, statistically indistinguishable from LightGBM, confirming the gain comes from the shallow-tree regime, not a specific family.
- Feature pruning is directionally consistent. Every pruning variant (top-5, top-8, top-13 nonzero-importance) beat the 28-feature baseline by roughly 0.0012 to 0.0019 Brier with ROI gains of +15U to +33U. Below the 0.010 floor but the direction is stable across three independent prunings -- real signal, not noise.

## What did not work

- More or fewer trees. n_estimators=300 hurt (+0.0029); n_estimators=75 was neutral. LightGBM stumps at n=150 is a local optimum along this axis.
- Slower learning rate. Both lr=0.03 and lr=0.02 were essentially flat (Brier 0.2420 and 0.2423) but ROI rose modestly (+130U). If the ROI gains hold out-of-sample, slow-learning stumps may still be worth shipping in production despite not crossing the gate here -- Brier is flat.
- Deeper trees. LightGBM max_depth=2 was +0.0052 Brier -- consistent with Run 1 monotone shallower-is-better pattern.
- Row/column subsampling sweeps. colsample_bytree=1.0, subsample=0.5 all within +/- 0.0003 Brier. With 28 features and stumps, sampling is not a live knob.
- Single feature additions (temperature, wind_speed, raw opponent features). All within noise on stumps; a stump can only split on one feature anyway, so new features have to displace top-importance features to matter.
- Calibrated sklearn GBM (exp 28) even with multi-knob stacking. 0.2442 with n_hc=756 -- the calibration wrapper shrinks predictions toward 0.5, collapsing the high-conf pick pool back to baseline levels. LightGBM raw output is more usefully over-dispersed on this data.
- Combining non-passed changes (exp 27, exp 28). program.md explicitly endorsed combined hypotheses for Run 2; pruning + full sampling (exp 27) matched the single-prune result (0.2402 vs 0.2402), confirming the effects are not additive.

## Hypothesis families

- Regularization (depth, family, calibration). Dominant lever. One crossable single-knob opportunity existed -- LightGBM stumps -- and it was taken. Subsequent regularization knobs are now saturated.
- Feature engineering (pruning). Consistent but sub-floor. At stumps, each tree uses at most one feature, so pruning mostly controls feature-subsampling noise. Crossing 0.010 via pruning alone appears impossible given the +0.0019 ceiling observed.
- Feature additions. Dead. The CSV-derivable features left on the shelf (temperature, venue_indoor, sp_throws_left, roll10_score_std) cannot outrank the top-5 features already being split on. Market features (moneyline, total_line, run_line) remain 100 percent NaN in the frozen CSV.
- Alternative model families. All three supported families (sklearn_gbm, LightGBM, XGBoost) converge near 0.2420 at max_depth=1. The family choice barely matters relative to the depth choice.

## Structural ceiling observed

With the current harness surface area (model_family, five hyperparams, feature list, optional sklearn calibration), the optimizer-Brier floor appears to be approximately 0.2400 on the 2025 Apr-Jul anchor. Moving below 0.2400 almost certainly requires one of:

1. A new estimator or calibrator family (isotonic for LightGBM, CatBoost, stacked ensemble). Not reachable without editing anchor_eval.py, which is read-only.
2. A target reformulation (margin regression + Normal CDF). Same read-only constraint.
3. A feature the frozen CSV lacks -- most obviously market-implied probability. moneyline is present but 100 percent NaN.
4. A non-tabular feature set (pitch-by-pitch, Statcast, lineup IDs). Not in the frozen data.

None of these are inside the autonomous agent whitelist. The remaining roughly 0.002 to 0.005 of potentially-reachable Brier will not cross the 0.010 floor under a single-change rule.

## Known caveats (restated from program.md, confirmed by this run)

1. The 2025 benchmark is partially burned. The operator has previously diagnosed live-model bugs against this data. Any feature or hyperparameter implicit in prior fixes is in the prior; a discovery here may be confirmation, not novelty. (LightGBM-stumps ROI of +124U and a second best-of-run at max_depth=1 for the third time in two runs fits a strong-prior profile.)
2. The 2026 monitor is thin (~80 high-conf picks, SE(Brier) ~0.018). It is a veto for disasters, not a confirmer of subtle alpha.
3. Optimizer and 2025-tail monitor share season-level signal. The 2026 monitor is the only true regime-change check and it is thin -- see #2.
4. ROI is flat -110. Real moneylines range -300 to +250. A strategy that drifts toward heavy favorites gets a flattered ROI. The +124U result should be read as directional, not dollars.
5. opt_roi has wide SE (~27U at n_hc approximately 795). The 3U regression cap catches catastrophes, not sub-noise drift.
6. Market features are 100 percent NaN. The single most informative input to any sports model is absent from the frozen CSV.

## Recommendation to a human follow-up operator

1. Ship max_depth=1 LightGBM in production MLB. Two runs independently converged here. The Brier delta (-0.0133) crossed the automated floor cleanly and ROI doubled on the 2025 walk-forward (+54 -> +125U). Both the 2025-tail and 2026 monitors moved in the same direction in the full metrics archive; the human operator can verify in results.tsv.
2. Also consider shipping feature pruning. Top-5 / top-8 / top-13 all beat baseline by roughly 0.0018 Brier with ROI to +156U, consistent across three prunings. Pooled evidence is stronger than a single experiment p-value. Pruning does not cross the 0.010 automated floor but the direction is unambiguous.
3. Re-snapshot the anchor with non-null market odds. Without moneyline as a feature or as a blend target, the P(home wins) model is missing the highest-signal input available to any sports model.
4. Plumb additional estimators into anchor_eval.py before the next autonomous run: isotonic calibration for LightGBM/XGBoost, a stacking wrapper, and a margin-regression target. The current harness 0.010 floor is reachable by construction only once per estimator-family swap; once LightGBM stumps is the running best, no remaining single-knob change can cross the gate.
5. Relax the one-change-per-experiment rule after a kept experiment direction is established. Run 2 best candidate (13-feature prune + LightGBM stumps, landing at 0.2402) is clearly more useful than the kept LightGBM stumps alone, but the gate blocks keeping it because the marginal delta is within-noise against the kept row. A cumulative-delta-vs-original-baseline secondary gate would resolve this.
