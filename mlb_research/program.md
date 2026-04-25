# MLB Auto-Research Mission

You are running an autonomous research loop to improve an MLB P(home wins)
model. Drive down **optimizer Brier score** on a frozen 2025 anchor, subject
to an ROI regression cap. No human in the loop. Do not stop to ask whether
you should continue.

## What to optimize

**Primary objective:** minimize `opt_brier` in `mlb_research/results.tsv`.

**Keep-eligibility rules** (enforced by the runner, also stated here):
- `opt_brier` must improve by at least **0.010** vs. the running best.
  Smaller improvements are within the max-of-50-trials noise floor
  (SE(Brier) at n=1403 ≈ 0.007, expected max over 50 trials ≈ 0.015).
- `opt_roi` must not regress more than **3.0 units** vs. the running
  best.
- `opt_n_hc` must be at least **500**. A KEEP with fewer than 500
  high-conf picks is almost certainly a resolution-collapse artifact
  (Brier floor is 0.25 for uniform 0.5 output) rather than real alpha.

A "cumulative-delta" secondary gate was prototyped for Run 3 prep and
removed after adversarial review correctly observed that its 0.015
threshold was at the search-noise floor on the same window. Multi-
change candidates that the 0.010 primary floor rejects (e.g. Run 2's
`prune-13 + LGBM stumps = 0.2402` with marginal Δ=0.0018) will REVERT
in the autonomous loop. A human can still promote them via a
deliberate human-override row in the ledger -- see PR #80's
"HUMAN OVERRIDE" pattern.

**Hidden overfit guards:** `mon25_*` and `mon26_*` columns in
`results.tsv` exist for human review only. They detect whether your
improvements are real or are overfitting the 2025 Apr-Jul window. You
MUST NOT reference them when forming hypotheses or when deciding keep
vs revert. Pretend they are not there.

## Rules

1. **Edit scope (WHITELIST).** The ONLY files you may create or modify
   are under `mlb_research/`, with these specific exceptions EVEN
   WITHIN that directory that are also read-only during a run:
   - `mlb_research/anchor/mlb_frozen.csv`
   - `mlb_research/anchor/anchor_manifest.json`
   - `mlb_research/anchor/snapshot_data.py`
   - `mlb_research/anchor/anchor_eval.py` (changing evaluation
     mid-run invalidates all prior rows)
   - `mlb_research/run_experiment.py` (same)
   - `mlb_research/program.md` (this file)

   **Everything outside `mlb_research/` is read-only.** This includes
   but is not limited to `mlb/`, `model.py`, `backtest.py`,
   `predict.py`, `main.py`, `features.py`, `league_config.py`,
   `kalshi/`, `betting/`, the root training CSVs, `settings.py`,
   anything in `.claude/`. You may READ these files to understand
   production behavior; you may NOT edit them. If your hypothesis
   requires changing code outside `mlb_research/`, shadow the
   relevant logic inside `mlb_research/` and wrap it.

   You MUST NOT run `mlb_research/anchor/snapshot_data.py --force`
   during the loop. Re-freezing mid-run changes the anchor SHA256
   and invalidates every previously recorded metric.

   `rest_days` is a reserved derived column: `anchor_eval.py`
   recomputes it from `team` + `date` at load time to match
   production backtest semantics. Adding `rest_days` to a config's
   feature list uses the recomputed value. If you want an
   alternative rest-quality feature, give it a new name.

   `high_conf_threshold` is pinned at the harness level (0.53).
   Setting it in a config will cause the eval to exit with an error.

   To test a new model family, feature, or training scheme: write
   your own module under `mlb_research/`, emit a config JSON, and
   call the runner. Wrap, do not edit.

2. **Every experiment goes through the runner.**
   ```
   uv run python mlb_research/run_experiment.py run \
       --config path/to/config.json \
       --change-type <features|hyperparams|blend|gate|data|training|evaluation|other> \
       --description "short summary"
   ```
   The runner writes a `pending` row to `results.tsv`, archives the
   config and metrics JSON under `mlb_research/experiments/<ts>/`, and
   prints a keep/revert recommendation using optimizer columns only.

3. **Keep/revert decision (you apply this).**
   - **Keep** if the runner says `KEEP` -- commit the change with the
     metric in the title:
     ```
     git add -A
     git commit -m "Exp N: <desc> -- opt_brier=0.xxxx (was 0.yyyy)"
     uv run python mlb_research/run_experiment.py finalize --status kept
     ```
   - **Revert** if the runner says `REVERT` -- discard edits to tracked
     files and remove untracked files *under mlb_research/* only:
     ```
     git restore .
     git clean -fd -e 'mlb_research/experiments/' -e 'mlb_research/results.tsv'
     uv run python mlb_research/run_experiment.py finalize --status reverted
     ```
     Do NOT run a repo-wide `git clean -fd`; it will wipe archived
     experiment directories and the pending ledger row's referenced
     metrics. The row stays in `results.tsv` with `status=reverted`
     so you do not retry the same dead end.
   - Rows with `status=reverted` and `status=not-kept` are evidence.
     Before forming a new hypothesis, read them. Do not repeat what
     already failed.

4. **One change per experiment.** If you sweep two knobs at once you
   cannot attribute the result. Exception: combining changes that
   already individually passed the keep gate into a final merged
   config is allowed once every ~10 experiments.

5. **Running best.** The runner defines running best as the row with
   the lowest `opt_brier` among rows whose `status` is `baseline` or
   `kept`. Compare new experiments against the running best, not
   against the initial baseline.

6. **Stop conditions.** This run terminates when any of these fire:
   - You have written 50 new experiment rows since baseline.
   - The last 15 experiments have all been reverted or not-kept.
   - `results.tsv` or a script you depend on is corrupted.
   When you stop, write a final summary to
   `mlb_research/RUN_SUMMARY.md` covering: best config, optimizer
   Brier trajectory, which hypothesis families worked, which did not.

## The LOOP

Repeat until a stop condition fires:

1. **Read state.** `cat mlb_research/results.tsv`. Identify running
   best. Scan recent `reverted` rows to avoid repeating mistakes.
2. **Form one hypothesis.** Pick from the menu below or invent one.
   Write a one-line rationale: "If X is true, changing Y should lower
   Brier because Z."
3. **Implement.** Create or modify files under `mlb_research/`.
   Typically this means: writing a new config JSON, or writing a
   small Python module that generates a config + any required
   feature augmentation. If you introduce a new feature column, it
   must be derivable from the frozen CSV alone (no network, no live
   data).
4. **Run.** Invoke the runner.
5. **Decide.** Read the recommendation. Apply keep or revert
   mechanically.
6. **Commit.** One commit per experiment. Metric in the title.
   Finalize the ledger row.
7. **Next.**

Do not stop between iterations. Do not ask for permission. Do not
narrate; get on with it.

## MLB hypothesis menu

This is a starting menu, not exhaustive. Add your own.

### Feature engineering (add a new feature column)
- **Bullpen fatigue.** Days since home team's bullpen threw >2
  innings. Rolling bullpen IP over last 3 days.
- **Park factors.** 3-year rolling runs/HR index per venue (joined on
  `venue_name`). Already-present `wind_speed` suggests weather hooks
  work.
- **Umpire strike-zone bias.** If umpire name/id is in the CSV, build
  a rolling called-strike-rate feature.
- **Lineup handedness / platoon advantage.** Batting order vs starting
  pitcher hand.
- **Travel distance.** Home-team miles traveled in last 3 days.
- **Day-of-week / day-night turnaround.** Rest quality, not just
  rest days.
- **Line movement / closing line.** If `moneyline` history is in the
  CSV, movement magnitude.
- **Starter form vs team offense interaction.** `sp_roll_era *
  opp_roll_rpg` or similar cross-terms.

### Rolling window sweeps
- Sweep pitcher rolling window (3 / 5 / 7 / 10 starts).
- Sweep team rolling window (5 / 10 / 15 / 20 games).
- Sweep season-rolling blend (EMA decay constants).

### Model hyperparameters
- `n_estimators`: 50, 100, 150, 250, 400.
- `max_depth`: 2, 3, 4, 5, 6.
- `learning_rate`: 0.02, 0.05, 0.1, 0.2.
- `calibration_fraction`: 0.1, 0.2, 0.3.
- Subsampling / feature fraction (GradientBoostingClassifier does not
  natively support col sampling -- a LightGBM swap does).

### Model family swap (NOW REACHABLE)
- Set `"model_family": "lightgbm"` in config. Supported families:
  `sklearn_gbm` (default), `lightgbm`, `xgboost`.
- LightGBM and XGBoost also accept `subsample` and
  `colsample_bytree` in hyperparams (default 0.8 each).
- `calibrated: true` applies to all three families. The
  `sklearn_gbm` + default `calibration_method: "sigmoid"` path
  routes through the frozen `TimeAwareCalibratedGBM` so the
  baseline row stays bit-reproducible. Every other
  `(family, method)` pair routes through the generalized
  `TimeAwareCalibrated` wrapper, which does the same
  trailing-window split but accepts any base estimator and
  either sigmoid (Platt) or isotonic calibration. Set
  `"calibration_method": "isotonic"` to try isotonic. LightGBM
  and XGBoost calibrated + stumps is a concrete Run 3 starting
  hypothesis (uncalibrated LGBM stumps was Run 2's best KEEP).
- Simple two-model averaging ensemble (write a wrapper estimator
  under `mlb_research/` that exposes `fit`/`predict_proba`).
- Stacked: GBM + logistic regression meta-learner.

### Calibration variants
- Isotonic vs sigmoid.
- Sweep `min_calibration_rows`.
- No calibration at all (may help if it is introducing bias).

### Feature selection
- Permutation importance on a training fold, drop bottom-k.
- Greedy forward selection starting from `is_home` + a single strong
  feature.
- Correlation-based pruning (drop features with |r| > 0.9 to another
  feature).

### Target reformulation
- Predict run-line cover instead of moneyline win (not reachable --
  run_line is 100% NaN in the frozen CSV).
- **Margin regression + Normal CDF (NOW REACHABLE).** Set
  `"target": "margin"` in config. The harness trains the chosen
  `model_family` as a regressor on run margin, estimates residual
  σ on a trailing holdout slice (same split discipline as the
  calibration wrappers), and returns `P(home wins) = Φ(μ/σ)`. All
  three families are supported. `calibrated: true` is rejected in
  combination with `target: "margin"` -- post-hoc calibration on
  top of CDF output is a separate hypothesis; add it later inside
  `mlb_research/` if needed. Mirrors the CBB CDF projection trick.

### Data hygiene
- Season-scoped rolling stats (reset at season boundary). Current CSV
  may already do this; verify.
- Drop games with missing starter ERA (may be polluting the model).

### Gate / blend (applies only if you add a simulated betting filter)
- Market blend toward implied prob before computing ROI (tests if
  calibration shrinkage helps ROI without hurting Brier).
- High-confidence threshold sweep (0.52, 0.53, 0.55, 0.58).

### Meta / evaluation
- Do NOT add folds to the anchor eval. It is frozen.
- It is OK to add a new eval script under `mlb_research/` for your
  own investigation, as long as you do not change `anchor_eval.py`
  or any decisions it drives.

## Starting state

Baseline is the first row in `mlb_research/results.tsv`:
`opt_brier=0.2553, opt_roi=+54.55U` over 795 high-confidence picks
on 1403 games (2025 Apr-Jul walk-forward). That is the number to
beat.

### Context from Run 1

Run 1 (11 experiments, all reverted) established that:
- **Regularization is the dominant lever.** `max_depth=1` reached
  `opt_brier=0.2456` (delta -0.0097, missed 0.010 floor by 0.0003)
  with ROI doubling to +105U. Both silent monitors corroborated.
- Feature additions (park_factor, opponent raw features) did not
  help; diffs already absorb the signal.
- Disabling calibration was catastrophic (+0.010 Brier).
- Moneyline/total_line/run_line are 100% NaN in the frozen CSV.

**Promising direction for Run 2:** Try `max_depth=1` as a
*combined* change with another knob (lr or n_estimators) since its
individual delta was just under the floor. Also try LightGBM and
XGBoost (now reachable via the `model_family` config key). Feature
pruning is also untried.

Read `mlb_research/RUN_SUMMARY.md` for the full Run 1 trajectory.

### Context from Run 2

Run 2 produced one KEEP (exp 15): LightGBM `max_depth=1` stumps,
uncalibrated. Running best advanced from baseline `opt_brier=0.2553`
to `opt_brier=0.2420` with ROI more than doubling (+54.55U ->
+124.64U). 18 subsequent experiments all reverted or hit the
consecutive-non-keep cap; the run ended at 0.2420 with a clear
"structural ceiling" around 0.2400 given the harness surface area
available to Run 2.

Three patterns showed directional signal but could not cross the
0.010 primary floor alone:
- **Feature pruning.** Top-5 / top-8 / top-13 nonzero-importance
  prunings each beat LGBM-stumps-baseline by ~0.0018 Brier with
  ROI gains to +156U. Three independent prunings converging is
  pooled evidence for real signal, not noise.
- **Slower LR on stumps.** `learning_rate=0.03` was Brier-flat
  (0.2420) but ROI rose to +130U. Suggests pick composition
  shifted toward higher-EV picks without improving resolution.
- **Calibrated stumps (sklearn).** Sub-floor Brier drop but pick
  population dropped sharply (resolution collapse), masking a
  real effect. Was not reachable on LGBM/XGBoost in Run 2.

**Promising direction for Run 3** (newly reachable in Run 3 prep,
committed before this run started):

1. **Multi-change candidates as research signal.** LGBM stumps +
   prune-13 was `0.2402` in Run 2 (cumulative -0.0151 from
   baseline, marginal -0.0018 from running best). With the
   primary-only gate the autonomous loop will REVERT this; the
   reverted row is still useful evidence and a human can promote
   it via an explicit override row after the run if pooled
   evidence (three independent prunings landed at 0.2401-0.2407)
   holds up against the silent monitors.
2. **Calibrated LGBM/XGBoost stumps** via the new
   `calibration_method` knob (`"sigmoid"` or `"isotonic"`).
   Run 2 could only test sklearn calibration, which resolution-
   collapsed; LGBM/XGBoost raw output may respond better. Smoke
   runs during Run 3 prep showed both hurt Brier on the obvious
   naive application, so combine with feature pruning or
   different `calibration_fraction` to avoid re-discovering
   that failure mode.
3. **Margin regression + Normal CDF** (`target: "margin"`).
   Fundamentally different estimator surface -- the CBB model's
   approach applied to MLB. Test all three families. Prep-time
   smoke run (sklearn GBM depth=4 on margin) landed at 0.2554
   Brier but with +115.73U ROI on 1171 picks -- ROI path looks
   meaningfully different from the classifier path, so don't be
   surprised if Brier parity hides real pick-quality alpha.

Read `mlb_research/RUN_SUMMARY.md` for the full Run 2 trajectory
and the recommendation that motivated Run 3's prep.

## Calibration-policy confound (RUN 3 PREP, KNOWN LIMITATION)

The frozen `TimeAwareCalibratedGBM` (used by the baseline path:
`sklearn_gbm + calibrated + sigmoid`) calibrates whenever total
training rows clear `min_calibration_rows`, with no minimum on the
holdout slice itself. The new generalized `TimeAwareCalibrated`
wrapper (used by everything else: LightGBM/XGBoost calibrated, or
sklearn isotonic) ALSO requires `min_holdout_rows >= 50`. In early
walk-forward folds (e.g. when training history is in the 200-249
row window), the frozen path may calibrate on a 40-49 row holdout
while the new wrapper skips calibration entirely.

This means cross-family Brier comparisons against the frozen
baseline are PARTLY confounded by wrapper policy, especially in
early-season folds. The same is true for `target=margin` vs the
frozen path. The right resolution is a deliberate re-baseline (a
ledger-invalidating action) once the policy difference matters in
practice -- see "Anchor refresh" in the Run 3 prep notes. Pending
that, treat near-noise Brier deltas vs the baseline (~0.255) on
new-wrapper paths with extra suspicion; multi-experiment patterns
that hold across families are stronger evidence than any single
row.

## Known caveats (document in RUN_SUMMARY.md)

These are limitations of the benchmark itself that no amount of
agent effort can fix. Your final summary must acknowledge them:

1. **The 2025 benchmark is partially burned.** The human operator
   has previously analyzed this data while diagnosing live-model
   bugs (April 2026 early-season issues). Any feature or
   hyperparameter the operator already suspected was useful is
   implicitly part of the prior. A strong-looking result here may
   be a confirmation of priors, not a novel discovery.

2. **The 2026 monitor is thin.** ~190 rows / ~84-95 games / ~50-80
   high-conf picks. SE(Brier) on the 2026 monitor is ~0.018, which
   is *larger* than the optimizer's noise floor. It can veto
   disasters; it cannot confirm subtle alpha.

3. **Optimizer ↔ 2025-tail monitor share season-level signal.** Same
   rosters, same bullpens, overlapping team-quality signals. A
   feature that overfits 2025-level team strengths will look fine
   on the tail monitor. The 2026 monitor is the only true
   regime-change check, and it's thin (see #2).

4. **ROI is computed at flat -110.** Real MLB moneylines range
   roughly -300 to +250. A strategy that preferentially picks heavy
   favorites gets a flattered ROI here. Brier is not affected; ROI
   is. Interpret ROI as a directional tie-breaker, not an absolute.

5. **`opt_roi` has wide SE.** At n_hc≈795, SE(ROI) ≈ 27 units. The
   3-unit regression cap filters only catastrophes, not sub-noise
   regressions.

Any alpha that survives on BOTH the Aug-Oct monitor AND the 2026
monitor (even weakly) with `opt_brier` down by ≥0.010 is a real
candidate. Anything less is probably noise.
