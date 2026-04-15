# MLB Auto-Research Mission

You are running an autonomous research loop to improve an MLB P(home wins)
model. Drive down **optimizer Brier score** on a frozen 2025 anchor, subject
to an ROI regression cap. No human in the loop. Do not stop to ask whether
you should continue.

## What to optimize

**Primary objective:** minimize `opt_brier` in `mlb_research/results.tsv`.

**Secondary gate:** `opt_roi` must not regress more than `3.0` units
compared to the running best.

**Hidden overfit guards:** `mon25_*` and `mon26_*` columns in
`results.tsv` exist for human review only. They detect whether your
improvements are real or are overfitting the 2025 Apr-Jul window. You
MUST NOT reference them when forming hypotheses or when deciding keep vs
revert. Pretend they are not there.

## Rules

1. **Edit scope.** You may create/modify any file under `mlb_research/`.
   You MUST NOT edit any of the following (they belong to the live
   production pipeline):
   - `mlb/` (any file)
   - `model.py`, `backtest.py`, `predict.py`, `main.py`, `features.py`
   - `mlb_training_data_processed.csv`
   - `mlb_research/anchor/mlb_frozen.csv` (the frozen benchmark data)
   - `mlb_research/anchor/anchor_manifest.json`
   - `mlb_research/anchor/snapshot_data.py`
   - `mlb_research/anchor/anchor_eval.py` (evaluation must not change
     mid-run; changing it invalidates all prior rows in `results.tsv`)

   To test a new model family, new feature, or new training scheme:
   write your own module under `mlb_research/`, emit a config JSON, and
   call the runner. Wrap, don't edit.

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
   - **Revert** if the runner says `REVERT` -- discard edits and mark
     the row reverted:
     ```
     git restore . && git clean -fd
     uv run python mlb_research/run_experiment.py finalize --status reverted
     ```
     The row stays in `results.tsv` with `status=reverted` so you do
     not retry the same dead end.
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

### Model family swap
- LightGBM via `lightgbm.LGBMClassifier` (wrap it, expose same
  `fit`/`predict_proba` interface).
- XGBoost via `xgboost.XGBClassifier`.
- CatBoost via `catboost.CatBoostClassifier`.
- Simple two-model averaging ensemble.
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
- Predict run-line cover instead of moneyline win.
- Predict score margin (regression) and derive P(home wins) from
  Normal CDF at 0 -- mirror CBB's CDF projection trick.

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
