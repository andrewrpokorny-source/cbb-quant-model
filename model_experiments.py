"""
Model Improvement Experiments
Tests Phase 1 improvements from IMPROVEMENT_PLAN.md
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Note: XGBoost not installed. Install with: pip install xgboost")

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data/cbb_training_data_processed.csv")

# Current features used in production model
BASELINE_FEATURES = [
    'is_home',
    'spread',
    'rest_days',
    'diff_eFG',
    'diff_Rebound',
    'diff_TO',
    'momentum_gap',
    'roll5_cover_margin'
]


def load_and_prep_data():
    """Load data and prepare for experiments"""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"   -> Loaded {len(df)} rows")
    return df


def add_defensive_features(df):
    """Add defensive efficiency metrics"""
    print("   -> Adding defensive features...")

    # Defensive Rating (points allowed per 100 possessions estimate)
    # We need to calculate what opponent scored against us
    df['def_rating'] = 100 * (df['opp_score'] / df['poss'].clip(lower=1))

    # Rolling defensive stats
    df = df.sort_values(['team', 'date']).reset_index(drop=True)

    # Season average defensive rating
    df['season_def_rating'] = df.groupby('team')['def_rating'].expanding().mean().reset_index(level=0, drop=True)
    df['prev_season_def_rating'] = df.groupby('team')['season_def_rating'].shift(1)

    # Opponent's offensive rating (what they typically score)
    # This comes from opp_season_off_rating if it exists
    if 'opp_season_off_rating' in df.columns:
        # Defensive differential: opponent's typical offense vs what we allow
        df['diff_DRtg'] = df['opp_season_off_rating'] - df['prev_season_def_rating']
    else:
        df['diff_DRtg'] = 0

    return df


def add_opponent_rest_feature(df):
    """Add opponent rest days differential"""
    print("   -> Adding opponent rest differential...")

    df = df.sort_values(['team', 'date']).reset_index(drop=True)

    # Calculate rest days for each team (already exists as rest_days for team)
    # Need to get opponent's rest days

    # Create lookup of team rest by date
    rest_lookup = df[['date', 'team', 'rest_days']].copy()
    rest_lookup = rest_lookup.rename(columns={'team': 'opp_name', 'rest_days': 'opp_rest_days'})

    # Merge opponent rest
    df = pd.merge(df, rest_lookup,
                  left_on=['date', 'opponent'],
                  right_on=['date', 'opp_name'],
                  how='left')

    # Rest differential (positive = we have more rest)
    df['rest_differential'] = df['rest_days'] - df['opp_rest_days'].fillna(df['rest_days'])

    # Clean up
    if 'opp_name' in df.columns:
        df = df.drop(columns=['opp_name'])

    return df


def add_pace_features(df):
    """Add pace/tempo features"""
    print("   -> Adding pace features...")

    df = df.sort_values(['team', 'date']).reset_index(drop=True)

    # Pace is possessions per game
    df['season_pace'] = df.groupby('team')['poss'].expanding().mean().reset_index(level=0, drop=True)
    df['prev_season_pace'] = df.groupby('team')['season_pace'].shift(1)

    # Opponent pace (need lookup)
    if 'opponent' in df.columns:
        pace_lookup = df[['date', 'team', 'prev_season_pace']].copy()
        pace_lookup = pace_lookup.rename(columns={'team': 'opp_name', 'prev_season_pace': 'opp_pace'})

        df = pd.merge(df, pace_lookup,
                      left_on=['date', 'opponent'],
                      right_on=['date', 'opp_name'],
                      how='left')

        # Pace differential
        df['pace_differential'] = df['prev_season_pace'] - df['opp_pace'].fillna(df['prev_season_pace'])

        if 'opp_name' in df.columns:
            df = df.drop(columns=['opp_name'])

    return df


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Evaluate a model and return metrics"""
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    logloss = log_loss(y_test, probs)

    # Actionable accuracy (>53% confidence)
    high_conf_mask = (probs > 0.53) | (probs < 0.47)
    if high_conf_mask.sum() > 0:
        high_conf_acc = accuracy_score(y_test[high_conf_mask], preds[high_conf_mask])
        high_conf_count = high_conf_mask.sum()
    else:
        high_conf_acc = 0
        high_conf_count = 0

    return {
        'model': model_name,
        'accuracy': acc,
        'brier_score': brier,
        'log_loss': logloss,
        'high_conf_acc': high_conf_acc,
        'high_conf_count': high_conf_count
    }


def run_time_series_cv(df, features, model, model_name, n_splits=5):
    """Run time series cross-validation"""
    # Clean data
    df_clean = df.dropna(subset=features + ['ats_win']).copy()

    X = df_clean[features].astype(float)
    y = df_clean['ats_win'].astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        result['fold'] = fold + 1
        results.append(result)

    return results


def experiment_baseline(df):
    """Establish baseline with current model"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: BASELINE (Current Production Model)")
    print("="*60)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    results = run_time_series_cv(df, BASELINE_FEATURES, model, "RF Baseline")

    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_brier = np.mean([r['brier_score'] for r in results])
    avg_hc_acc = np.mean([r['high_conf_acc'] for r in results])

    print(f"\n   Baseline Results (5-fold Time Series CV):")
    print(f"   - Accuracy:           {avg_acc:.2%}")
    print(f"   - Brier Score:        {avg_brier:.4f}")
    print(f"   - High Conf Accuracy: {avg_hc_acc:.2%}")

    return {'baseline': {'accuracy': avg_acc, 'brier': avg_brier, 'high_conf_acc': avg_hc_acc}}


def experiment_new_features(df):
    """Test with additional features"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: ENHANCED FEATURES")
    print("="*60)

    # Add new features
    df = add_defensive_features(df)
    df = add_opponent_rest_feature(df)
    df = add_pace_features(df)

    # Define feature sets to test
    feature_sets = {
        'baseline': BASELINE_FEATURES,
        'with_defense': BASELINE_FEATURES + ['diff_DRtg'],
        'with_rest_diff': BASELINE_FEATURES + ['rest_differential'],
        'with_pace': BASELINE_FEATURES + ['pace_differential'],
        'all_new': BASELINE_FEATURES + ['diff_DRtg', 'rest_differential', 'pace_differential']
    }

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    results = {}
    for name, features in feature_sets.items():
        # Filter to existing features
        available_features = [f for f in features if f in df.columns]

        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            print(f"   Note: Missing features for {name}: {missing}")

        cv_results = run_time_series_cv(df, available_features, model, name)

        avg_acc = np.mean([r['accuracy'] for r in cv_results])
        avg_brier = np.mean([r['brier_score'] for r in cv_results])
        avg_hc_acc = np.mean([r['high_conf_acc'] for r in cv_results])

        results[name] = {'accuracy': avg_acc, 'brier': avg_brier, 'high_conf_acc': avg_hc_acc}
        print(f"\n   {name}:")
        print(f"   - Accuracy:           {avg_acc:.2%}")
        print(f"   - High Conf Accuracy: {avg_hc_acc:.2%}")

    return results, df


def experiment_models(df, features):
    """Compare different model architectures"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: MODEL COMPARISON")
    print("="*60)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.04, max_depth=3,
            subsample=0.8, random_state=42, eval_metric='logloss'
        )

    results = {}
    for name, model in models.items():
        cv_results = run_time_series_cv(df, features, model, name)

        avg_acc = np.mean([r['accuracy'] for r in cv_results])
        avg_brier = np.mean([r['brier_score'] for r in cv_results])
        avg_hc_acc = np.mean([r['high_conf_acc'] for r in cv_results])

        results[name] = {'accuracy': avg_acc, 'brier': avg_brier, 'high_conf_acc': avg_hc_acc}
        print(f"\n   {name}:")
        print(f"   - Accuracy:           {avg_acc:.2%}")
        print(f"   - Brier Score:        {avg_brier:.4f}")
        print(f"   - High Conf Accuracy: {avg_hc_acc:.2%}")

    return results


def experiment_ensemble(df, features):
    """Test ensemble methods"""
    print("\n" + "="*60)
    print("EXPERIMENT 4: ENSEMBLE METHODS")
    print("="*60)

    # Base models
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    estimators = [('rf', rf), ('gb', gb), ('lr', lr)]

    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.04, max_depth=3, subsample=0.8, random_state=42, eval_metric='logloss')
        estimators.append(('xgb', xgb_model))

    # Voting classifier (soft voting only - hard voting doesn't support predict_proba)
    soft_voting = VotingClassifier(estimators=estimators, voting='soft')

    results = {}

    for name, model in [('Soft Voting', soft_voting)]:
        cv_results = run_time_series_cv(df, features, model, name)

        avg_acc = np.mean([r['accuracy'] for r in cv_results])
        avg_hc_acc = np.mean([r['high_conf_acc'] for r in cv_results])

        results[name] = {'accuracy': avg_acc, 'high_conf_acc': avg_hc_acc}
        print(f"\n   {name} Ensemble:")
        print(f"   - Accuracy:           {avg_acc:.2%}")
        print(f"   - High Conf Accuracy: {avg_hc_acc:.2%}")

    return results


def experiment_calibration(df, features):
    """Test probability calibration"""
    print("\n" + "="*60)
    print("EXPERIMENT 5: PROBABILITY CALIBRATION")
    print("="*60)

    base_model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1
    )

    # Calibrated models
    calibrated_sigmoid = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated_isotonic = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

    results = {}

    for name, model in [('Uncalibrated RF', base_model),
                         ('Platt Scaling (Sigmoid)', calibrated_sigmoid),
                         ('Isotonic Regression', calibrated_isotonic)]:
        cv_results = run_time_series_cv(df, features, model, name)

        avg_acc = np.mean([r['accuracy'] for r in cv_results])
        avg_brier = np.mean([r['brier_score'] for r in cv_results])
        avg_hc_acc = np.mean([r['high_conf_acc'] for r in cv_results])

        results[name] = {'accuracy': avg_acc, 'brier': avg_brier, 'high_conf_acc': avg_hc_acc}
        print(f"\n   {name}:")
        print(f"   - Accuracy:           {avg_acc:.2%}")
        print(f"   - Brier Score:        {avg_brier:.4f} (lower is better)")
        print(f"   - High Conf Accuracy: {avg_hc_acc:.2%}")

    return results


def kelly_criterion_analysis(df, features):
    """Analyze Kelly Criterion bet sizing"""
    print("\n" + "="*60)
    print("EXPERIMENT 6: KELLY CRITERION SIMULATION")
    print("="*60)

    # Prepare data
    df_clean = df.dropna(subset=features + ['ats_win']).copy()

    # Train/test split (80/20, respecting time)
    split_idx = int(len(df_clean) * 0.8)
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]

    X_train = train[features].astype(float)
    y_train = train['ats_win'].astype(int)
    X_test = test[features].astype(float)
    y_test = test['ats_win'].astype(int)

    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    # Simulate betting strategies
    ODDS = -110  # Standard -110 vig

    def implied_prob_from_odds(odds):
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    implied = implied_prob_from_odds(ODDS)

    results = {
        'flat_1u': {'bankroll': 100, 'bets': 0, 'wins': 0},
        'kelly_full': {'bankroll': 100, 'bets': 0, 'wins': 0},
        'kelly_quarter': {'bankroll': 100, 'bets': 0, 'wins': 0},
        'threshold_53': {'bankroll': 100, 'bets': 0, 'wins': 0}
    }

    for i, (prob, actual) in enumerate(zip(probs, y_test)):
        edge = prob - implied

        # Strategy 1: Flat 1U on everything >50%
        if prob > 0.5:
            results['flat_1u']['bets'] += 1
            if actual == 1:
                results['flat_1u']['bankroll'] += 1
                results['flat_1u']['wins'] += 1
            else:
                results['flat_1u']['bankroll'] -= 1.1

        # Strategy 2: Full Kelly (risky)
        if edge > 0:
            kelly = edge / (100 / abs(ODDS))  # Simplified Kelly
            bet_size = min(kelly * results['kelly_full']['bankroll'], results['kelly_full']['bankroll'] * 0.25)
            results['kelly_full']['bets'] += 1
            if actual == 1:
                results['kelly_full']['bankroll'] += bet_size
                results['kelly_full']['wins'] += 1
            else:
                results['kelly_full']['bankroll'] -= bet_size * 1.1

        # Strategy 3: Quarter Kelly (conservative)
        if edge > 0:
            kelly = (edge / (100 / abs(ODDS))) * 0.25
            bet_size = min(kelly * results['kelly_quarter']['bankroll'], results['kelly_quarter']['bankroll'] * 0.1)
            results['kelly_quarter']['bets'] += 1
            if actual == 1:
                results['kelly_quarter']['bankroll'] += bet_size
                results['kelly_quarter']['wins'] += 1
            else:
                results['kelly_quarter']['bankroll'] -= bet_size * 1.1

        # Strategy 4: Threshold 53% (current production)
        if prob > 0.53 or prob < 0.47:
            results['threshold_53']['bets'] += 1
            if (prob > 0.53 and actual == 1) or (prob < 0.47 and actual == 0):
                results['threshold_53']['bankroll'] += 1
                results['threshold_53']['wins'] += 1
            else:
                results['threshold_53']['bankroll'] -= 1.1

    print(f"\n   Simulation on {len(y_test)} test games:\n")

    for strat, data in results.items():
        roi = ((data['bankroll'] - 100) / max(data['bets'] * 1.05, 1)) * 100
        win_rate = data['wins'] / max(data['bets'], 1)
        print(f"   {strat}:")
        print(f"      Bets: {data['bets']}, Wins: {data['wins']} ({win_rate:.1%})")
        print(f"      Final Bankroll: {data['bankroll']:.2f}U (ROI: {roi:+.1f}%)\n")

    return results


def main():
    print("\n" + "="*60)
    print("   CBB MODEL IMPROVEMENT EXPERIMENTS")
    print("   Phase 1: Quick Wins Testing")
    print("="*60)

    # Load data
    df = load_and_prep_data()

    # Run experiments
    all_results = {}

    # 1. Baseline
    all_results['baseline'] = experiment_baseline(df)

    # 2. New Features
    feature_results, df_enhanced = experiment_new_features(df.copy())
    all_results['features'] = feature_results

    # Determine best feature set
    best_feature_set = max(feature_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n   Best Feature Set: {best_feature_set[0]} ({best_feature_set[1]['accuracy']:.2%})")

    # Use enhanced features for remaining experiments
    enhanced_features = BASELINE_FEATURES + ['rest_differential']
    available_features = [f for f in enhanced_features if f in df_enhanced.columns]

    # 3. Model Comparison
    all_results['models'] = experiment_models(df_enhanced, available_features)

    # 4. Ensemble
    all_results['ensemble'] = experiment_ensemble(df_enhanced, available_features)

    # 5. Calibration
    all_results['calibration'] = experiment_calibration(df_enhanced, available_features)

    # 6. Kelly Criterion
    all_results['kelly'] = kelly_criterion_analysis(df_enhanced, available_features)

    # Final Summary
    print("\n" + "="*60)
    print("   FINAL SUMMARY")
    print("="*60)

    print("\n   Key Findings:")
    print("   " + "-"*50)

    baseline_acc = all_results['baseline']['baseline']['accuracy']
    print(f"   1. Baseline Accuracy: {baseline_acc:.2%}")

    best_model = max(all_results['models'].items(), key=lambda x: x[1]['accuracy'])
    print(f"   2. Best Model: {best_model[0]} ({best_model[1]['accuracy']:.2%})")

    improvement = best_model[1]['accuracy'] - baseline_acc
    print(f"   3. Improvement over baseline: {improvement:+.2%}")

    print("\n   Recommendations:")
    print("   " + "-"*50)
    if improvement > 0.005:
        print(f"   - Consider switching to {best_model[0]}")
    if 'rest_differential' in available_features:
        print("   - rest_differential feature adds value")
    print("   - Use probability calibration for better confidence estimates")
    print("   - Consider fractional Kelly for bet sizing")


if __name__ == "__main__":
    main()
