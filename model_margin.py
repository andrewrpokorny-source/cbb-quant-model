"""
Margin Prediction Model

Instead of predicting ATS win (binary), this model predicts the actual
point margin. This keeps predictions independent of Vegas spreads.
Used by the line shopping module for monotonic spread probability curves.

Flow:
1. Model predicts: "Home team wins by X points" based on team quality
2. Compare predicted margin to Vegas spread
3. Edge = predicted_margin + spread (positive = bet home, negative = bet away)
   (spread is negative when home is favored, so adding it shrinks the edge)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")
MODEL_FILE = os.path.join(BASE_DIR, "cbb_margin_model.pkl")

# Fallback when sigma is missing from an old-format pkl
DEFAULT_SIGMA = 9.0  # Approximate RMSE for a reasonable margin model

# Features for margin prediction - NO SPREAD (that's what we're trying to beat)
FEATURES = [
    # Game context
    'is_home',
    'rest_days',
    # Team quality differentials
    'diff_eFG',           # Effective FG% differential
    'diff_Rebound',       # Rebounding differential
    'diff_TO',            # Turnover differential
    # Home team recent form
    'momentum_gap',       # Recent vs season performance
    'prev_roll5_margin',  # Recent scoring margin
    'prev_volatility',    # Consistency
    'prev_blowout_rate',  # Dominance indicator
    # Opponent quality
    'opp_win_pct',        # Opponent's win percentage
    'opp_season_off_rating',  # Opponent's offensive rating
    # Home team season stats
    'prev_win_pct',       # Home team's win percentage
    'prev_games_played',  # Sample size / experience
]


def train_and_evaluate():
    print("--- MARGIN PREDICTION MODEL (Experimental) ---")
    print("    Predicts point differential WITHOUT using spread as input\n")

    if not os.path.exists(DATA_FILE):
        print("No processed data found. Run features.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} games")

    # 2. Define target - actual margin (home team perspective)
    target = 'margin'

    # 3. Validate columns
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        print(f"Missing features: {missing_cols}")
        return

    if target not in df.columns:
        print(f"Missing target column: {target}")
        return

    # 4. Clean & Prep
    df_model = df.dropna(subset=FEATURES + [target, 'spread', 'ats_win']).copy()

    X = df_model[FEATURES].astype(float)
    y = df_model[target].astype(float)

    print(f"Training on {len(X)} clean games")
    print(f"Features: {len(FEATURES)} (spread NOT included)\n")

    # 5. Split (time-series aware - no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Keep test set metadata for ATS evaluation
    test_indices = X_test.index
    test_spreads = df_model.loc[test_indices, 'spread'].values
    test_ats_win = df_model.loc[test_indices, 'ats_win'].values

    # 6. Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 7. Evaluate - Regression metrics
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("=== REGRESSION METRICS ===")
    print(f"MAE:  {mae:.2f} points")
    print(f"RMSE: {rmse:.2f} points")
    print(f"R2:   {r2:.3f}")

    # 8. Evaluate - ATS prediction accuracy
    # Home covers when predicted_margin + spread > 0
    # (matches features.py: ats_win = team_score + spread > opp_score)
    predicted_home_covers = (preds + test_spreads) > 0
    actual_home_covers = test_ats_win == 1

    ats_correct = (predicted_home_covers == actual_home_covers).sum()
    ats_total = len(test_ats_win)
    ats_accuracy = ats_correct / ats_total

    print(f"\n=== ATS PREDICTION ===")
    print(f"ATS Accuracy: {ats_accuracy:.1%} ({ats_correct}/{ats_total})")

    # 9. Edge-based evaluation (only bet when model disagrees with spread)
    edge = preds + test_spreads  # Positive = model likes home more than Vegas

    # Different edge thresholds
    for threshold in [1.0, 2.0, 3.0, 5.0]:
        # Bets where model sees edge
        strong_home = edge > threshold
        strong_away = edge < -threshold

        home_bets = strong_home.sum()
        away_bets = strong_away.sum()
        home_wins = 0
        away_wins = 0

        if home_bets > 0:
            home_wins = (strong_home & actual_home_covers).sum()
            home_acc = home_wins / home_bets
        else:
            home_acc = 0

        if away_bets > 0:
            away_wins = (strong_away & ~actual_home_covers).sum()
            away_acc = away_wins / away_bets
        else:
            away_acc = 0

        total_bets = home_bets + away_bets
        total_wins = home_wins + away_wins
        total_acc = total_wins / total_bets if total_bets > 0 else 0

        print(f"\nEdge >= {threshold} points:")
        print(f"  Home bets: {home_bets} ({home_acc:.1%} win rate)")
        print(f"  Away bets: {away_bets} ({away_acc:.1%} win rate)")
        print(f"  Total: {total_bets} bets, {total_acc:.1%} accuracy")

    # 10. Feature importance
    print(f"\n=== TOP FEATURES ===")
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(7).iterrows():
        print(f"  {row['feature']:<25}: {row['importance']:.3f}")

    # 11. Compare to current classifier
    print(f"\n=== COMPARISON TO CURRENT MODEL ===")
    try:
        current_model = joblib.load(os.path.join(BASE_DIR, "cbb_model_v1.pkl"))

        # Current model uses spread as a feature
        current_features = list(current_model.feature_names_in_)
        print(f"Current model features: {len(current_features)}")
        print(f"  Uses 'spread' as input: {'spread' in current_features}")

        # Get current model's predictions on test set
        X_test_current = df_model.loc[test_indices, current_features].astype(float)
        current_probs = current_model.predict_proba(X_test_current)[:, 1]
        current_preds = current_probs > 0.5

        current_correct = (current_preds == actual_home_covers).sum()
        current_accuracy = current_correct / ats_total

        print(f"\nCurrent model ATS accuracy: {current_accuracy:.1%}")
        print(f"Margin model ATS accuracy:  {ats_accuracy:.1%}")
        print(f"Difference: {(ats_accuracy - current_accuracy)*100:+.1f}%")

    except Exception as e:
        print(f"Could not load current model for comparison: {e}")

    # 12. Compute sigma (std of training residuals) for line shopping norm.cdf
    train_preds = model.predict(X_train)
    residuals = y_train.values - train_preds
    sigma = float(np.std(residuals))
    print(f"\n=== RESIDUAL SIGMA ===")
    print(f"Sigma (std of training residuals): {sigma:.2f} points")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Invalid sigma={sigma} from training residuals. Check training data.")

    # 13. Save model + sigma together
    joblib.dump({'model': model, 'sigma': sigma}, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")

    return model


def load_margin_model(path=MODEL_FILE):
    """
    Load margin model + sigma from pkl file.

    Handles both old format (raw model) and new format ({'model': ..., 'sigma': ...}).
    Returns (model, sigma). Defaults sigma to DEFAULT_SIGMA for old-format files.
    """
    data = joblib.load(path)
    if isinstance(data, dict) and 'model' in data:
        sigma = data.get('sigma')
        if sigma is None:
            print(f"WARNING: Margin model missing 'sigma', defaulting to {DEFAULT_SIGMA}")
            sigma = DEFAULT_SIGMA
        if not isinstance(sigma, (int, float)) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Invalid sigma={sigma} in margin model. Re-run model_margin.py.")
        return data['model'], float(sigma)
    # Old format: raw model, no sigma saved
    print(f"WARNING: Old-format margin model, defaulting sigma to {DEFAULT_SIGMA}")
    return data, DEFAULT_SIGMA


def predict_margin(model, features_dict):
    """
    Predict point margin for a game.

    Args:
        model: Trained margin prediction model
        features_dict: Dict with feature values (no spread needed)

    Returns:
        Predicted margin (positive = home team wins by X)
    """
    input_df = pd.DataFrame([features_dict])

    # Ensure all features present
    missing = [col for col in FEATURES if col not in input_df.columns]
    if missing:
        print(f"WARNING: predict_margin missing features (defaulting to 0.0): {missing}")
    for col in missing:
        input_df[col] = 0.0

    input_df = input_df[FEATURES].astype(float)
    return model.predict(input_df)[0]


def calculate_edge(predicted_margin, vegas_spread):
    """
    Calculate betting edge.

    Args:
        predicted_margin: Model's predicted margin (home perspective)
        vegas_spread: Vegas spread (negative = home favored)

    Returns:
        edge: predicted_margin + vegas_spread
              Positive = value on home team
              Negative = value on away team

    Example:
        Model predicts home wins by 5, spread is -7 (home favored by 7):
        edge = 5 + (-7) = -2 -> bet away (market overvalues home)
    """
    return predicted_margin + vegas_spread


if __name__ == "__main__":
    train_and_evaluate()
