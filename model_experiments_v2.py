"""
Model Improvement Experiments V2
Additional feature exploration
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")

BASELINE_FEATURES = [
    'is_home', 'spread', 'rest_days', 'diff_eFG',
    'diff_Rebound', 'diff_TO', 'momentum_gap', 'roll5_cover_margin'
]


def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['team', 'date']).reset_index(drop=True)
    print(f"   -> Loaded {len(df)} rows")
    return df


def add_all_experimental_features(df):
    """Add all experimental features at once"""
    print("\nEngineering experimental features...")

    # --- 1. DEFENSIVE RATING (from v1 - confirmed useful) ---
    print("   -> Defensive rating...")
    df['def_rating'] = 100 * (df['opp_score'] / df['poss'].clip(lower=1))
    df['season_def_rating'] = df.groupby('team')['def_rating'].expanding().mean().reset_index(level=0, drop=True)
    df['prev_season_def_rating'] = df.groupby('team')['season_def_rating'].shift(1)

    if 'opp_season_off_rating' in df.columns:
        df['diff_DRtg'] = df['opp_season_off_rating'] - df['prev_season_def_rating']
    else:
        df['diff_DRtg'] = 0

    # --- 2. OPPONENT REST DIFFERENTIAL ---
    print("   -> Rest differential...")
    rest_lookup = df[['date', 'team', 'rest_days']].drop_duplicates()
    rest_lookup = rest_lookup.rename(columns={'team': 'opp_name', 'rest_days': 'opp_rest_days'})
    df = pd.merge(df, rest_lookup, left_on=['date', 'opponent'], right_on=['date', 'opp_name'], how='left')
    df['rest_differential'] = df['rest_days'] - df['opp_rest_days'].fillna(df['rest_days'])
    df = df.drop(columns=['opp_name'], errors='ignore')

    # --- 3. WIN/LOSS STREAK ---
    print("   -> Win/loss streaks...")
    df['game_win'] = (df['team_score'] > df['opp_score']).astype(int)
    df['ats_result'] = (df['team_score'] + df['spread'] > df['opp_score']).astype(int)

    # Straight up streak
    def calc_streak(series):
        streak = []
        current = 0
        for val in series:
            if val == 1:
                current = max(1, current + 1)
            else:
                current = min(-1, current - 1)
            streak.append(current)
        return streak

    df['win_streak'] = df.groupby('team')['game_win'].transform(lambda x: calc_streak(x.values))
    df['prev_win_streak'] = df.groupby('team')['win_streak'].shift(1).fillna(0)

    # ATS streak
    df['ats_streak'] = df.groupby('team')['ats_result'].transform(lambda x: calc_streak(x.values))
    df['prev_ats_streak'] = df.groupby('team')['ats_streak'].shift(1).fillna(0)

    # --- 4. SCORE VOLATILITY ---
    print("   -> Score volatility...")
    df['roll5_score_std'] = df.groupby('team')['team_score'].rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
    df['prev_volatility'] = df.groupby('team')['roll5_score_std'].shift(1).fillna(10)

    # Opponent volatility lookup
    vol_lookup = df[['date', 'team', 'prev_volatility']].drop_duplicates()
    vol_lookup = vol_lookup.rename(columns={'team': 'opp_name', 'prev_volatility': 'opp_volatility'})
    df = pd.merge(df, vol_lookup, left_on=['date', 'opponent'], right_on=['date', 'opp_name'], how='left')
    df['volatility_diff'] = df['prev_volatility'] - df['opp_volatility'].fillna(df['prev_volatility'])
    df = df.drop(columns=['opp_name'], errors='ignore')

    # --- 5. GAMES PLAYED (sample size) ---
    print("   -> Games played...")
    df['games_played'] = df.groupby('team').cumcount()
    df['prev_games_played'] = df.groupby('team')['games_played'].shift(1).fillna(0)

    # Early season flag (first 5 games)
    df['early_season'] = (df['prev_games_played'] < 5).astype(int)

    # --- 6. SPREAD INTERACTIONS ---
    print("   -> Spread interactions...")
    df['home_favorite'] = ((df['is_home'] == 1) & (df['spread'] < 0)).astype(int)
    df['home_underdog'] = ((df['is_home'] == 1) & (df['spread'] > 0)).astype(int)
    df['road_favorite'] = ((df['is_home'] == 0) & (df['spread'] < 0)).astype(int)
    df['road_underdog'] = ((df['is_home'] == 0) & (df['spread'] > 0)).astype(int)

    # Spread magnitude
    df['spread_abs'] = df['spread'].abs()
    df['big_favorite'] = (df['spread'] < -10).astype(int)
    df['big_underdog'] = (df['spread'] > 10).astype(int)

    # --- 7. BLOWOUT TENDENCY ---
    print("   -> Blowout/close game tendency...")
    df['margin'] = df['team_score'] - df['opp_score']
    df['cover_margin'] = df['team_score'] + df['spread'] - df['opp_score']

    # Rolling average margin and cover margin
    df['roll5_margin'] = df.groupby('team')['margin'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['prev_roll5_margin'] = df.groupby('team')['roll5_margin'].shift(1).fillna(0)

    df['roll5_cover'] = df.groupby('team')['cover_margin'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['prev_roll5_cover'] = df.groupby('team')['roll5_cover'].shift(1).fillna(0)

    # Blowout rate (won/lost by 15+)
    df['blowout_win'] = (df['margin'] > 15).astype(int)
    df['blowout_loss'] = (df['margin'] < -15).astype(int)
    df['roll5_blowout_rate'] = df.groupby('team')['blowout_win'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['prev_blowout_rate'] = df.groupby('team')['roll5_blowout_rate'].shift(1).fillna(0)

    # --- 8. SCORING TREND ---
    print("   -> Scoring trends...")
    df['season_ppg'] = df.groupby('team')['team_score'].expanding().mean().reset_index(level=0, drop=True)
    df['prev_season_ppg'] = df.groupby('team')['season_ppg'].shift(1)
    df['roll3_ppg'] = df.groupby('team')['team_score'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df['prev_roll3_ppg'] = df.groupby('team')['roll3_ppg'].shift(1)

    # Scoring momentum (recent vs season)
    df['scoring_momentum'] = df['prev_roll3_ppg'] - df['prev_season_ppg']

    # --- 9. CONFERENCE GAMES (approximate) ---
    print("   -> Season timing...")
    # Use date to approximate conference play (after Dec 15 typically)
    df['month'] = df['date'].dt.month
    df['conference_szn'] = ((df['month'] >= 1) & (df['month'] <= 3)).astype(int)

    # --- 10. OPPONENT QUALITY PROXY ---
    print("   -> Opponent quality proxy...")
    # Use opponent's season win rate as quality proxy
    opp_stats = df.groupby(['team', 'date']).agg({
        'game_win': 'first',
        'games_played': 'first'
    }).reset_index()

    opp_stats['season_wins'] = opp_stats.groupby('team')['game_win'].cumsum()
    opp_stats['win_pct'] = opp_stats['season_wins'] / (opp_stats['games_played'] + 1).clip(lower=1)
    opp_stats['prev_win_pct'] = opp_stats.groupby('team')['win_pct'].shift(1).fillna(0.5)

    opp_lookup = opp_stats[['date', 'team', 'prev_win_pct']].rename(
        columns={'team': 'opp_name', 'prev_win_pct': 'opp_win_pct'}
    )
    df = pd.merge(df, opp_lookup, left_on=['date', 'opponent'], right_on=['date', 'opp_name'], how='left')
    df['opp_win_pct'] = df['opp_win_pct'].fillna(0.5)
    df = df.drop(columns=['opp_name'], errors='ignore')

    print("   -> Feature engineering complete!")
    return df


def run_cv(df, features, model_name="Model"):
    """Run time series CV and return average accuracy"""
    available = [f for f in features if f in df.columns]
    if len(available) < len(features):
        missing = set(features) - set(available)
        # print(f"      Missing: {missing}")

    df_clean = df.dropna(subset=available + ['ats_win']).copy()
    X = df_clean[available].astype(float)
    y = df_clean['ats_win'].astype(int)

    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    high_conf_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        scores.append(accuracy_score(y_test, preds))

        # High confidence
        high_conf = (probs > 0.53) | (probs < 0.47)
        if high_conf.sum() > 0:
            high_conf_scores.append(accuracy_score(y_test[high_conf], preds[high_conf]))

    return np.mean(scores), np.mean(high_conf_scores) if high_conf_scores else 0


def test_individual_features(df):
    """Test each new feature individually"""
    print("\n" + "="*60)
    print("INDIVIDUAL FEATURE IMPACT")
    print("="*60)

    # Baseline
    base_acc, base_hc = run_cv(df, BASELINE_FEATURES)
    print(f"\n   Baseline: {base_acc:.2%} (HC: {base_hc:.2%})")
    print("-" * 50)

    # New features to test
    new_features = {
        'diff_DRtg': 'Defensive Rating Diff',
        'rest_differential': 'Rest Days Differential',
        'prev_win_streak': 'Win Streak',
        'prev_ats_streak': 'ATS Streak',
        'prev_volatility': 'Score Volatility',
        'volatility_diff': 'Volatility Differential',
        'early_season': 'Early Season Flag',
        'home_favorite': 'Home Favorite',
        'home_underdog': 'Home Underdog',
        'road_favorite': 'Road Favorite',
        'big_favorite': 'Big Favorite (>10)',
        'big_underdog': 'Big Underdog (>10)',
        'spread_abs': 'Spread Magnitude',
        'prev_roll5_margin': 'Recent Margin (5g)',
        'prev_roll5_cover': 'Recent Cover Margin',
        'prev_blowout_rate': 'Blowout Rate',
        'scoring_momentum': 'Scoring Momentum',
        'conference_szn': 'Conference Season',
        'opp_win_pct': 'Opponent Win %',
        'prev_games_played': 'Games Played',
    }

    results = []
    for feat, name in new_features.items():
        if feat in df.columns:
            test_features = BASELINE_FEATURES + [feat]
            acc, hc = run_cv(df, test_features)
            delta = acc - base_acc
            results.append({
                'feature': feat,
                'name': name,
                'accuracy': acc,
                'high_conf': hc,
                'delta': delta
            })
            symbol = "+" if delta > 0 else ""
            print(f"   + {name:<25}: {acc:.2%} ({symbol}{delta:.2%})")

    # Sort by improvement
    results.sort(key=lambda x: x['delta'], reverse=True)

    print("\n" + "="*60)
    print("TOP FEATURES BY IMPROVEMENT")
    print("="*60)
    for r in results[:5]:
        print(f"   {r['name']:<25}: +{r['delta']:.2%}")

    return results


def test_feature_combinations(df, top_features):
    """Test combinations of top features"""
    print("\n" + "="*60)
    print("FEATURE COMBINATIONS")
    print("="*60)

    base_acc, _ = run_cv(df, BASELINE_FEATURES)
    print(f"\n   Baseline: {base_acc:.2%}")

    # Get top 5 improving features
    improving = [r['feature'] for r in top_features if r['delta'] > 0][:5]

    if not improving:
        print("   No improving features found")
        return

    print(f"\n   Testing combinations of: {improving}")
    print("-" * 50)

    # Test incrementally adding features
    current_features = BASELINE_FEATURES.copy()

    for feat in improving:
        current_features = current_features + [feat]
        acc, hc = run_cv(df, current_features)
        delta = acc - base_acc
        print(f"   + {feat:<20}: {acc:.2%} (+{delta:.2%}) | HC: {hc:.2%}")

    # Test all top features together
    all_top = BASELINE_FEATURES + improving
    acc, hc = run_cv(df, all_top)
    delta = acc - base_acc
    print(f"\n   ALL TOP FEATURES:     {acc:.2%} (+{delta:.2%}) | HC: {hc:.2%}")

    return all_top


def test_feature_importance(df, features):
    """Show feature importance for best model"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Gradient Boosting)")
    print("="*60)

    available = [f for f in features if f in df.columns]
    df_clean = df.dropna(subset=available + ['ats_win']).copy()

    X = df_clean[available].astype(float)
    y = df_clean['ats_win'].astype(int)

    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
    )
    model.fit(X, y)

    importance = pd.DataFrame({
        'feature': available,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n   Top 10 Features:")
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"   {row['feature']:<22}: {row['importance']:.3f} {bar}")

    return importance


def main():
    print("\n" + "="*60)
    print("   CBB MODEL EXPERIMENTS V2")
    print("   Extended Feature Analysis")
    print("="*60)

    # Load and prep
    df = load_data()
    df = add_all_experimental_features(df)

    # Test individual features
    individual_results = test_individual_features(df)

    # Test combinations
    best_features = test_feature_combinations(df, individual_results)

    # Feature importance
    if best_features:
        test_feature_importance(df, best_features)

    # Final summary
    print("\n" + "="*60)
    print("   SUMMARY")
    print("="*60)

    improving = [r for r in individual_results if r['delta'] > 0]
    print(f"\n   Features that improve accuracy: {len(improving)}")
    for r in improving:
        print(f"      - {r['name']}: +{r['delta']:.2%}")

    neutral = [r for r in individual_results if abs(r['delta']) < 0.002]
    print(f"\n   Neutral features: {len(neutral)}")

    hurting = [r for r in individual_results if r['delta'] < -0.002]
    print(f"\n   Features that hurt accuracy: {len(hurting)}")
    for r in hurting:
        print(f"      - {r['name']}: {r['delta']:.2%}")


if __name__ == "__main__":
    main()
