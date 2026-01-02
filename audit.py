import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# --- CONFIG ---
DATA_FILE = "cbb_training_data_processed.csv"

def create_matchup_features(df):
    """
    LAYER 1: ADVANCED MATCHUP LOGIC
    """
    # 1. SOS Adjusted Efficiency (The "KenPom" Logic)
    # My Offense vs Your Defense (Not just Your General Quality)
    # Note: 'opp_season_opp_eFG' = The eFG% the opponent ALLOWS (Defense).
    df['off_advantage'] = df['season_team_eFG'] - df['opp_season_opp_eFG']
    
    # My Defense vs Your Offense
    # 'opp_season_team_eFG' = The Opponent's Offense.
    # 'season_opp_eFG' = My Allowed eFG% (Defense).
    df['def_advantage'] = df['season_opp_eFG'] - df['opp_season_team_eFG']
    
    # 2. Style Matchups
    # 3-Point Disparity: If I shoot a ton of 3s and you don't...
    df['diff_3PR'] = df['season_team_3PR'] - df['opp_season_team_3PR']
    
    # Rebounding Mismatch
    df['glass_advantage'] = df['season_team_ORB'] - df['opp_season_team_ORB']
    
    # 3. Volatility Gap (Chaos Factor)
    df['diff_volatility'] = df['roll5_score_volatility'] - df['opp_roll5_score_volatility']
    
    # 4. Momentum
    df['momentum_gap'] = df['roll3_team_eFG'] - df['season_team_eFG']
    
    return df

def audit_models():
    print("--- 🔬 RUNNING MATCHUP AUDIT 🔬 ---")
    
    try:
        df = pd.read_csv(DATA_FILE).dropna()
        df = df.sort_values('date')
    except:
        print("❌ Data not found."); return

    df = create_matchup_features(df)
    
    # The Full Arsenal of Features
    feature_cols = [
        'is_home', 'spread', 'rest_days', 'roll5_cover_margin',
        'off_advantage', 'def_advantage', 
        'glass_advantage', 'diff_3PR', 'diff_volatility',
        'momentum_gap'
    ]
    
    # Filter for what actually exists in data
    final_feats = [c for c in feature_cols if c in df.columns]
    X = df[final_feats]
    y = df['ats_win']
    
    print(f"   -> Auditing {len(final_feats)} Matchup Factors.")

    # 1. MODEL SHOOTOUT
    print("\n--- 🥊 MODEL SHOOTOUT (RF vs XGB) 🥊 ---")
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=4, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=150, learning_rate=0.04, max_depth=3, subsample=0.8, random_state=42)
    
    rf_scores, xgb_scores = [], []
    
    for train, test in tscv.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        
        rf.fit(X_train, y_train)
        rf_scores.append(accuracy_score(y_test, rf.predict(X_test)))
        
        xgb_clf.fit(X_train, y_train)
        xgb_scores.append(accuracy_score(y_test, xgb_clf.predict(X_test)))

    print(f"   Random Forest: {np.mean(rf_scores):.1%}")
    print(f"   XGBoost:       {np.mean(xgb_scores):.1%}")
    
    # Use XGB for the Deep Dive (usually handles interactions better)
    winner = xgb_clf
    winner.fit(X, y)
    df['prob'] = winner.predict_proba(X)[:, 1]
    df['correct'] = ((df['prob'] > 0.5) == df['ats_win']).astype(int)
    
    # 2. MATCHUP ANALYSIS (The New Part)
    print("\n--- 🧩 STYLE MATCHUP ANALYSIS 🧩 ---")
    
    # A. The "Glass Eater" Edge (Rebounding)
    # Does the model perform better when there is a massive rebounding disparity?
    df['rebound_mismatch'] = pd.qcut(df['glass_advantage'], 4, labels=["We get Crushed", "Disadvantage", "Advantage", "We Crush Glass"])
    print("\n1. REBOUNDING MISMATCH (Win Rate)")
    print(df.groupby('rebound_mismatch', observed=False)['correct'].mean().apply(lambda x: f"{x:.1%}"))

    # B. The "Chaos" Edge (3-Point Variance)
    # Do we predict better when teams play a high-variance style?
    df['chaos_factor'] = pd.qcut(df['diff_3PR'], 4, labels=["Low 3pt Reliance", "Standard", "High", "3pt Barrage"])
    print("\n2. 3-POINT RELIANCE (Win Rate)")
    print(df.groupby('chaos_factor', observed=False)['correct'].mean().apply(lambda x: f"{x:.1%}"))
    
    # C. Feature Importance
    print("\n--- 🔑 KEY DRIVERS ---")
    imps = winner.feature_importances_
    indices = np.argsort(imps)[::-1]
    for i in range(min(5, len(imps))):
        print(f"   {final_feats[indices[i]]:<20}: {imps[indices[i]]:.4f}")

if __name__ == "__main__":
    audit_models()