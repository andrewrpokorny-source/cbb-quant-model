# CBB Spread Model Improvement Plan

## Current State Summary
- **Model:** Random Forest Classifier (200 trees, max depth 5)
- **Features:** 8 features (is_home, spread, rest_days, diff_eFG, diff_Rebound, diff_TO, momentum_gap, roll5_cover_margin)
- **Performance:** ~51-55% accuracy (modest edge over 50/50)
- **Data:** 23,925 processed game records from ESPN API

---

## Improvement Categories

### 1. Feature Engineering Enhancements

#### A. Defensive Efficiency Metrics
- [ ] **Defensive Rating (DRtg):** Points allowed per 100 possessions
- [ ] **Opponent eFG% Allowed:** How well team defends shots
- [ ] **Defensive Rebound Rate:** Percentage of opponent misses secured
- [ ] **Steal Rate / Block Rate:** Disruption metrics
- [ ] **diff_DRtg:** Defensive rating differential between teams

#### B. Pace & Tempo Features
- [ ] **Possessions per game:** Team's pace preference
- [ ] **Pace mismatch:** Fast team vs slow team differential
- [ ] **Points per possession variance:** Consistency metric

#### C. Shooting Profile Features
- [ ] **3-Point Attempt Rate:** Percentage of shots from 3
- [ ] **3-Point dependency mismatch:** High 3PA team vs good 3P defense
- [ ] **Free throw rate:** FTA/FGA ratio (drawing fouls)
- [ ] **And-1 / FT reliance:** Teams that live at the line

#### D. Schedule & Situational Features
- [ ] **Opponent rest days:** Not just own rest, but matchup rest differential
- [ ] **Conference vs non-conference flag:** Different dynamics
- [ ] **Travel distance:** Approximate miles traveled for away games
- [ ] **Back-to-back flag:** Specifically flag B2B situations
- [ ] **Days since last loss:** "Revenge" or frustration factor
- [ ] **Season progress:** Early season (volatile) vs late season (stable)

#### E. Advanced Matchup Features
- [ ] **Style clash score:** Rebounding team vs poor rebounder
- [ ] **Chaos factor:** High variance team (TO prone) vs consistent team
- [ ] **Glass eating differential:** ORB rate vs opponent DRB rate
- [ ] **Tempo control:** Which team dictates pace

#### F. Market & Line Features
- [ ] **Line movement:** Opening line vs current line
- [ ] **Public betting percentage:** Fade the public signal
- [ ] **Sharp money indicator:** Reverse line movement

---

### 2. Model Architecture Improvements

#### A. Alternative Algorithms
- [ ] **XGBoost:** Already tested in audit.py, shows competitive results
- [ ] **LightGBM:** Faster training, handles categorical features natively
- [ ] **CatBoost:** Good with categorical data, less overfitting
- [ ] **Logistic Regression:** Baseline comparison, interpretable coefficients

#### B. Ensemble Methods
- [ ] **Stacking:** Combine RF + XGBoost + LogReg predictions
- [ ] **Voting Classifier:** Majority vote from multiple models
- [ ] **Blending:** Weighted average of model probabilities

#### C. Neural Network Approaches
- [ ] **Simple MLP:** Dense layers for non-linear patterns
- [ ] **LSTM/GRU:** Sequence model for team performance trajectories
- [ ] **Attention mechanism:** Weight recent games differently

#### D. Hyperparameter Optimization
- [ ] **Grid Search / Random Search:** Systematic tuning
- [ ] **Bayesian Optimization:** More efficient parameter search
- [ ] **Cross-validation:** k-fold with time-series awareness

---

### 3. Data Source Enhancements

#### A. External Data Integration
- [ ] **KenPom/Barttorvik ratings:** Adjusted efficiency metrics
- [ ] **Player-level stats:** Leading scorer PPG, key player efficiency
- [ ] **Injury reports:** Manual or scraped injury data
- [ ] **Weather data:** For neutral site games (dome vs outdoor practice)

#### B. Historical Context
- [ ] **Head-to-head history:** Last 3 meetings ATS record
- [ ] **Venue-specific performance:** Team's record at specific arenas
- [ ] **Conference tournament history:** March performance patterns

#### C. Real-time Data
- [ ] **Live odds from multiple books:** Find best line
- [ ] **Consensus lines:** Average across books
- [ ] **Steam moves:** Sudden line shifts

---

### 4. Training & Validation Improvements

#### A. Data Splitting Strategy
- [ ] **Walk-forward validation:** More robust than single split
- [ ] **Season holdout:** Train on 2024-25, validate on 2025-26
- [ ] **Conference-based splits:** Test generalization across conferences

#### B. Feature Selection
- [ ] **Recursive Feature Elimination (RFE):** Remove low-impact features
- [ ] **Permutation importance:** Measure true feature contribution
- [ ] **SHAP values:** Interpretable feature importance
- [ ] **Correlation analysis:** Remove redundant features

#### C. Probability Calibration
- [ ] **Platt scaling:** Calibrate RF probabilities
- [ ] **Isotonic regression:** Non-parametric calibration
- [ ] **Reliability diagrams:** Visualize calibration quality

#### D. Class Imbalance Handling
- [ ] **Check ATS win distribution:** Ensure balanced classes
- [ ] **SMOTE or undersampling:** If imbalanced
- [ ] **Class weights:** Adjust for any imbalance

---

### 5. Betting Strategy Improvements

#### A. Bet Sizing
- [ ] **Kelly Criterion:** Optimal bet size based on edge and bankroll
- [ ] **Fractional Kelly:** 1/4 or 1/2 Kelly for reduced variance
- [ ] **Unit sizing tiers:** 1U, 2U, 3U based on confidence

#### B. Confidence Thresholds
- [ ] **Dynamic threshold:** Adjust based on recent performance
- [ ] **Tiered confidence:** Different strategies for 53-55%, 55-60%, 60%+
- [ ] **Expected value calculation:** Only bet when EV > threshold

#### C. Portfolio Approach
- [ ] **Correlation tracking:** Avoid correlated bets on same day
- [ ] **Max exposure per conference:** Diversification
- [ ] **Bankroll management:** Track drawdowns, adjust sizing

#### D. Performance Analytics
- [ ] **ROI by feature:** Which feature values are most profitable
- [ ] **Time-based analysis:** Performance by day of week, month
- [ ] **Closing line value (CLV):** Track if beating closing lines

---

### 6. Infrastructure Improvements

#### A. Automation
- [ ] **Scheduled cron jobs:** Fully automated daily pipeline
- [ ] **Error alerting:** Slack/email notifications on failures
- [ ] **Data validation:** Automated checks for data quality

#### B. Monitoring Dashboard
- [ ] **Real-time performance tracking:** Streamlit enhancements
- [ ] **Model drift detection:** Alert when accuracy drops
- [ ] **Feature distribution monitoring:** Detect data shifts

#### C. Version Control
- [ ] **Model versioning:** Track model iterations with MLflow or similar
- [ ] **Data versioning:** DVC for dataset tracking
- [ ] **Experiment tracking:** Log all model experiments

---

## Priority Implementation Order

### Phase 1: Quick Wins (High Impact, Low Effort)
1. Add defensive efficiency metrics (DRtg, opponent eFG% allowed)
2. Add opponent rest days differential
3. Implement XGBoost as alternative model
4. Add probability calibration
5. Implement Kelly Criterion for bet sizing

### Phase 2: Medium-Term Improvements
6. Add pace/tempo features
7. Build ensemble model (RF + XGBoost + LogReg)
8. Integrate KenPom/Barttorvik data
9. Implement walk-forward validation
10. Add line movement tracking

### Phase 3: Advanced Enhancements
11. Build LSTM model for team trajectories
12. Add player-level features
13. Implement full MLflow tracking
14. Build advanced matchup analysis
15. Add closing line value tracking

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| ATS Accuracy | 51-55% | 55-58% |
| ROI (at -110) | ~0-5% | 5-10% |
| Calibration (Brier Score) | Unknown | < 0.24 |
| Actionable Bets/Day | ~10-15 | 15-25 |
| Closing Line Value | Unknown | +1-2% |

---

## Next Steps

1. **Choose Phase 1 items to implement first**
2. **Set up A/B testing framework** to compare improvements
3. **Establish baseline metrics** before making changes
4. **Document each improvement** with before/after results
