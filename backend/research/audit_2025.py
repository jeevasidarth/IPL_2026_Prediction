import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLET_DATA = os.path.join(BASE_DIR, 'training_matchup_triplets_bat.csv')
MATCH_DATA = os.path.join(BASE_DIR, 'ipl_match_features_training.csv')

def run_audit():
    print("--- STARTING 2025 MODEL AUDIT ---")
    
    # 1. LOAD DATA
    df_triplet = pd.read_csv(TRIPLET_DATA)
    df_match = pd.read_csv(MATCH_DATA)
    
    # 2. STAGE 1 AUDIT (Player Level)
    print("\n--- Phase 1: Player Performance Audit (2025 Hold-out) ---")
    train_triplet = df_triplet[df_triplet['year'] < 2025].copy()
    test_triplet = df_triplet[df_triplet['year'] == 2025].copy()
    
    # Features for "Expert Matchup"
    # CatBoost will handle batter/bowler names directly!
    features_bat = ['batter', 'bowler', 'innings', 'balls', 'bat_global_avg', 'bat_global_sr', 'bowl_global_econ']
    cat_features = [0, 1] 
    
    print(f"Training Stage 1 (CatBoost) on {len(train_triplet)} interactions...")
    bat_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=0)
    bat_model.fit(train_triplet[features_bat], train_triplet['runs'], cat_features=cat_features)
    
    y_bat_pred = bat_model.predict(test_triplet[features_bat])
    mae_bat = mean_absolute_error(test_triplet['runs'], y_bat_pred)
    print(f"DONE: Batter Score MAE (2025): {mae_bat:.2f} runs per matchup")
    
    # 3. STAGE 2 AUDIT (Win Prob)
    print("\n--- Phase 2: Win Probability Audit (2025 Hold-out) ---")
    train_match = df_match[df_match['year'] < 2025]
    test_match = df_match[df_match['year'] == 2025]
    
    features_win = [
        'is_afternoon', 'temp_i1', 'hum_i1', 'dew_factor',
        'team1_h2h_rp_ball', 'team2_h2h_rp_ball',
        'team1_avg_bat_form', 'team2_avg_bat_form'
    ]
    
    print(f"Training Stage 2 on {len(train_match)} matches...")
    win_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6)
    win_model.fit(train_match[features_win], train_match['winner_is_team1'])
    
    y_win_pred = win_model.predict(test_match[features_win])
    y_win_proba = win_model.predict_proba(test_match[features_win])[:, 1]
    
    acc = accuracy_score(test_match['winner_is_team1'], y_win_pred)
    auc = roc_auc_score(test_match['winner_is_team1'], y_win_proba)
    
    print(f"DONE: Winner Accuracy (2025): {acc*100:.1f}%")
    print(f"DONE: ROC-AUC (2025): {auc:.3f}")

    # 4. EXPORT AUDIT RESULTS
    audit_results = {
        'player_mae': float(mae_bat),
        'winner_acc': float(acc),
        'winner_auc': float(auc),
        'n_matches_2025': int(len(test_match))
    }
    
    with open('audit_2025_results.json', 'w') as f:
        import json
        json.dump(audit_results, f, indent=4)
    
    print("\nAudit Complete. Results saved to audit_2025_results.json.")

if __name__ == "__main__":
    run_audit()
