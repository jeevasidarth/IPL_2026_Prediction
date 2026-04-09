import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import os
import json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLET_DATA = os.path.join(BASE_DIR, 'training_matchup_triplets_bat.csv')
MATCH_DATA = os.path.join(BASE_DIR, 'ipl_match_features_training.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models/')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_production_models():
    print("--- STARTING FINAL PRODUCTION TRAINING (2008-2025) ---")
    
    # 1. LOAD DATA
    df_triplet = pd.read_csv(TRIPLET_DATA)
    df_match = pd.read_csv(MATCH_DATA)
    
    # 2. STAGE 1: PERFORMANCE (CatBoost)
    # Target: runs in a matchup
    print("Training Stage 1 Batting: Matchup Performance (CatBoost)...")
    features_bat = ['batter', 'bowler', 'innings', 'balls', 'bat_global_avg', 'bat_global_sr', 'bowl_global_econ']
    cat_features = [0, 1] 
    
    bat_model = CatBoostRegressor(iterations=1200, learning_rate=0.03, depth=6, verbose=0)
    bat_model.fit(df_triplet[features_bat], df_triplet['runs'], cat_features=cat_features)
    bat_model.save_model(os.path.join(MODEL_DIR, 'ensemble_bat_catboost.cbm'))
    
    print("Training Stage 1 Bowling: Economy Performance (CatBoost)...")
    # Economy features
    bowl_model = CatBoostRegressor(iterations=1200, learning_rate=0.03, depth=6, verbose=0)
    bowl_model.fit(df_triplet[features_bat], df_triplet['economy'], cat_features=cat_features)
    bowl_model.save_model(os.path.join(MODEL_DIR, 'ensemble_bowl_catboost.cbm'))
    
    print("DONE: Stage 1 Production models saved.")

    # 3. STAGE 2: WIN CLASSIFIER (XGBoost)
    print("Training Stage 2 Production: Win probability Meta-Model (XGBoost)...")
    
    features_win = [
        'is_afternoon', 'temp_i1', 'hum_i1', 'dew_factor',
        'team1_h2h_rp_ball', 'team2_h2h_rp_ball',
        'team1_avg_bat_form', 'team2_avg_bat_form'
    ]
    
    # Full data including 2025
    X_meta = df_match[features_win]
    y_meta = df_match['winner_is_team1']
    
    meta_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=7)
    meta_model.fit(X_meta, y_meta)
    
    # Save XGBoost
    joblib.dump(meta_model, os.path.join(MODEL_DIR, 'ensemble_win_meta_model.pkl'))
    print("DONE: Stage 2 Production model saved.")
    
    print("\n--- ✅ PRODUCTION TRAINING SUCCESSFUL (IPL 2026 READY) ✅ ---")

if __name__ == "__main__":
    train_production_models()
