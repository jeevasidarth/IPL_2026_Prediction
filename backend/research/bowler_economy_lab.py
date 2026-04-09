import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
import os
import json
import time

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'training_matchup_triplets_bat.csv')

def run_lab():
    print("--- STARTING BOWLER ECONOMY RESEARCH LAB ---")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # Split: Train (2021-2024), Test (2025)
    train_df = df[df['year'] < 2025].copy()
    test_df = df[df['year'] == 2025].copy()
    
    print(f"Training on: {len(train_df)} rows")
    print(f"Testing on: {len(test_df)} rows (2025 Season)")
    
    # Features
    features = ['batter', 'bowler', 'innings', 'balls', 'bat_global_avg', 'bat_global_sr', 'bowl_global_econ']
    target = 'economy'
    
    # Encoding for non-CatBoost models
    le_bat = LabelEncoder()
    le_bow = LabelEncoder()
    
    # Combine for encoding to handle all names
    all_batters = pd.concat([train_df['batter'], test_df['batter']])
    all_bowlers = pd.concat([train_df['bowler'], test_df['bowler']])
    
    le_bat.fit(all_batters)
    le_bow.fit(all_bowlers)
    
    def prepare_data(data):
        d = data.copy()
        d['batter'] = le_bat.transform(d['batter'])
        d['bowler'] = le_bow.transform(d['bowler'])
        return d[features], d[target]

    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)
    
    # 2. DEFINING MODELS
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, verbosity=-1, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    }
    
    results = []
    
    # 3. COMPETITION LOOP
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Note: CatBoost can handle categorical features natively, but for this lab 
        # we used LabelEncoding for consistency across comparisons.
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        # Calculate Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        duration = time.time() - start_time
        
        print(f"Results for {name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
        
        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "TrainTime(s)": round(duration, 2)
        })
    
    # 4. LEADERBOARD
    leaderboard = pd.DataFrame(results).sort_values(by="MAE")
    print("\n--- FINAL LEADERBOARD (Ordered by MAE) ---")
    print(leaderboard.to_string(index=False))
    
    # Save results to JSON
    leaderboard.to_json("bowler_lab_results.json", orient="records", indent=4)

if __name__ == "__main__":
    run_lab()
