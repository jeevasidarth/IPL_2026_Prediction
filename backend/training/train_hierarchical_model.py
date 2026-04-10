import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os

# Files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATTER_TRAIN = os.path.join(BASE_DIR, '..', 'raw_data', 'training_batter_hierarchical.csv')
BOWLER_TRAIN = os.path.join(BASE_DIR, '..', 'raw_data', 'training_bowler_hierarchical.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def train():
    df_bat = pd.read_csv(BATTER_TRAIN)
    df_bowl = pd.read_csv(BOWLER_TRAIN)
    
    # Cast Categoricals to string explicitly
    for col in ['player_name', 'venue', 'pitch_type']:
        if col in df_bat.columns: df_bat[col] = df_bat[col].astype(str)
        if col in df_bowl.columns: df_bowl[col] = df_bowl[col].astype(str)
    
    # 1. TEMPORAL SPLIT (2025 as hold-out)
    train_bat = df_bat[df_bat['year'] < 2025]
    test_bat = df_bat[df_bat['year'] == 2025]
    
    train_bowl = df_bowl[df_bowl['year'] < 2025]
    test_bowl = df_bowl[df_bowl['year'] == 2025]
    
    print(f"Batter Training Rows: {len(train_bat)}, Test (2025) Rows: {len(test_bat)}")
    
    # Batting Features
    bat_features = [
        'player_name', 'venue', 'pitch_type',
        'is_afternoon', 'temp_i1', 'hum_i1', 'dew_i1',
        'global_avg', 'venue_avg', 'pitch_avg', 'recent_form_avg'
    ]
    
    # Bowling Features
    bowl_features = [
        'player_name', 'venue', 'pitch_type', 'phase',
        'is_afternoon', 'temp_i1', 'hum_i1', 'dew_i1',
        'global_wkt_avg', 'global_econ_avg', 'venue_wkt_avg', 'recent_form_economy'
    ]
    
    cat_cols_bat = ['player_name', 'venue', 'pitch_type']
    cat_cols_bowl = ['player_name', 'venue', 'pitch_type', 'phase']
    
    # Deep Grid
    param_grid = {
        'iterations': [500, 800],
        'depth': [6, 8],
        'learning_rate': [0.03, 0.05]
    }
    
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # 2. TRAIN BATTER MODEL (Team Level aggregation model)
    print("\n--- Training Batter Regressor (Tuning via CV) ---")
    X_bat_train = train_bat[bat_features]
    y_bat_train = train_bat['runs']
    
    cat_bat = CatBoostRegressor(random_state=42, verbose=0)
    grid_bat = GridSearchCV(cat_bat, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_bat.fit(X_bat_train, y_bat_train, **{'cat_features': cat_cols_bat})
    
    best_bat = grid_bat.best_estimator_
    print(f"Best Batter Params: {grid_bat.best_params_}")
    
    # Evaluate on 2025
    y_bat_pred = best_bat.predict(test_bat[bat_features])
    mae_bat = mean_absolute_error(test_bat['runs'], y_bat_pred)
    print(f"Batter MAE (2025 Hold-out): {mae_bat:.2f} runs")
    
    # 3. TRAIN BOWLER MODEL
    print("\n--- Training Bowler Regressor (Tuning via CV) ---")
    X_bowl_train = train_bowl[bowl_features]
    y_bowl_train = train_bowl['economy']
    
    cat_bowl = CatBoostRegressor(random_state=42, verbose=0)
    grid_bowl = GridSearchCV(cat_bowl, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_bowl.fit(X_bowl_train, y_bowl_train, **{'cat_features': cat_cols_bowl})
    
    best_bowl = grid_bowl.best_estimator_
    print(f"Best Bowler Params: {grid_bowl.best_params_}")
    
    # Evaluate on 2025
    y_bowl_pred = best_bowl.predict(test_bowl[bowl_features])
    mae_bowl = mean_absolute_error(test_bowl['economy'], y_bowl_pred)
    print(f"Bowler Econ MAE (2025 Hold-out): {mae_bowl:.2f}")
    
    # Save models
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    best_bat.save_model(os.path.join(MODELS_DIR, 'batter_runs_model.cbm'))
    best_bowl.save_model(os.path.join(MODELS_DIR, 'bowler_econ_model.cbm'))
    print("\nSUCCESS: Hierarchical models trained and saved as .cbm!")

if __name__ == "__main__":
    train()
