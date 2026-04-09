import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Files
BATTER_TRAIN = 'training_batter_hierarchical.csv'
BOWLER_TRAIN = 'training_bowler_hierarchical.csv'

def train():
    df_bat = pd.read_csv(BATTER_TRAIN)
    df_bowl = pd.read_csv(BOWLER_TRAIN)
    
    # 1. TEMPORAL SPLIT (2025 as hold-out)
    train_bat = df_bat[df_bat['year'] < 2025]
    test_bat = df_bat[df_bat['year'] == 2025]
    
    train_bowl = df_bowl[df_bowl['year'] < 2025]
    test_bowl = df_bowl[df_bowl['year'] == 2025]
    
    print(f"Batter Training Rows: {len(train_bat)}, Test (2025) Rows: {len(test_bat)}")
    
    # Batting Features
    bat_features = [
        'is_afternoon', 'temp_i1', 'hum_i1', 'dew_i1',
        'global_avg', 'venue_avg', 'pitch_avg', 'recent_form_avg'
    ]
    
    # Bowling Features
    bowl_features = [
        'is_afternoon', 'temp_i1', 'hum_i1', 'dew_i1',
        'global_wkt_avg', 'global_econ_avg', 'venue_wkt_avg', 'recent_form_economy'
    ]
    
    # Hyperparameter Grid
    param_grid = {
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 300]
    }
    
    # 2. TRAIN BATTER MODEL
    print("\n--- Training Batter Regressor (Tuning via CV) ---")
    X_bat_train = train_bat[bat_features]
    y_bat_train = train_bat['runs']
    
    grid_bat = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_absolute_error')
    grid_bat.fit(X_bat_train, y_bat_train)
    
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
    
    grid_bowl = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_absolute_error')
    grid_bowl.fit(X_bowl_train, y_bowl_train)
    
    best_bowl = grid_bowl.best_estimator_
    print(f"Best Bowler Params: {grid_bowl.best_params_}")
    
    # Evaluate on 2025
    y_bowl_pred = best_bowl.predict(test_bowl[bowl_features])
    mae_bowl = mean_absolute_error(test_bowl['economy'], y_bowl_pred)
    print(f"Bowler Econ MAE (2025 Hold-out): {mae_bowl:.2f}")
    
    # Save models
    joblib.dump(best_bat, 'models/batter_runs_model.pkl')
    joblib.dump(best_bowl, 'models/bowler_econ_model.pkl')
    print("\nSUCCESS: Hierarchical models trained and saved!")

if __name__ == "__main__":
    import os
    if not os.path.exists('models'): os.makedirs('models')
    train()
