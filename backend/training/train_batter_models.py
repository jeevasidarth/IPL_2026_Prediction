import pandas as pd
import numpy as np
import os
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, '..', 'raw_data', 'training_batter_model_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def train_and_evaluate():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Validation constraint check
    if 'strike_rate' not in df.columns:
        raise ValueError("Data missing 'strike_rate' target column.")

    df['strike_rate'] = df['strike_rate'].clip(upper=400)
    
    # Ensure nominal categories are strings
    df['batter'] = df['batter'].astype(str)
    df['bowler'] = df['bowler'].astype(str)
    df['venue'] = df['venue'].astype(str)
    df['phase'] = df['phase'].astype(str)

    # 1. TEMPORAL SPLIT (2019-2024 for Train, 2025 for Test)
    train_df = df[df['year'] < 2025].copy()
    test_df = df[df['year'] == 2025].copy()
    
    print(f"Training set: {len(train_df)} rows. Test set (2025): {len(test_df)} rows.")
    
    # Feature Selection (Now including Categorical Embeddings!)
    features = [
        'batter', 'bowler', 'venue', 'phase',
        'temp', 'humidity', 'dew_point', 
        'bat_global_avg', 'bat_global_sr', 'bowl_global_econ',
        'recent_form_avg', 'recent_form_sr'
    ]
    
    cat_features = ['batter', 'bowler', 'venue', 'phase']
    
    X_train = train_df[features]
    y_train = train_df['strike_rate']
    X_test = test_df[features]
    y_test = test_df['strike_rate']

    print("\nStarting CatBoost Hyperparameter Tuning...")
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    cat = CatBoostRegressor(
        random_state=42, 
        verbose=0
    )
    
    # Deepened param grid for maximum accuracy
    cat_params = {
        'iterations': [500, 800], 
        'depth': [6, 8], 
        'learning_rate': [0.03, 0.05]
    }
    
    cat_grid = GridSearchCV(cat, cat_params, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    cat_grid.fit(X_train, y_train, **{'cat_features': cat_features})
    
    best_model = cat_grid.best_estimator_
    cv_mae = -cat_grid.best_score_
    
    print(f"CatBoost Best Params: {cat_grid.best_params_}")
    print(f"CatBoost CV MAE: {cv_mae:.2f}")

    # Evaluate on hold-out Test dataset
    if len(test_df) == 0:
         print("Warning: No 2025 Test data available. Check data split.")
         return

    y_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print("\n--- FINAL EVALUATION ON 2025 HOLD-OUT SET ---")
    print(f"Hold-out MAE: {test_mae:.2f} Strike Rate points")
    print(f"Hold-out R2:  {test_r2:.4f}")
    
    # Save the architecture
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    save_path = os.path.join(MODELS_DIR, 'batter_matchup_model.cbm')
    best_model.save_model(save_path)
        
    print(f"\nModel saved successfully to: {save_path}")

if __name__ == "__main__":
    train_and_evaluate()
