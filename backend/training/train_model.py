import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import json

DATA_FILE = 'ipl_match_features_training.csv'
MODEL_DIR = 'models/'

import os
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Fill any NA values with medians or 0
df.fillna(0, inplace=True)

# Encode categorical variables: venue, pitch_type
le_venue = LabelEncoder()
df['venue_encoded'] = le_venue.fit_transform(df['venue'])

le_pitch = LabelEncoder()
df['pitch_type_encoded'] = le_pitch.fit_transform(df['pitch_type'])

le_ground = LabelEncoder()
df['ground_size_encoded'] = le_ground.fit_transform(df['ground_size'])

# Select features
features = [
    'venue_encoded', 'pitch_type_encoded', 'ground_size_encoded',
    'team1_bat_first', 'is_afternoon', 'dew_factor', 'temp_i1', 'hum_i1',
    'team1_h2h_rp_ball', 'team2_h2h_rp_ball',
    'team1_avg_bat_form', 'team1_avg_bat_sr_form', 'team1_avg_bowl_wickets_form', 'team1_avg_bowl_econ_form',
    'team2_avg_bat_form', 'team2_avg_bat_sr_form', 'team2_avg_bowl_wickets_form', 'team2_avg_bowl_econ_form',
    'bat_first_dew_interaction', 'bat_first_pitch_interaction'
]

X = df[features]
y_win = df['winner_is_team1']
y_score = df['first_innings_score']

# Train-Test Split
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X, y_win, test_size=0.15, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_score, test_size=0.15, random_state=42)

print("\n--- Training Win Probability Model (XGBClassifier) ---")
clf = xgb.XGBClassifier(
    n_estimators=300, 
    learning_rate=0.03, 
    max_depth=6, 
    eval_metric='logloss',
    random_state=42
)
clf.fit(X_train_w, y_train_w)

y_pred_w = clf.predict(X_test_w)
y_prob_w = clf.predict_proba(X_test_w)[:, 1]

acc = accuracy_score(y_test_w, y_pred_w)
auc = roc_auc_score(y_test_w, y_prob_w)
print(f"Accuracy: {acc:.3f}")
print(f"ROC-AUC: {auc:.3f}")

print("\n--- Training First Innings Score Model (XGBRegressor) ---")
reg = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    random_state=42
)
reg.fit(X_train_s, y_train_s)
y_pred_s = reg.predict(X_test_s)
mae = mean_absolute_error(y_test_s, y_pred_s)
print(f"Mean Absolute Error (Runs): {mae:.2f}")

# Save models and encoders
print("\nSaving models and encoders...")
joblib.dump(clf, f'{MODEL_DIR}win_classifier.pkl')
joblib.dump(reg, f'{MODEL_DIR}score_regressor.pkl')
joblib.dump(le_venue, f'{MODEL_DIR}le_venue.pkl')
joblib.dump(le_pitch, f'{MODEL_DIR}le_pitch.pkl')
joblib.dump(le_ground, f'{MODEL_DIR}le_ground.pkl')

print("SUCCESS: Models trained and saved successfully!")
