import pandas as pd
import numpy as np
import os
import json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(BASE_DIR, '..', 'raw_data', 'archive (3)', 'IPL.csv')
ENV_DATA = os.path.join(BASE_DIR, '..', 'raw_data', 'ipl_2019_2025_environmental_final.csv')
MATCHUP_STATS = os.path.join(BASE_DIR, 'data', 'matchup_stats.json')
OUTPUT_CSV = os.path.join(BASE_DIR, '..', 'raw_data', 'training_batter_model_data.csv')

def build_data():
    print("Step 1: Reading Ball-by-Ball Data (2019-2025)...")
    df = pd.read_csv(RAW_DATA, usecols=['match_id', 'year', 'venue', 'innings', 'batter', 'bowler', 'runs_batter', 'valid_ball', 'over'])
    
    # Filter 2019 to 2025
    df = df[(df['year'] >= 2019) & (df['year'] <= 2025)]
    
    # Extract Phase from Over
    # T20 Phases: 0-5 Powerplay, 6-14 Middle, 15-19 Death
    conditions = [
        (df['over'] <= 5),
        (df['over'] >= 6) & (df['over'] <= 14),
        (df['over'] >= 15)
    ]
    choices = ['Powerplay', 'Middle', 'Death']
    df['phase'] = np.select(conditions, choices, default='Middle')
    
    print("Step 2: Aggregating into Batter-vs-Bowler Triplets by Phase...")
    triplets = df.groupby(['match_id', 'year', 'venue', 'batter', 'bowler', 'phase']).agg(
        runs=('runs_batter', 'sum'),
        balls=('valid_ball', 'sum')
    ).reset_index()
    
    # Filter for meaningful encounters to avoid noise (e.g. 6 off 1 ball = 600 SR)
    # We enforce at least 3 balls faced
    triplets = triplets[triplets['balls'] >= 3].copy()
    
    # Target Variable
    triplets['strike_rate'] = (triplets['runs'] / triplets['balls']) * 100
    
    print("Step 3: Enriching with Environmental Data...")
    env_df = pd.read_csv(ENV_DATA, usecols=['match_id', 'temp', 'humidity', 'dew_point'])
    # Convert types in environmental data just in case
    env_df['temp'] = pd.to_numeric(env_df['temp'], errors='coerce').fillna(30.0)
    env_df['humidity'] = pd.to_numeric(env_df['humidity'], errors='coerce').fillna(60.0)
    env_df['dew_point'] = pd.to_numeric(env_df['dew_point'], errors='coerce').fillna(20.0)
    
    # Merge env data
    triplets = triplets.merge(env_df, on='match_id', how='left')
    
    # Fill any missing env mapping with average baselines
    triplets['temp'] = triplets['temp'].fillna(30.0)
    triplets['humidity'] = triplets['humidity'].fillna(60.0)
    triplets['dew_point'] = triplets['dew_point'].fillna(20.0)
    
    print("Step 4: Enriching with Global & Recent Form Player Statistics...")
    with open(MATCHUP_STATS, 'r') as f:
        stats = json.load(f)
    
    # Map Global Batter Stats
    bat_stats = pd.DataFrame.from_dict(stats['batter_global'], orient='index').reset_index()
    bat_stats.columns = ['batter', 'bat_global_avg', 'bat_global_sr']
    
    # Map Global Bowler Stats
    bowl_stats = pd.DataFrame.from_dict(stats['bowler_global'], orient='index').reset_index()
    bowl_stats.columns = ['bowler', 'bowl_global_econ']
    
    # Merge global stats
    triplets = triplets.merge(bat_stats, on='batter', how='left')
    triplets = triplets.merge(bowl_stats, on='bowler', how='left')
    
    # Fill defaults for new/unknown players
    triplets['bat_global_avg'] = triplets['bat_global_avg'].fillna(18.0)
    triplets['bat_global_sr'] = triplets['bat_global_sr'].fillna(130.0)
    triplets['bowl_global_econ'] = triplets['bowl_global_econ'].fillna(8.5)

    # NOW: Map Dynamic Recent Form for the Batter 
    print("Step 5: Mering Dynamic Recent Match Form...")
    RECENT_FORM_DB = os.path.join(BASE_DIR, '..', 'raw_data', 'batter_performance_comprehensive.csv')
    try:
        recent_df = pd.read_csv(RECENT_FORM_DB, usecols=['match_id', 'player_name', 'recent_form_avg', 'recent_form_sr'])
        recent_df.rename(columns={'player_name': 'batter'}, inplace=True)
        # Drop duplicates just in case there are multiple entries
        recent_df.drop_duplicates(subset=['match_id', 'batter'], inplace=True)
        triplets = triplets.merge(recent_df, on=['match_id', 'batter'], how='left')
        
        # Fill missing recent form with global stats for very first matches
        triplets['recent_form_sr'] = triplets['recent_form_sr'].fillna(triplets['bat_global_sr'])
        triplets['recent_form_avg'] = triplets['recent_form_avg'].fillna(triplets['bat_global_avg'])
    except Exception as e:
        print(f"Warning: Could not bind recent form: {e}")
        triplets['recent_form_sr'] = triplets['bat_global_sr']
        triplets['recent_form_avg'] = triplets['bat_global_avg']
    
    print(f"Final Dataset Size: {len(triplets)} rows")
    print(f"Saving to {OUTPUT_CSV}...")
    triplets.to_csv(OUTPUT_CSV, index=False)
    print("SUCCESS: Data processing complete!")

if __name__ == "__main__":
    build_data()
