import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATTER_FILE = os.path.join(BASE_DIR, '..', 'raw_data', 'batter_performance_comprehensive.csv')
BOWLER_FILE = os.path.join(BASE_DIR, '..', 'raw_data', 'bowler_performance_comprehensive.csv')
ENV_FILE = os.path.join(BASE_DIR, '..', 'raw_data', 'ipl_2019_2025_environmental_final_v2.csv')
OUTPUT_BATTER = os.path.join(BASE_DIR, '..', 'raw_data', 'training_batter_hierarchical.csv')
OUTPUT_BOWLER = os.path.join(BASE_DIR, '..', 'raw_data', 'training_bowler_hierarchical.csv')

def build():
    print("Loading datasets...")
    df_bat = pd.read_csv(BATTER_FILE)
    df_bowl = pd.read_csv(BOWLER_FILE)
    df_env = pd.read_csv(ENV_FILE)
    
    # 1. Processing Batters
    df_bat['date'] = pd.to_datetime(df_bat['date'])
    df_bat = df_bat.sort_values(['player_name', 'date'])
    df_bat['year'] = df_bat['date'].dt.year
    
    # Hierarchical Features: CUMULATIVE (don't leak future data)
    print("Calculating Hierarchical Batter Stats...")
    # Global Cumulative
    df_bat['cum_runs'] = df_bat.groupby('player_name')['runs'].shift().fillna(0).groupby(df_bat['player_name']).cumsum()
    df_bat['cum_matches'] = df_bat.groupby('player_name').cumcount()
    df_bat['global_avg'] = df_bat['cum_runs'] / (df_bat['cum_matches'] + 1)
    
    # Venue Cumulative
    df_bat['venue_cum_runs'] = df_bat.groupby(['player_name', 'venue'])['runs'].shift().fillna(0).groupby([df_bat['player_name'], df_bat['venue']]).cumsum()
    df_bat['venue_cum_matches'] = df_bat.groupby(['player_name', 'venue']).cumcount()
    df_bat['venue_avg'] = df_bat['venue_cum_runs'] / (df_bat['venue_cum_matches'] + 1)
    
    # Pitch Cumulative
    df_bat['pitch_cum_runs'] = df_bat.groupby(['player_name', 'pitch_type'])['runs'].shift().fillna(0).groupby([df_bat['player_name'], df_bat['pitch_type']]).cumsum()
    df_bat['pitch_cum_matches'] = df_bat.groupby(['player_name', 'pitch_type']).cumcount()
    df_bat['pitch_avg'] = df_bat['pitch_cum_runs'] / (df_bat['pitch_cum_matches'] + 1)
    
    # 2. Join with Environmental Context
    # We need to map match_id to environmental factors
    env_cols = ['match_id', 'is_afternoon', 'temp_i1', 'hum_i1', 'dew_i1']
    df_bat = df_bat.merge(df_env[env_cols], on='match_id', how='left')
    
    # 3. Processing Bowlers (Similar Logic)
    print("Calculating Hierarchical Bowler Stats...")
    df_bowl['date'] = pd.to_datetime(df_bowl['date'])
    df_bowl = df_bowl.sort_values(['player_name', 'date'])
    df_bowl['year'] = df_bowl['date'].dt.year
    
    df_bowl['cum_wickets'] = df_bowl.groupby('player_name')['wickets'].shift().fillna(0).groupby(df_bowl['player_name']).cumsum()
    df_bowl['cum_econ_sum'] = df_bowl.groupby('player_name')['economy'].shift().fillna(0).groupby(df_bowl['player_name']).cumsum()
    df_bowl['cum_matches'] = df_bowl.groupby('player_name').cumcount()
    
    df_bowl['global_wkt_avg'] = df_bowl['cum_wickets'] / (df_bowl['cum_matches'] + 1)
    df_bowl['global_econ_avg'] = df_bowl['cum_econ_sum'] / (df_bowl['cum_matches'] + 1)
    
    # Venue Bowler
    df_bowl['venue_cum_wkt'] = df_bowl.groupby(['player_name', 'venue'])['wickets'].shift().fillna(0).groupby([df_bowl['player_name'], df_bowl['venue']]).cumsum()
    df_bowl['venue_cum_matches'] = df_bowl.groupby(['player_name', 'venue']).cumcount()
    df_bowl['venue_wkt_avg'] = df_bowl['venue_cum_wkt'] / (df_bowl['venue_cum_matches'] + 1)
    
    df_bowl = df_bowl.merge(df_env[env_cols], on='match_id', how='left')
    
    # Handle NAs from Shift
    df_bat = df_bat.fillna(0)
    df_bowl = df_bowl.fillna(0)
    
    # Filter for training years (2019-2025)
    df_bat = df_bat[df_bat['year'] >= 2019]
    df_bowl = df_bowl[df_bowl['year'] >= 2019]
    
    print(f"Saving {len(df_bat)} batter rows and {len(df_bowl)} bowler rows...")
    df_bat.to_csv(OUTPUT_BATTER, index=False)
    df_bowl.to_csv(OUTPUT_BOWLER, index=False)
    print("Done!")

if __name__ == "__main__":
    build()
