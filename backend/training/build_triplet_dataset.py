import pandas as pd
import numpy as np
import os
import json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(BASE_DIR, 'archive (3)', 'IPL.csv')
OUTPUT_BAT_MATCHUP = os.path.join(BASE_DIR, 'training_matchup_triplets_bat.csv')
OUTPUT_BOWL_MATCHUP = os.path.join(BASE_DIR, 'training_matchup_triplets_bowl.csv')

def build_triplet_datasets():
    print("Building Matchup Triplet Dataset (Batter-Bowler-Match)...")
    usecols = ['match_id', 'year', 'venue', 'innings', 'batter', 'bowler', 'runs_batter', 'valid_ball', 'wicket_kind']
    df = pd.read_csv(RAW_DATA, usecols=usecols)
    
    # Filter for last 5 years
    df = df[df['year'] >= 2021]
    
    # 1. Batter vs Bowler in Match
    # One row per batter vs bowler in the same match
    bowler_wickets = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
    df['is_bowler_wicket'] = df['wicket_kind'].isin(bowler_wickets)
    
    matchup_triplets = df.groupby(['match_id', 'year', 'venue', 'innings', 'batter', 'bowler']).agg(
        runs=('runs_batter', 'sum'),
        balls=('valid_ball', 'sum'),
        wickets=('is_bowler_wicket', 'sum')
    ).reset_index()
    
    # Filter out cases with 0 balls faced to avoid division by zero
    matchup_triplets = matchup_triplets[matchup_triplets['balls'] > 0].copy()
    
    matchup_triplets['strike_rate'] = (matchup_triplets['runs'] / matchup_triplets['balls']) * 100
    matchup_triplets['economy'] = (matchup_triplets['runs'] / matchup_triplets['balls']) * 6

    # 2. Add features from existing stats (could be done here or in training)
    # For now, let's keep it clean as a raw triplet dataset
    
    print(f"Saving {len(matchup_triplets)} triplets to {OUTPUT_BAT_MATCHUP}...")
    matchup_triplets.to_csv(OUTPUT_BAT_MATCHUP, index=False)
    
    print("DONE building triplet dataset.")

if __name__ == "__main__":
    build_triplet_datasets()
