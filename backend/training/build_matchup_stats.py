import pandas as pd
import numpy as np
import json
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(BASE_DIR, 'archive (3)', 'IPL.csv')
OUTPUT_JSON = os.path.join(BASE_DIR, 'matchup_stats.json')
REGISTRY_JSON = os.path.join(BASE_DIR, 'scripts', 'player_registry.json')

def load_registry():
    with open(REGISTRY_JSON, 'r') as f:
        return json.load(f)

def build_matchups():
    print("Loading IPL Raw Data (Last 5 years)...")
    # Load only necessary columns to save memory
    usecols = ['year', 'batter', 'bowler', 'runs_batter', 'valid_ball', 'player_out', 'wicket_kind']
    df = pd.read_csv(RAW_DATA, usecols=usecols)
    
    # Filter for last 5 years (2021-2025)
    df = df[df['year'] >= 2021]
    
    print(f"Processing {len(df)} deliveries...")
    
    # Define bowler-credited wickets
    bowler_wickets = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
    df['is_bowler_wicket'] = df['wicket_kind'].isin(bowler_wickets)
    
    matchup_gb = df.groupby(['batter', 'bowler']).agg(
        total_runs=('runs_batter', 'sum'),
        balls_faced=('valid_ball', 'sum'),
        dismissals=('is_bowler_wicket', 'sum')
    ).reset_index()

    # Calculate SR and Avg
    matchup_gb['strike_rate'] = (matchup_gb['total_runs'] / matchup_gb['balls_faced']) * 100
    matchup_gb['average'] = np.where(matchup_gb['dismissals'] > 0, 
                                     matchup_gb['total_runs'] / matchup_gb['dismissals'], 
                                     matchup_gb['total_runs']) # If not out, avg is total runs (proxy)

    # 2. Global Fallbacks (In case specific matchup is missing)
    batter_global = df.groupby('batter').agg(
        global_avg_runs=('runs_batter', 'mean'),
        global_sr=('runs_batter', lambda x: (x.sum() / df.loc[x.index, 'valid_ball'].sum()) * 100 if df.loc[x.index, 'valid_ball'].sum() > 0 else 0)
    ).to_dict('index')

    bowler_global = df.groupby('bowler').agg(
        global_econ=('runs_batter', lambda x: (x.sum() / df.loc[x.index, 'valid_ball'].sum()) * 6 if df.loc[x.index, 'valid_ball'].sum() > 0 else 8.5)
    ).to_dict('index')

    # Convert Matchup GB to nested dict for fast lookup
    # { 'Batter Name': { 'Bowler Name': { stats } } }
    matchup_dict = {}
    for _, row in matchup_gb.iterrows():
        b = row['batter']
        bow = row['bowler']
        if b not in matchup_dict: matchup_dict[b] = {}
        matchup_dict[b][bow] = {
            'runs': int(row['total_runs']),
            'balls': int(row['balls_faced']),
            'sr': float(row['strike_rate']),
            'avg': float(row['average']),
            'outs': int(row['dismissals'])
        }

    # Final result
    result = {
        'matchups': matchup_dict,
        'batter_global': batter_global,
        'bowler_global': bowler_global,
        'metadata': {
            'years': '2021-2025',
            'deliveries_processed': len(df)
        }
    }

    print(f"Saving {len(matchup_gb)} unique matchups to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    build_matchups()
