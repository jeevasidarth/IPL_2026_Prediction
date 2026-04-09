import pandas as pd
import json
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLET_DATA = os.path.join(BASE_DIR, 'training_matchup_triplets_bat.csv')
MATCHUP_STATS = os.path.join(BASE_DIR, 'matchup_stats.json')

def enrich_triplets():
    print("Enriching Triplet Data with Global Stats...")
    df = pd.read_csv(TRIPLET_DATA)
    
    with open(MATCHUP_STATS, 'r') as f:
        stats = json.load(f)
    
    # Map Global Batter Stats
    bat_stats = pd.DataFrame.from_dict(stats['batter_global'], orient='index').reset_index()
    bat_stats.columns = ['batter', 'bat_global_avg', 'bat_global_sr']
    
    # Map Global Bowler Stats
    bowl_stats = pd.DataFrame.from_dict(stats['bowler_global'], orient='index').reset_index()
    bowl_stats.columns = ['bowler', 'bowl_global_econ']
    
    # Merge
    df = df.merge(bat_stats, on='batter', how='left')
    df = df.merge(bowl_stats, on='bowler', how='left')
    
    # Fill NaNs for new players
    df['bat_global_avg'] = df['bat_global_avg'].fillna(18.0)
    df['bat_global_sr'] = df['bat_global_sr'].fillna(130.0)
    df['bowl_global_econ'] = df['bowl_global_econ'].fillna(8.5)
    
    print(f"Data enriched. Final columns: {df.columns.tolist()}")
    df.to_csv(TRIPLET_DATA, index=False)
    print("Enriched triplet data saved.")

if __name__ == "__main__":
    enrich_triplets()
