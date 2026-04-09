import pandas as pd
import json
import os

BATTER_FILE = 'training_batter_hierarchical.csv'
BOWLER_FILE = 'training_bowler_hierarchical.csv'
OUTPUT_LOOKUP = 'player_stats_lookup.json'

def create_lookup():
    print("Creating Player Stats Lookup...")
    df_bat = pd.read_csv(BATTER_FILE)
    df_bowl = pd.read_csv(BOWLER_FILE)
    
    # Sort to get latest data
    df_bat['date'] = pd.to_datetime(df_bat['date'])
    df_bowl['date'] = pd.to_datetime(df_bowl['date'])
    
    lookup = {
        'batters': {},
        'bowlers': {}
    }
    
    # Batter Lookup: Group by player and get latest global stats
    # Also store venue and pitch averages as sub-dictionaries
    for player, group in df_bat.groupby('player_name'):
        latest = group.iloc[-1]
        
        # Venue Averages
        venue_avgs = group.groupby('venue')['runs'].mean().to_dict()
        # Pitch Averages
        pitch_avgs = group.groupby('pitch_type')['runs'].mean().to_dict()
        
        lookup['batters'][player] = {
            'global_avg': float(latest['global_avg']),
            'recent_form_avg': float(latest['recent_form_avg']),
            'venue_avgs': venue_avgs,
            'pitch_avgs': pitch_avgs
        }
    
    # Bowler Lookup
    for player, group in df_bowl.groupby('player_name'):
        latest = group.iloc[-1]
        
        venue_wkt_avgs = group.groupby('venue')['wickets'].mean().to_dict()
        
        lookup['bowlers'][player] = {
            'global_wkt_avg': float(latest['global_wkt_avg']),
            'global_econ_avg': float(latest['global_econ_avg']),
            'recent_form_economy': float(latest['recent_form_economy']),
            'venue_wkt_avgs': venue_wkt_avgs
        }
        
    with open(OUTPUT_LOOKUP, 'w') as f:
        json.dump(lookup, f, indent=4)
    
    print(f"Lookup saved to {OUTPUT_LOOKUP}")

if __name__ == "__main__":
    create_lookup()
