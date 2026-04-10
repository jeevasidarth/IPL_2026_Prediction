import pandas as pd
import json
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATTER_CSV = os.path.join(BASE_DIR, '..', 'raw_data', 'batter_performance_comprehensive.csv')
BOWLER_CSV = os.path.join(BASE_DIR, '..', 'raw_data', 'bowler_performance_comprehensive.csv')
MATCHUP_STATS = os.path.join(BASE_DIR, 'data', 'matchup_stats.json')
RISING_STARS = os.path.join(BASE_DIR, 'data', 'rising_stars.json')
OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'inference_lookup.json')

def generate():
    print("Generating Inference Lookup...")
    
    # Load data
    df_bat = pd.read_csv(BATTER_CSV)
    df_bowl = pd.read_csv(BOWLER_CSV)
    
    with open(MATCHUP_STATS, 'r') as f:
        global_stats = json.load(f)
        
    rising_stars = {}
    if os.path.exists(RISING_STARS):
        with open(RISING_STARS, 'r') as f:
            rising_stars = json.load(f)

    lookup = {
        "batters": {},
        "bowlers": {}
    }

    # Get latest recent form for each batter
    print("Processing Batters...")
    latest_bat = df_bat.sort_values('date').groupby('player_name').last()
    for name, row in latest_bat.iterrows():
        # Mix in global stats as fallbacks
        global_b = global_stats['batter_global'].get(name, {"bat_global_avg": 18.0, "bat_global_sr": 130.0})
        
        lookup["batters"][name] = {
            "recent_form_avg": float(row['recent_form_avg']),
            "recent_form_sr": float(row['recent_form_sr']),
            "global_avg": float(global_b.get('bat_global_avg', 18.0)),
            "global_sr": float(global_b.get('bat_global_sr', 130.0))
        }

    # Get latest recent form for each bowler
    print("Processing Bowlers...")
    latest_bowl = df_bowl.sort_values('date').groupby('player_name').last()
    for name, row in latest_bowl.iterrows():
        global_bow = global_stats['bowler_global'].get(name, {"bowl_global_econ": 8.5})
        
        lookup["bowlers"][name] = {
            "recent_form_economy": float(row['recent_form_economy']),
            "global_econ": float(global_bow.get('bowl_global_econ', 8.5)),
            "global_wkt_avg": float(global_bow.get('bowl_global_wkt_avg', 1.0) if 'bowl_global_wkt_avg' in global_bow else 1.0)
        }

    # 3. Add Manually defined/missing players from Squads
    SQUADS_JSON = os.path.join(BASE_DIR, 'data', 'squads_2026_enriched.json')
    if os.path.exists(SQUADS_JSON):
        print("Checking Squads for missing players...")
        with open(SQUADS_JSON, 'r') as f:
            squads = json.load(f)
        
        all_squad_players = []
        for s in squads.values():
            all_squad_players.extend(s.get('Batters', []))
            all_squad_players.extend(s.get('Bowlers', []))
        
        # Registry mapping for checking
        REGISTRY = os.path.join(BASE_DIR, 'data', 'player_registry.json')
        with open(REGISTRY, 'r') as f:
            reg = json.load(f)

        for p in set(all_squad_players):
            mapped_name = reg.get(p, p)
            # If player not in batters and not in bowlers in lookup
            in_bat = mapped_name in lookup['batters']
            in_bowl = mapped_name in lookup['bowlers']
            
            if not in_bat and not in_bowl:
                print(f"Injecting baseline for new player: {p} (as {mapped_name})")
                # Detect role based on squad list
                is_bowler = any(p in s.get('Bowlers', []) for s in squads.values())
                
                # Default baseline stats
                if is_bowler:
                    lookup["bowlers"][mapped_name] = {
                        "recent_form_economy": 8.5,
                        "global_econ": 8.2,
                        "global_wkt_avg": 1.0
                    }
    # 4. APPLY RISING STAR BOOSTS (Dedicated Pass)
    print("Applying Rising Star Prodigy Stats...")
    for p, star in rising_stars.items():
        mapped_name = reg.get(p, p)
        if star['role'] == 'batter':
            print(f"Boosting BATTER: {p} (as {mapped_name}) -> {star['avg']} Avg, {star['sr']} SR")
            lookup["batters"][mapped_name] = {
                "recent_form_avg": star['avg'],
                "recent_form_sr": star['sr'],
                "global_avg": star['avg'],
                "global_sr": star['sr']
            }
        elif star['role'] == 'bowler':
            print(f"Boosting BOWLER: {p} (as {mapped_name}) -> {star['econ']} Econ")
            lookup["bowlers"][mapped_name] = {
                "recent_form_economy": star['econ'],
                "global_econ": star['econ'],
                "global_wkt_avg": star.get('wkt_avg', 1.0)
            }

    # Custom Override for Specific Players (like Muzarabani)
    if "Blessing Muzarabani" in lookup["bowlers"]:
        lookup["bowlers"]["Blessing Muzarabani"] = {
            "recent_form_economy": 8.1,
            "global_econ": 8.0,
            "global_wkt_avg": 1.2
        }

    # Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(lookup, f, indent=4)
    
    print(f"Inference lookup saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    generate()
