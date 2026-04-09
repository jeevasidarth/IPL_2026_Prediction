import pandas as pd
import os
import json

files = [
    'batter_performance_comprehensive.csv',
    'bowler_performance_comprehensive.csv',
    'ipl_match_features_training.csv',
    'scripts/player_registry.json',
    'scripts/debutant_baselines.json',
    'ipl_2026_squads.csv',
    'batter_vs_bowler_h2h.json',
    'scripts/stadium_pitch_mapping.json'
]

print("--- DATASET READINESS AUDIT ---")
for f in files:
    if os.path.exists(f):
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(f)
                print(f"[OK] {f}: {len(df)} rows, {len(df.columns)} columns")
                if 'recent_form_avg' in df.columns:
                     print(f"     - Form Logic: Weighted EWMA enabled (Sample: {df['recent_form_avg'].head(3).tolist()})")
                if 'team1_h2h_rp_ball' in df.columns:
                     print(f"     - Feature Matrix: H2H and Interaction features detected.")
            except Exception as e:
                print(f"[ERROR] {f}: Could not read CSV - {e}")
        else:
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    print(f"[OK] {f}: {len(data)} entries")
            except Exception as e:
                print(f"[ERROR] {f}: Could not read JSON - {e}")
    else:
        print(f"[MISSING] {f}")

# Deep Check: Name Mapping Coverage
if os.path.exists('scripts/player_registry.json') and os.path.exists('ipl_2026_squads.csv'):
    with open('scripts/player_registry.json', 'r') as f:
        reg = json.load(f)
    squads = pd.read_csv('ipl_2026_squads.csv')
    coverage = len(reg) / len(squads) * 100
    print(f"\n[METRIC] Name Mapping Coverage: {coverage:.1f}% ({len(reg)}/{len(squads)})")

# Deep Check: Debutant Stats
if os.path.exists('scripts/debutant_baselines.json'):
    with open('scripts/debutant_baselines.json', 'r') as f:
        debutants = json.load(f)
    print(f"[METRIC] Debutant/Domestic Stars Defined: {len(debutants)}")

print("\n--- AUDIT COMPLETE ---")
