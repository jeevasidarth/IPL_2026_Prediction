import pandas as pd
import numpy as np
import joblib
import json
import argparse
import os
from catboost import CatBoostRegressor

# Setup paths — backend/ is the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MATCHUP_JSON = os.path.join(DATA_DIR, 'matchup_stats.json')
REGISTRY_JSON = os.path.join(DATA_DIR, 'player_registry.json')
PITCH_MAPPING = os.path.join(DATA_DIR, 'stadium_pitch_mapping.json')

# Models
BAT_MODEL = os.path.join(MODEL_DIR, 'ensemble_bat_catboost.cbm')
BOWL_MODEL = os.path.join(MODEL_DIR, 'ensemble_bowl_catboost.cbm')
META_MODEL = os.path.join(MODEL_DIR, 'ensemble_win_meta_model.pkl')

class EnsembleMatchPredictor:
    def __init__(self):
        print("Initializing Hybrid Ensemble Engine...")
        with open(MATCHUP_JSON, 'r') as f: self.matchups = json.load(f)
        with open(REGISTRY_JSON, 'r') as f: self.registry = json.load(f)
        with open(PITCH_MAPPING, 'r') as f: self.pitch_map = json.load(f)
        
        self.bat_model = CatBoostRegressor()
        self.bat_model.load_model(BAT_MODEL)
        
        self.bowl_model = CatBoostRegressor()
        self.bowl_model.load_model(BOWL_MODEL)
        
        self.meta_model = joblib.load(META_MODEL)

    def get_matchup_expectation(self, batter_name, bowler_name):
        """Calculates expected runs for a specific batter vs bowler matchup."""
        b_mapped = self.registry.get(batter_name, batter_name)
        bow_mapped = self.registry.get(bowler_name, bowler_name)
        
        # 1. Check direct matchup (last 5 years)
        match_stats = self.matchups['matchups'].get(b_mapped, {}).get(bow_mapped)
        
        if match_stats:
            # Use historical SR and Avg to calculate expected outcome for 10 balls faced
            exp_runs = (match_stats['sr'] / 100) * 10
            return exp_runs
        
        # 2. Fallback to Global stats
        b_global = self.matchups['batter_global'].get(b_mapped, {'global_avg_runs': 18.0, 'global_sr': 130.0})
        bow_global = self.matchups['bowler_global'].get(bow_mapped, {'global_econ': 8.5})
        
        # Hybrid expectation (Bat SR vs Bowl Econ)
        avg_sr = b_global['global_sr']
        avg_econ = bow_global['global_econ']
        
        # Simple balanced expectation for 10 balls
        exp_runs = ((avg_sr / 100) + (avg_econ / 6)) / 2 * 10
        return exp_runs

    def simulate_innings(self, bat_team_xi, bowl_team_xi, env):
        """Simulates an entire innings using matchups."""
        total_score = 0
        batter_contributions = []
        
        # Assume top 7-8 batters face the bulk of the 120 balls
        # We simulate interactions for each batter vs the entire bowling unit
        for i, batter in enumerate(bat_team_xi[:8]):
            # Each major batter faces approx 15-20 balls on avg if they stay in
            expected_balls = 20 - (i * 2) # Early batters face more balls
            if expected_balls < 5: expected_balls = 5
            
            # Predict outcome vs each bowler in opposing XI (weighted equally for simplicity)
            matchup_runs = []
            for bowler in bowl_team_xi: # Usually team has 5-6 bowlers
                runs_per_10 = self.get_matchup_expectation(batter, bowler)
                matchup_runs.append(runs_per_10)
            
            avg_matchup_runs = sum(matchup_runs) / len(matchup_runs)
            batter_total = (avg_matchup_runs / 10) * expected_balls
            
            # Adjust by individual form if we have it (optional enhancement)
            total_score += batter_total
            batter_contributions.append((batter, round(batter_total, 1)))
            
        return {'score': round(total_score), 'batters': batter_contributions}

    def predict_match(self, team1_xi, team2_xi, venue, env):
        # 1. Simulate Scenarios
        print(f"\nAnalyzing {len(team1_xi)} vs {len(team2_xi)} matchups...")
        
        # Scenario A: Team 1 Bats First
        s1 = self.simulate_innings(team1_xi, team2_xi, env)
        # Scenario B: Team 2 Bats First
        s2 = self.simulate_innings(team2_xi, team1_xi, env)
        
        # 2. Meta-Prediction for Win Probability
        # Features: [is_afternoon, temp, hum, dew_factor, t1_score, t2_score, t1_avg_bat, t2_avg_bat]
        # We map our simulated outputs to the meta-model's expected features
        meta_features = pd.DataFrame([{
            'is_afternoon': env.get('is_afternoon', 0),
            'temp_i1': env.get('temp_i1', 30.0),
            'hum_i1': env.get('hum_i1', 65.0),
            'dew_factor': env.get('dew_i1', 20.0),
            'team1_h2h_rp_ball': s1['score'] / 120,
            'team2_h2h_rp_ball': s2['score'] / 120,
            'team1_avg_bat_form': s1['score'] / 7, # Proxy for team batting strength
            'team2_avg_bat_form': s2['score'] / 7
        }])
        
        win_prob = float(self.meta_model.predict_proba(meta_features)[0][1])
        
        return {
            'team1_score': int(s1['score']),
            'team2_score': int(s2['score']),
            'win_prob': win_prob,
            'team1_highlights': [(n, float(r)) for n, r in s1['batters']],
            'team2_highlights': [(n, float(r)) for n, r in s2['batters']]
        }

if __name__ == "__main__":
    # Test run
    predictor = EnsembleMatchPredictor()
    # Mock data
    t1 = ["Virat Kohli", "Phil Salt", "Rajat Patidar", "Jitesh Sharma", "Tim David"]
    t2 = ["Jasprit Bumrah", "Trent Boult", "Hardik Pandya", "Deepak Chahar"]
    res = predictor.predict_match(t1, t2, "Wankhede", {})
    print(f"Result: {res}")
