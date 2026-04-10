import pandas as pd
import numpy as np
import joblib
import json
import os
from catboost import CatBoostRegressor

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOOKUP_JSON = os.path.join(DATA_DIR, 'inference_lookup.json')
REGISTRY_JSON = os.path.join(DATA_DIR, 'player_registry.json')
PITCH_MAPPING = os.path.join(DATA_DIR, 'stadium_pitch_mapping.json')
SQUADS_JSON = os.path.join(DATA_DIR, 'squads_2026_enriched.json')

# New 3-Phase Models
BATTER_MATCHUP_MODEL = os.path.join(MODEL_DIR, 'batter_matchup_model.cbm') # SR by phase
BOWLER_ECON_MODEL = os.path.join(MODEL_DIR, 'bowler_econ_model.cbm')       # Econ by phase
HIERARCHICAL_RUNS_MODEL = os.path.join(MODEL_DIR, 'batter_runs_model.cbm')  # Base team expectation

class EnsembleMatchPredictor:
    def __init__(self):
        print("Initializing High-Intensity 3-Phase Engine...")
        with open(LOOKUP_JSON, 'r') as f: self.lookup = json.load(f)
        with open(REGISTRY_JSON, 'r') as f: self.registry = json.load(f)
        with open(PITCH_MAPPING, 'r') as f: self.pitch_map = json.load(f)
        
        # Load CatBoost models
        self.sr_model = CatBoostRegressor()
        self.sr_model.load_model(BATTER_MATCHUP_MODEL)
        
        self.econ_model = CatBoostRegressor()
        self.econ_model.load_model(BOWLER_ECON_MODEL)
        
        self.runs_model = CatBoostRegressor()
        self.runs_model.load_model(HIERARCHICAL_RUNS_MODEL)

        # Load Definitive Roles from Squads
        with open(SQUADS_JSON, 'r') as f: squads = json.load(f)
        self.role_map = {}
        for team, units in squads.items():
            for p in units.get('Batters', []): self.role_map[p] = 'batter'
            # If in both, 'bowler' often indicates an All-rounder who bowls regularly
            for p in units.get('Bowlers', []): 
                if p in self.role_map: self.role_map[p] = 'allrounder'
                else: self.role_map[p] = 'bowler'
        
        # Keep meta-model for final win-prob calibration
        META_MODEL = os.path.join(MODEL_DIR, 'ensemble_win_meta_model.pkl')
        if os.path.exists(META_MODEL):
            self.meta_model = joblib.load(META_MODEL)
        else:
            self.meta_model = None

    def get_player_stats(self, name, role='batter'):
        mapped_name = self.registry.get(name, name)
        category = 'batters' if role == 'batter' else 'bowlers'
        return self.lookup[category].get(mapped_name, {})

    def get_pitch_data(self, venue):
        for k, v in self.pitch_map.items():
            if k in venue or venue in k:
                return v
        return {"pitch_type": "Balanced", "avg_score": 165}

    def simulate_innings(self, bat_team_xi, bowl_team_xi, venue_name, env):
        pitch_info = self.get_pitch_data(venue_name)
        
        phases = [
            {"name": "Powerplay", "overs": 6},
            {"name": "Middle", "overs": 9},
            {"name": "Death", "overs": 5}
        ]
        
        total_score = 0
        phase_stats = []
        player_projections = {}
        
        # Initialize projections for all players robustly
        for p in set(bat_team_xi + bowl_team_xi):
            player_projections[p] = {'role': 'allrounder', 'runs': 0, 'balls': 0, 'econ': 0, 'wkts': 0, 'overs': 0, 'phases': []}
            
        for p in bowl_team_xi: player_projections[p]['role'] = 'bowler'
        for p in bat_team_xi: player_projections[p]['role'] = 'batter'
        
        current_runs = 0
        worm_data = [0]
        worm_wickets = []
        
        for p_idx, phase_data in enumerate(phases):
            phase_name = phase_data['name']
            overs = phase_data['overs']
            
            # 1. Bowling Unit Performance
            phase_econs = []
            phase_wkts_per_over = []
            for bowler in bowl_team_xi[:6]:
                stats = self.get_player_stats(bowler, 'bowler')
                feat = pd.DataFrame([{
                    'player_name': self.registry.get(bowler, bowler),
                    'venue': venue_name,
                    'pitch_type': pitch_info['pitch_type'],
                    'phase': phase_name,
                    'is_afternoon': env.get('is_afternoon', 0),
                    'temp_i1': env.get('temp_i1', 30.0),
                    'hum_i1': env.get('hum_i1', 65.0),
                    'dew_i1': env.get('dew_i1', 20.0),
                    'global_wkt_avg': stats.get('global_wkt_avg', 0.8),
                    'global_econ_avg': stats.get('global_econ', 8.5),
                    'venue_wkt_avg': 0.8,
                    'recent_form_economy': stats.get('recent_form_economy', 8.5)
                }])
                pred_econ = self.econ_model.predict(feat)[0]
                phase_econs.append(pred_econ)
                
                # Wicket probability heuristic
                avg_wkt = (12 - pred_econ) / 10 * (stats.get('global_wkt_avg', 0.8))
                phase_wkts_per_over.append(avg_wkt / (overs if overs > 0 else 1))

                b_proj = player_projections.setdefault(bowler, {'role': 'bowler', 'econ': 0, 'wkts': 0, 'overs': 0})
                b_proj['overs'] += overs / 6 
                b_proj['econ'] = (b_proj['econ'] * (b_proj['overs'] - overs/6) + pred_econ * (overs/6)) / b_proj['overs']
                
                est_wkt = avg_wkt * (overs/6)
                b_proj['wkts'] += est_wkt
                b_proj['phases'].append({"phase": phase_name, "econ": round(pred_econ, 1), "wkts": round(est_wkt, 2)}) 

            avg_phase_econ = np.mean(phase_econs)
            avg_phase_wkt_prob = np.mean(phase_wkts_per_over)
            
            # 2. Batting Unit Performance
            phase_srs = {}
            active_batters = bat_team_xi[:7] 
            for batter in active_batters:
                stats = self.get_player_stats(batter, 'batter')
                feat = pd.DataFrame([{
                    'batter': self.registry.get(batter, batter),
                    'bowler': self.registry.get(bowl_team_xi[0], bowl_team_xi[0]),
                    'venue': venue_name,
                    'phase': phase_name,
                    'temp': env.get('temp_i1', 30.0),
                    'humidity': env.get('hum_i1', 65.0),
                    'dew_point': env.get('dew_i1', 20.0),
                    'bat_global_avg': stats.get('global_avg', 20.0),
                    'bat_global_sr': stats.get('global_sr', 130.0),
                    'bowl_global_econ': avg_phase_econ,
                    'recent_form_avg': stats.get('recent_form_avg', 20.0),
                    'recent_form_sr': stats.get('recent_form_sr', 130.0)
                }])
                pred_sr = self.sr_model.predict(feat)[0]
                phase_srs[batter] = pred_sr
                
            avg_phase_sr = np.mean(list(phase_srs.values()))
            
            # 3. Phase Scoring & Worm Generation
            runs_from_sr = (avg_phase_sr / 100) * 6 * overs
            runs_from_econ = avg_phase_econ * overs
            phase_runs = (runs_from_sr + runs_from_econ) / 2
            
            # Over-by-over distribution
            current_over_offset = len(worm_data) - 1
            for o in range(overs):
                # Add slight random noise (+/- 25%) while maintaining average
                noise = np.random.uniform(0.75, 1.25)
                over_runs = (phase_runs / overs) * noise
                current_runs += over_runs
                worm_data.append(round(current_runs, 2))
                
                # Check for wicket (stochastic)
                if np.random.random() < (avg_phase_wkt_prob * 1.2): # Scale slightly for visual impact
                    worm_wickets.append({"over": current_over_offset + o + 1, "runs": round(current_runs, 2)})

            # Update batter stats
            for b in active_batters:
                share = (phase_srs[b] / sum(phase_srs.values()))
                b_runs = phase_runs * share
                player_projections[b]['runs'] += b_runs
                player_projections[b]['balls'] += (overs * 6) * share
                player_projections[b]['phases'].append({
                    "phase": phase_name,
                    "sr": round(phase_srs[b], 1),
                    "runs": round(b_runs, 1)
                })
            
            total_score += phase_runs
            phase_stats.append({
                "phase": phase_name,
                "runs": round(phase_runs, 1),
                "sr": round(avg_phase_sr, 1),
                "econ": round(avg_phase_econ, 1)
            })

        return {
            'score': round(total_score),
            'worms': worm_data,
            'worm_wickets': worm_wickets,
            'player_stats': player_projections,
            'phases': phase_stats
        }

    def compare_players(self, p1_name, p2_name, venue_name, env):
        # Use definitive role map, fallback to historical lookup
        p1_role = self.role_map.get(p1_name)
        if not p1_role:
            p1_role = 'bowler' if self.registry.get(p1_name, p1_name) in self.lookup['bowlers'] else 'batter'
            
        p2_role = self.role_map.get(p2_name)
        if not p2_role:
            p2_role = 'bowler' if self.registry.get(p2_name, p2_name) in self.lookup['bowlers'] else 'batter'

        # If comparing p1 vs p2 and one is an allrounder, adapt to other
        if p1_role == 'allrounder': p1_role = 'bowler' if p2_role == 'batter' else 'batter'
        if p2_role == 'allrounder': p2_role = 'bowler' if p1_role == 'batter' else 'batter'
        
        p1_stats = self.get_player_stats(p1_name, p1_role)
        p2_stats = self.get_player_stats(p2_name, p2_role)
        
        # Scenario: Batter vs Bowler
        if p1_role != p2_role:
            batter = p1_name if p1_role == 'batter' else p2_name
            bowler = p1_name if p1_role == 'bowler' else p2_name
            
            b_stats = p1_stats if p1_role == 'batter' else p2_stats
            w_stats = p1_stats if p1_role == 'bowler' else p2_stats
            
            pitch_info = self.get_pitch_data(venue_name)
            matchup_data = []
            
            for phase in ["Powerplay", "Middle", "Death"]:
                # Predict SR
                feat_sr = pd.DataFrame([{
                    'batter': self.registry.get(batter, batter),
                    'bowler': self.registry.get(bowler, bowler),
                    'venue': venue_name,
                    'phase': phase,
                    'temp': env.get('temp_i1', 30.0),
                    'humidity': env.get('hum_i1', 65.0),
                    'dew_point': env.get('dew_i1', 20.0),
                    'bat_global_avg': b_stats.get('global_avg', 20.0),
                    'bat_global_sr': b_stats.get('global_sr', 130.0),
                    'bowl_global_econ': w_stats.get('global_econ', 8.5),
                    'recent_form_avg': b_stats.get('recent_form_avg', 20.0),
                    'recent_form_sr': b_stats.get('recent_form_sr', 130.0)
                }])
                pred_sr = self.sr_model.predict(feat_sr)[0]
                
                # Predict Econ (from bowler perspective)
                feat_econ = pd.DataFrame([{
                    'player_name': self.registry.get(bowler, bowler),
                    'venue': venue_name,
                    'pitch_type': pitch_info['pitch_type'],
                    'phase': phase,
                    'is_afternoon': env.get('is_afternoon', 0),
                    'temp_i1': env.get('temp_i1', 30.0),
                    'hum_i1': env.get('hum_i1', 65.0),
                    'dew_i1': env.get('dew_i1', 20.0),
                    'global_wkt_avg': w_stats.get('global_wkt_avg', 0.8),
                    'global_econ_avg': w_stats.get('global_econ', 8.5),
                    'venue_wkt_avg': 0.8,
                    'recent_form_economy': w_stats.get('recent_form_economy', 8.5)
                }])
                pred_econ = self.econ_model.predict(feat_econ)[0]
                
                # Heuristic for Wicket Prob in an over
                # High Econ = low wkt prob, High phase pressure = high wkt prob
                wkt_prob = (12 - pred_econ) / 10 * (w_stats.get('global_wkt_avg', 1.0))
                
                matchup_data.append({
                    "phase": phase,
                    "sr": round(pred_sr, 1),
                    "econ": round(pred_econ, 1),
                    "wkt_prob": round(min(0.99, max(0.01, wkt_prob)), 2)
                })
            
            return {
                "type": "matchup",
                "batter": batter,
                "bowler": bowler,
                "data": matchup_data
            }
            
        else:
            # Scenario: Same Role Comparison
            role = p1_role
            return {
                "type": "comparison",
                "role": role,
                "player1": {"name": p1_name, "stats": p1_stats},
                "player2": {"name": p2_name, "stats": p2_stats}
            }

    def predict_match(self, team1_xi, team2_xi, venue, env):
        # 1. Simulate Scenarios
        s1 = self.simulate_innings(team1_xi, team2_xi, venue, env)
        s2 = self.simulate_innings(team2_xi, team1_xi, venue, env)
        
        # 2. Heuristic Win Probability Calibrator
        score_diff = s1['score'] - s2['score']
        win_prob = 1 / (1 + np.exp(-0.08 * score_diff))
        
        return {
            'team1_score': int(s1['score']),
            'team2_score': int(s2['score']),
            'win_prob': float(win_prob),
            'team1_projections': s1['player_stats'],
            'team2_projections': s2['player_stats'],
            'team1_phases': s1['phases'],
            'team2_phases': s2['phases'],
            'team1_worms': s1['worms'],
            'team2_worms': s2['worms'],
            'team1_worm_wickets': s1['worm_wickets'],
            'team2_worm_wickets': s2['worm_wickets']
        }

if __name__ == "__main__":
    predictor = EnsembleMatchPredictor()
    t1 = ["RD Gaikwad", "MS Dhoni", "S Dube", "Rachin Ravindra", "RA Jadeja"]
    t2 = ["JJ Bumrah", "TA Boult", "B Kumar", "YS Chahal", "HH Pandya"]
    res = predictor.predict_match(t1, t2, "Wankhede Stadium, Mumbai", {"temp": 32, "humidity": 70, "dew_point": 22})
    print(json.dumps(res, indent=2))
