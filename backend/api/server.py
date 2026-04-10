from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os
import json

# Add api directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict_match import EnsembleMatchPredictor

app = FastAPI(title="IPL 2026 Prediction API")

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hybrid Ensemble Engine
engine = EnsembleMatchPredictor()

class PredictRequest(BaseModel):
    team1: str
    team2: str
    team1_xi: list[str] = []
    team2_xi: list[str] = []
    venue: str
    is_afternoon: int = 0
    hum_i1: Optional[float] = None
    dew_i1: Optional[float] = None
    hum_i2: Optional[float] = None
    dew_i2: Optional[float] = None

# File Paths — backend/ is the base
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, 'data')
SQUADS_PATH = os.path.join(DATA_DIR, 'squads_2026_enriched.json')
TOURNAMENT_RESULTS_PATH = os.path.join(DATA_DIR, 'tournament_2026_results.json')
PITCH_MAPPING_PATH = os.path.join(DATA_DIR, 'stadium_pitch_mapping.json')

@app.get("/metadata")
async def get_metadata():
    """Returns available teams, full squads, and venues."""
    try:
        with open(SQUADS_PATH, 'r') as f:
            squads = json.load(f)
        with open(PITCH_MAPPING_PATH, 'r') as f:
            pitches = json.load(f)
        
        return {
            "teams": sorted(list(squads.keys())),
            "squads": squads,
            "venues": sorted(list(pitches.keys()))
        }
    except Exception as e:
        print(f"Metadata Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tournament-results")
async def get_tournament_results():
    """Returns the pre-calculated tournament simulation results."""
    try:
        if not os.path.exists(TOURNAMENT_RESULTS_PATH):
            return {"status": "error", "message": "Tournament results not generated yet."}
        
        with open(TOURNAMENT_RESULTS_PATH, 'r') as f:
            data = json.load(f)
        return {"status": "success", "results": data}
    except Exception as e:
        print(f"Tournament Results Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class CompareRequest(BaseModel):
    player1: str
    player2: str
    venue: str
    hum_i1: Optional[float] = None
    dew_i1: Optional[float] = None

@app.post("/compare-players")
async def compare_players(req: CompareRequest):
    try:
        env = {
            'is_afternoon': 0, 
            'temp_i1': 28.0, 
            'hum_i1': req.hum_i1 if req.hum_i1 is not None else 65.0,
            'dew_i1': req.dew_i1 if req.dew_i1 is not None else 20.0
        }
        res = engine.compare_players(req.player1, req.player2, req.venue, env)
        return {"status": "success", "comparison": res}
    except Exception as e:
        print(f"Comparison Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        print(f"--- Ensembling Prediction Request: {req.team1} vs {req.team2} ---")
        
        env = {
            'is_afternoon': req.is_afternoon, 
            'temp_i1': 28.0, 
            'hum_i1': req.hum_i1 if req.hum_i1 is not None else 65.0,
            'dew_i1': req.dew_i1 if req.dew_i1 is not None else 20.0,
            'hum_i2': req.hum_i2 if req.hum_i2 is not None else 75.0,
            'dew_i2': req.dew_i2 if req.dew_i2 is not None else 25.0
        }

        # Scenario A: Team 1 Bats First
        res_a = engine.predict_match(req.team1_xi, req.team2_xi, req.venue, env)
        
        # Scenario B: Team 2 Bats First
        res_b = engine.predict_match(req.team2_xi, req.team1_xi, req.venue, env)

        return {
            "status": "success",
            "scenario_a": {
                "bat_first": req.team1,
                "target": res_a['team1_score'],
                "win_prob": round(res_a['win_prob'] * 100, 1),
                "projections": res_a['team1_projections'],
                "phase_stats": res_a['team1_phases'],
                "worm_data": res_a['team1_worms'],
                "worm_wickets": res_a['team1_worm_wickets']
            },
            "scenario_b": {
                "bat_first": req.team2,
                "target": res_b['team1_score'],
                "win_prob": round(res_b['win_prob'] * 100, 1),
                "projections": res_b['team2_projections'],
                "phase_stats": res_b['team1_phases'],
                "worm_data": res_b['team1_worms'],
                "worm_wickets": res_b['team1_worm_wickets']
            }
        }
    except Exception as e:
        import traceback
        print(f"Ensemble Prediction Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart-pick")
async def smart_pick(req: PredictRequest):
    try:
        # Load Data using internal constants
        data_lookup_path = os.path.join(DATA_DIR, 'player_stats_lookup.json')
        
        with open(PITCH_MAPPING_PATH, 'r') as f:
            pitch_map = json.load(f)
        with open(SQUADS_PATH, 'r') as f:
            squads = json.load(f)
        with open(data_lookup_path, 'r') as f:
            stats_lookup = json.load(f)

        # 1. Resolve Pitch Type
        venue_info = pitch_map.get(req.venue, {"pitch_type": "Balanced"})
        p_type = venue_info["pitch_type"]

        # 2. Get Squad
        team_squad = squads.get(req.team1, {"Batters": [], "Bowlers": []})
        
        # 3. Score Batters
        scored_batters = []
        for p in team_squad["Batters"]:
            p_stats = stats_lookup["batters"].get(p, {})
            p_avg = p_stats.get("pitch_avgs", {}).get(p_type, p_stats.get("global_avg", 20))
            p_sr = p_stats.get("recent_form_sr", 130)
            
            # Weighted Score: Pitch Avg (60%) + Recent SR (40%)
            score = (p_avg / 30) * 0.6 + (p_sr / 150) * 0.4
            scored_batters.append({"name": p, "score": score})
        
        scored_batters.sort(key=lambda x: x["score"], reverse=True)
        top_batters = [b["name"] for b in scored_batters[:6]]

        # 4. Score Bowlers (Avoid duplicates from Batters if any)
        scored_bowlers = []
        for p in team_squad["Bowlers"]:
            if p in top_batters: continue
            
            p_stats = stats_lookup["batters"].get(p, {}) # Lookup stores some bowlers in 'batters'
            
            # Simple scoring for bowlers: Lower recent_form_economy is better
            econ = p_stats.get("recent_form_economy", 8.5)
            score = 15 - econ 
            scored_bowlers.append({"name": p, "score": score})

        scored_bowlers.sort(key=lambda x: x["score"], reverse=True)
        top_bowlers = [b["name"] for b in scored_bowlers[:5]]

        return {
            "status": "success",
            "team": req.team1,
            "venue": req.venue,
            "pitch_type": p_type,
            "xi": top_batters + top_bowlers
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for deployment (default to 8000 for local)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
