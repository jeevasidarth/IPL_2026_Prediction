from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    hum_i1: float = 70.0
    dew_i1: float = 22.0
    hum_i2: float = 85.0
    dew_i2: float = 30.0

# File Paths — backend/ is the base
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, 'data')
SQUADS_PATH = os.path.join(DATA_DIR, 'squads_2026_enriched.json')
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
        return {"teams": ["Error Loading Data"], "squads": {}, "venues": []}

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        print(f"--- Ensembling Prediction Request: {req.team1} vs {req.team2} ---")
        
        env = {
            'is_afternoon': req.is_afternoon, 
            'temp_i1': 28.0, 
            'hum_i1': req.hum_i1, 'dew_i1': req.dew_i1,
            'hum_i2': req.hum_i2, 'dew_i2': req.dew_i2
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
                "notable_performers": [{"name": n, "runs": r} for n, r in res_a['team1_highlights'][:3]],
                "bowlers": []
            },
            "scenario_b": {
                "bat_first": req.team2,
                "target": res_b['team1_score'],
                "win_prob": round(res_b['win_prob'] * 100, 1),
                "notable_performers": [{"name": n, "runs": r} for n, r in res_b['team1_highlights'][:3]],
                "bowlers": []
            }
        }
    except Exception as e:
        import traceback
        print(f"Ensemble Prediction Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
