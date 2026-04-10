import os
import json
import sys
import pandas as pd
from itertools import permutations

# Add backend directory to sys path so we can import predict_match
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from api.predict_match import EnsembleMatchPredictor

DATA_DIR = os.path.join(BASE_DIR, 'data')
SQUADS_PATH = os.path.join(DATA_DIR, 'squads_2026_enriched.json')
RESULTS_PATH = os.path.join(DATA_DIR, 'tournament_2026_results.json')

HOME_VENUES = {
    "RCB": "M Chinnaswamy Stadium",
    "CSK": "MA Chidambaram Stadium, Chepauk, Chennai",
    "DC": "Arun Jaitley Stadium",
    "SRH": "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "MI": "Wankhede Stadium, Mumbai",
    "KKR": "Eden Gardens, Kolkata",
    "PBKS": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "RR": "Sawai Mansingh Stadium",
    "GT": "Narendra Modi Stadium, Ahmedabad",
    "LSG": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow"
}

def get_best_xi(squad):
    """Simple heuristic: top 6 batters, top 5 bowlers from the list"""
    batters = squad.get("Batters", [])
    bowlers = squad.get("Bowlers", [])
    
    xi = []
    for b in batters:
        if b not in xi and len(xi) < 6:
            xi.append(b)
            
    for b in bowlers:
        if b not in xi and len(xi) < 11:
            xi.append(b)
            
    # Fallback if too few
    for b in batters + bowlers:
        if b not in xi and len(xi) < 11:
            xi.append(b)
            
    return xi

def simulate_tournament():
    print("Initializing Tournament Engine...")
    engine = EnsembleMatchPredictor()
    
    with open(SQUADS_PATH, 'r') as f:
        squads = json.load(f)
        
    teams = list(squads.keys())
    
    # Generate all home-away matches (Double Round-Robin)
    matches = list(permutations(teams, 2))
    total_matches = len(matches)
    
    # Static environment for fairness
    env = {
        'is_afternoon': 0, 
        'temp_i1': 30.0, 
        'hum_i1': 65.0, 
        'dew_i1': 20.0,
        'hum_i2': 75.0, 
        'dew_i2': 25.0
    }

    # Load real-world standings (Step 1)
    STANDINGS_PATH = os.path.join(DATA_DIR, 'current_standings.json')
    seeded_data = {}
    if os.path.exists(STANDINGS_PATH):
        with open(STANDINGS_PATH, 'r') as f:
            seeded_data = json.load(f)
            print(f"Seeding tournament from real-world 2026 state (as of {seeded_data.get('as_of')})...")

    # Initialize Table
    points_table = {}
    for t in teams:
        seed = seeded_data.get('standings', {}).get(t, {})
        # Note: We translate real NRR back into an estimated run difference for the simulation
        # NRR = (RunsFor/Overs) - (RunsAgainst/Overs)
        # We'll just seed the points/wins/losses directly and use a momentum boost.
        points_table[t] = {
            "points": seed.get("points", 0),
            "wins": seed.get("wins", 0),
            "losses": seed.get("losses", 0),
            "played": seed.get("played", 0),
            "runs_for": seed.get("played", 0) * 160 + (seed.get("nrr", 0) * seed.get("played", 0) * 20), 
            "runs_against": seed.get("played", 0) * 160,
            "nrr": seed.get("nrr", 0.0)
        }
 
    # Streak tracking for internal momentum
    streaks = {t: (3 if seeded_data.get('standings', {}).get(t, {}).get('wins', 0) >= 2 else 0) for t in teams} 

    print("Simulating remaining matches to complete the 18-game season...")
    # Target exactly 18 matches per team
    GAMES_CAP = 18
    matches_simulated = 0
    
    # We shuffle the match pool slightly to ensure a more natural distribution 
    # of "remaining" games across the schedule
    import random
    random.seed(42) # Deterministic for debugging
    random.shuffle(matches)

    for i, (home_team, away_team) in enumerate(matches):
        # Optimization: Only simulate if BOTH teams haven't finished their 18-match quota
        if points_table[home_team]["played"] >= GAMES_CAP or points_table[away_team]["played"] >= GAMES_CAP:
            continue

        venue = HOME_VENUES.get(home_team, "Wankhede Stadium, Mumbai")
        home_xi = get_best_xi(squads[home_team])
        away_xi = get_best_xi(squads[away_team])
        
        # Initial prediction
        res = engine.predict_match(home_xi, away_xi, venue, env)
        base_win_prob = res['win_prob']
        
        # Apply Season Momentum (Step 2)
        home_momentum = max(-0.15, min(0.15, streaks[home_team] * 0.03))
        away_momentum = max(-0.15, min(0.15, streaks[away_team] * 0.03))
        
        adj_win_prob = base_win_prob + (home_momentum - away_momentum)
        adj_win_prob = max(0.01, min(0.99, adj_win_prob))
        
        # Win Condition
        if adj_win_prob >= 0.5:
            winner, loser = home_team, away_team
            streaks[winner] = streaks[winner] + 1 if streaks[winner] >= 0 else 1
            streaks[loser] = streaks[loser] - 1 if streaks[loser] <= 0 else -1
        else:
            winner, loser = away_team, home_team
            streaks[winner] = streaks[winner] + 1 if streaks[winner] >= 0 else 1
            streaks[loser] = streaks[loser] - 1 if streaks[loser] <= 0 else -1
            
        points_table[winner]["wins"] += 1
        points_table[winner]["points"] += 2
        points_table[loser]["losses"] += 1
        points_table[home_team]["played"] += 1
        points_table[away_team]["played"] += 1
        
        points_table[home_team]["runs_for"] += res['team1_score']
        points_table[home_team]["runs_against"] += res['team2_score']
        points_table[away_team]["runs_for"] += res['team2_score']
        points_table[away_team]["runs_against"] += res['team1_score']

        matches_simulated += 1
        if matches_simulated % 10 == 0:
            print(f"Simulated {matches_simulated} additional matches...")

    # Calculate NRR
    # Using simple NRR: (Total Runs Scored / Overs Faced) - (Total Runs Conceded / Overs Bowled)
    # Since we assume 20 overs per innings in this static sim
    for t in teams:
        overs_played = points_table[t]["played"] * 20
        nrr = (points_table[t]["runs_for"] / overs_played) - (points_table[t]["runs_against"] / overs_played)
        points_table[t]["nrr"] = round(nrr, 3)

    # Sort Points Table
    print("Resolving Points Table...")
    sorted_teams = sorted(points_table.items(), key=lambda x: (x[1]['points'], x[1]['nrr']), reverse=True)
    
    top4 = [t[0] for t in sorted_teams[:4]]
    print(f"Playoffs Decided: {top4}")
    
    # --- PLAYOFFS ---
    def play_match(t1, t2, venue_name):
        xi1 = get_best_xi(squads[t1])
        xi2 = get_best_xi(squads[t2])
        res = engine.predict_match(xi1, xi2, venue_name, env)
        winner = t1 if res['win_prob'] >= 0.5 else t2
        loser = t2 if res['win_prob'] >= 0.5 else t1
        return winner, loser, res

    print("Simulating Playoffs...")
    neutral_venue = "Narendra Modi Stadium, Ahmedabad"
    
    # Qualifier 1: 1st vs 2nd
    print("Q1: " + top4[0] + " vs " + top4[1])
    q1_winner, q1_loser, _ = play_match(top4[0], top4[1], neutral_venue)
    
    # Eliminator: 3rd vs 4th
    print("Eliminator: " + top4[2] + " vs " + top4[3])
    elim_winner, elim_loser, _ = play_match(top4[2], top4[3], neutral_venue)
    
    # Qualifier 2: Q1 Loser vs Elim Winner
    print("Q2: " + q1_loser + " vs " + elim_winner)
    q2_winner, q2_loser, _ = play_match(q1_loser, elim_winner, neutral_venue)
    
    # Final: Q1 Winner vs Q2 Winner
    print("GRAND FINAL: " + q1_winner + " vs " + q2_winner)
    champion, runner_up, final_res = play_match(q1_winner, q2_winner, neutral_venue)
    
    print(f"*** {champion} WIN IPL 2026 ***")

    # Final Output Struct
    output = {
        "standings": [
            {
                "rank": idx + 1,
                "team": t[0],
                "played": t[1]["played"],
                "wins": t[1]["wins"],
                "losses": t[1]["losses"],
                "points": t[1]["points"],
                "nrr": t[1]["nrr"]
            } for idx, t in enumerate(sorted_teams)
        ],
        "playoffs": {
            "qualifier_1": {"team1": top4[0], "team2": top4[1], "winner": q1_winner},
            "eliminator": {"team1": top4[2], "team2": top4[3], "winner": elim_winner},
            "qualifier_2": {"team1": q1_loser, "team2": elim_winner, "winner": q2_winner},
            "final": {
                "team1": q1_winner, 
                "team2": q2_winner, 
                "winner": champion,
                "score": final_res
            }
        },
        "champion": champion
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"Saved tournament results to {RESULTS_PATH}")

if __name__ == "__main__":
    simulate_tournament()
