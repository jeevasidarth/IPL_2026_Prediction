import csv
import json
from collections import defaultdict
from collections import Counter
import os

ENV_DATA = 'ipl_2019_2025_environmental_final_v2.csv'
BALL_DATA = 'archive (3)/IPL.csv'
BATTER_PERF = 'batter_performance_comprehensive.csv'
BOWLER_PERF = 'bowler_performance_comprehensive.csv'
H2H_DATA = 'batter_vs_bowler_h2h.json'

OUTPUT_FEATURE_MATRIX = 'ipl_match_features_training.csv'

print("Loading historical performance data...")

# Load Batter pre-match form dictionary
# Key: (player_name, match_id) -> {'form_avg': x, 'form_sr': y}
batter_form = {}
with open(BATTER_PERF, 'r') as f:
    for row in csv.DictReader(f):
        batter_form[(row['player_name'], row['match_id'])] = {
            'form_avg': float(row['recent_form_avg']),
            'form_sr': float(row['recent_form_sr'])
        }

# Load Bowler pre-match form dictionary
bowler_form = {}
with open(BOWLER_PERF, 'r') as f:
    for row in csv.DictReader(f):
        bowler_form[(row['player_name'], row['match_id'])] = {
            'form_wickets': float(row['recent_form_wickets']),
            'form_economy': float(row['recent_form_economy'])
        }

# Pre-compute Match Participants from ball-by-ball
# identify top 6 batters and top 5 bowlers per team per match
print("Identifying match participants (Top 6 batters, Top 5 bowlers)...")
match_participants = defaultdict(lambda: defaultdict(lambda: {'batters': [], 'bowlers': []}))
batter_counts = defaultdict(lambda: defaultdict(Counter))
bowler_counts = defaultdict(lambda: defaultdict(Counter))

with open(BALL_DATA, 'r') as f:
    raw_reader = csv.reader(f)
    header = [h.strip() for h in next(raw_reader)]
    f.seek(0)
    next(f)
    reader = csv.DictReader(f, fieldnames=header)
    
    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items()}
        m_id = row['match_id']
        bat_team = row['batting_team']
        bowl_team = row['bowling_team']
        
        # Order of appearance proxy: just count balls faced/bowled. Top 6 most balls faced = top 6.
        batter_counts[m_id][bat_team][row['batter']] += 1
        bowler_counts[m_id][bowl_team][row['bowler']] += 1

# Resolve top participants
for m_id in batter_counts:
    for team in batter_counts[m_id]:
        # Top 6 batters by balls faced
        top_bats = [b for b, c in batter_counts[m_id][team].most_common(6)]
        match_participants[m_id][team]['batters'] = top_bats

for m_id in bowler_counts:
    for team in bowler_counts[m_id]:
        # Top 5 bowlers by balls bowled
        top_bowls = [b for b, c in bowler_counts[m_id][team].most_common(5)]
        match_participants[m_id][team]['bowlers'] = top_bowls

print("Building final feature matrix...")
features = []

def safe_mean(vals):
    return float(sum(vals) / len(vals)) if len(vals) > 0 else 0.0

# Load H2H data
with open(H2H_DATA, 'r') as f:
    h2h_data = json.load(f)

def get_h2h_advantage(batters, bowlers):
    total_runs = 0
    total_balls = 0
    total_outs = 0
    for bat in batters:
        bat_h2h = h2h_data.get(bat, {})
        for bowl in bowlers:
            stats = bat_h2h.get(bowl)
            if stats:
                total_runs += stats['runs']
                total_balls += stats['balls']
                total_outs += stats['outs']
    
    runs_per_ball = total_runs / total_balls if total_balls > 0 else 1.2 # average T20 SR ~120
    runs_per_out = total_runs / total_outs if total_outs > 0 else 25.0 # average T20 Avg ~25
    return runs_per_ball, runs_per_out

with open(ENV_DATA, 'r') as f:
    env_reader = csv.DictReader(f)
    
    for row in env_reader:
        m_id = row['match_id']
        
        # Determine Teams. 
        # Env data doesn't explicitly guarantee "team1" vs "team2" columns, except winner & toss_winner.
        # We need to find the two teams from match_participants
        teams_in_match = list(match_participants[m_id].keys())
        if len(teams_in_match) != 2: continue
        
        team1 = teams_in_match[0]
        team2 = teams_in_match[1]
        
        def safe_float(val, default=0.0):
            try:
                if val == 'NA' or not val:
                    return default
                return float(val)
            except ValueError:
                return default
        
        # Determine who batted first
        toss_winner = row['toss_winner']
        toss_decision = row['toss_decision'].lower()
        team1_bat_first = 0
        if toss_winner == team1:
            if toss_decision == 'bat': team1_bat_first = 1
            else: team1_bat_first = 0
        else:
            if toss_decision == 'bat': team1_bat_first = 0
            else: team1_bat_first = 1

        feature_row = {
            'match_id': m_id,
            'year': row['year'],
            'venue': row['venue'],
            'pitch_type': row['pitch_type'],
            'ground_size': row['ground_size'],
            'team1_bat_first': team1_bat_first, # New Scenario-based feature
            'is_afternoon': row['is_afternoon'],
            'winner_is_team1': 1 if row['winner'] == team1 else 0,
            
            # Env stats
            'dew_factor': safe_float(row['dew_i2']) - safe_float(row['dew_i1']),
            'temp_i1': safe_float(row['temp_i1'], 25),
            'hum_i1': safe_float(row['hum_i1'], 50),
        }
        
        # Lineups for H2H
        t1_bats = match_participants[m_id][team1]['batters']
        t1_bowls = match_participants[m_id][team1]['bowlers']
        t2_bats = match_participants[m_id][team2]['batters']
        t2_bowls = match_participants[m_id][team2]['bowlers']

        # H2h features
        t1_vs_t2_rp_ball, t1_vs_t2_rp_out = get_h2h_advantage(t1_bats, t2_bowls)
        t2_vs_t1_rp_ball, t2_vs_t1_rp_out = get_h2h_advantage(t2_bats, t1_bowls)
        
        feature_row['team1_h2h_rp_ball'] = t1_vs_t2_rp_ball
        feature_row['team2_h2h_rp_ball'] = t2_vs_t1_rp_ball
        
        # Batting strength
        for t_label, t_name in [('team1', team1), ('team2', team2)]:
            t_bats = match_participants[m_id][t_name]['batters']
            t_bowls = match_participants[m_id][t_name]['bowlers']
            
            # Fetch pre-match form (Now weighted EWMA)
            bat_avgs = []
            bat_srs = []
            for b in t_bats:
                f_data = batter_form.get((b, m_id))
                if f_data:
                    bat_avgs.append(f_data['form_avg'])
                    bat_srs.append(f_data['form_sr'])
                    
            bowl_wicks = []
            bowl_econs = []
            for b in t_bowls:
                f_data = bowler_form.get((b, m_id))
                if f_data:
                    bowl_wicks.append(f_data['form_wickets'])
                    bowl_econs.append(f_data['form_economy'])
            
            feature_row[f'{t_label}_avg_bat_form'] = safe_mean(bat_avgs)
            feature_row[f'{t_label}_avg_bat_sr_form'] = safe_mean(bat_srs)
            feature_row[f'{t_label}_avg_bowl_wickets_form'] = safe_mean(bowl_wicks)
            feature_row[f'{t_label}_avg_bowl_econ_form'] = safe_mean(bowl_econs)
            
        # TOSS/SCENARIO INTERACTIONS
        # Help the model learn that chasing is better when dew is high
        feature_row['bat_first_dew_interaction'] = feature_row['team1_bat_first'] * feature_row['dew_factor']
        # Interaction with pitch type (encoded value) - assumes pitch_type mapping is available
        feature_row['bat_first_pitch_interaction'] = feature_row['team1_bat_first'] * (1 if feature_row['pitch_type'] == 'Batting' else 0)
        
        features.append(feature_row)

print("Calculating 1st innings score targets...")
# We need 1st innings total runs from BALL_DATA
first_innings_scores = defaultdict(int)
with open(BALL_DATA, 'r') as f:
    raw_reader = csv.reader(f)
    header = [h.strip() for h in next(raw_reader)]
    f.seek(0)
    next(f)
    reader = csv.DictReader(f, fieldnames=header)
    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items()}
        if row.get('innings', '') == '1':
            rt = int(float(row.get('runs_total', 0) or 0))
            first_innings_scores[row['match_id']] += rt

for f in features:
    f['first_innings_score'] = first_innings_scores.get(f['match_id'], 160)

# Write out
if len(features) > 0:
    keys = features[0].keys()
    with open(OUTPUT_FEATURE_MATRIX, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(features)
    print(f"SUCCESS: Generated {len(features)} feature rows spanning {len(keys)} columns.")
else:
    print("Warning: No features generated")
