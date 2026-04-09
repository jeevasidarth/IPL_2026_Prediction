import csv
import json
from collections import defaultdict

# Files
INPUT_CSV = r"archive (3)/IPL.csv"
PITCH_MAPPING = "scripts/stadium_pitch_mapping.json"
OUTPUT_FILE = "bowler_performance_comprehensive.csv"

with open(PITCH_MAPPING, 'r') as f:
    pitch_map = json.load(f)

def get_pitch_type(venue):
    if not venue: return "Balanced"
    for k, v in pitch_map.items():
        if k in venue or venue in k:
            return v['pitch_type']
    return "Balanced"

print("Step 1: Aggregating Bowler Ball-by-Ball data...")

# bowler -> match_id -> stats
bowler_match_stats = defaultdict(lambda: defaultdict(lambda: {
    'runs': 0, 'balls': 0, 'wickets': 0, 'dots': 0,
    'date': '', 'venue': '', 'pitch': ''
}))

with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    raw_reader = csv.reader(f)
    header = [h.strip() for h in next(raw_reader)]
    
    f.seek(0)
    next(f)
    reader = csv.DictReader(f, fieldnames=header)
    
    for i, row in enumerate(reader):
        row = {k.strip(): v.strip() for k, v in row.items()}
        match_id = row['match_id']
        bowler = row['bowler']
        
        # Runs & Extras
        bowler_runs = int(row.get('runs_bowler', 0) if str(row.get('runs_bowler')).isdigit() else 0)
        valid_ball = str(row.get('valid_ball', '1')) == '1'
        
        stats = bowler_match_stats[bowler][match_id]
        stats['runs'] += bowler_runs
        
        # Balls
        if valid_ball:
            stats['balls'] += 1
            if bowler_runs == 0:
                stats['dots'] += 1
        
        # Wickets (excluding certain types)
        w_type = row.get('wicket_kind', '').lower()
        if w_type and w_type not in ['run out', 'retired hurt', 'obstructing the field', 'retired out']:
            stats['wickets'] += 1
            
        stats['date'] = row.get('date', '')
        stats['venue'] = row.get('venue', '')
        
        if i % 100000 == 0:
            print(f"Processed {i} balls...")

print("Step 2: Processing match history for each bowler...")

final_rows = []
for bowler, matches in bowler_match_stats.items():
    sorted_match_ids = sorted(matches.keys(), key=lambda x: matches[x]['date'])
    
    match_history = []
    for m_id in sorted_match_ids:
        m = matches[m_id]
        
        # Calculate Weighted Rolling Form (Last 5 matches)
        # Weights for [M-5, M-4, M-3, M-2, M-1] = [0.1, 0.15, 0.2, 0.25, 0.3]
        last_5 = match_history[-5:]
        n = len(last_5)
        if n > 0:
            weights = [0.1, 0.15, 0.2, 0.25, 0.3][-n:]
            total_w = sum(weights)
            norm_weights = [w/total_w for w in weights]
            
            recent_wickets = sum(x['wickets'] * w for x, w in zip(last_5, norm_weights))
            recent_econ = sum((x['runs'] / (x['balls']/6) if x['balls'] > 0 else 9.0) * w for x, w in zip(last_5, norm_weights))
        else:
            recent_wickets = 0
            recent_econ = 0
        
        overs = f"{m['balls'] // 6}.{m['balls'] % 6}"
        economy = round(m['runs'] / (m['balls'] / 6), 2) if m['balls'] > 0 else 0
        
        row = {
            'player_name': bowler,
            'match_id': m_id,
            'date': m['date'],
            'venue': m['venue'],
            'pitch_type': get_pitch_type(m['venue']),
            'overs': overs,
            'runs_conceded': m['runs'],
            'wickets': m['wickets'],
            'economy': economy,
            'dots': m['dots'],
            'recent_form_wickets': round(recent_wickets, 2),
            'recent_form_economy': round(recent_econ, 2),
            'match_count': len(match_history) + 1
        }
        
        final_rows.append(row)
        match_history.append(m)

# Save
fieldnames = ['player_name', 'match_id', 'date', 'venue', 'pitch_type', 'overs', 'runs_conceded', 'wickets', 'economy', 'dots', 'recent_form_wickets', 'recent_form_economy', 'match_count']
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_rows)

print(f"Dataset finished! {len(final_rows)} bowler match-records saved to {OUTPUT_FILE}")
