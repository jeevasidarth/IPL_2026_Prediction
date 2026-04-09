import csv
import json
from collections import defaultdict
from datetime import datetime

# Files
INPUT_CSV = r"archive (3)/IPL.csv"
SQUAD_CSV = "ipl_2026_squads.csv"
PITCH_MAPPING = "scripts/stadium_pitch_mapping.json"
OUTPUT_FILE = "batter_performance_comprehensive.csv"

# Load Squads to focus on current players if needed (but we'll do all)
# Load Pitch Mapping
with open(PITCH_MAPPING, 'r') as f:
    pitch_map = json.load(f)

def get_pitch_type(venue):
    if not venue: return "Balanced"
    for k, v in pitch_map.items():
        if k in venue or venue in k:
            return v['pitch_type']
    return "Balanced"

print("Step 1: Aggregating Ball-by-Ball data...")

# batter -> match_id -> stats
batter_match_stats = defaultdict(lambda: defaultdict(lambda: {
    'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 
    'is_out': 0, 'dismissal_bowler': 'NA', 'date': '', 'venue': '', 'pitch': ''
}))

# batter -> bowler -> stats (Head-to-Head)
h2h_stats = defaultdict(lambda: defaultdict(lambda: {'runs': 0, 'balls': 0, 'outs': 0}))

with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    # Use fieldnames from first row and strip any whitespace
    raw_reader = csv.reader(f)
    header = [h.strip() for h in next(raw_reader)]
    
    # Re-initialize reader with clean header
    f.seek(0)
    next(f) # skip header
    reader = csv.DictReader(f, fieldnames=header)
    
    for i, row in enumerate(reader):
        # Safety check: Strip all keys and values
        row = {k.strip(): (v.strip() if v else v) for k, v in row.items()}
        
        match_id = row.get('match_id')
        batter = row.get('batter')
        bowler = row.get('bowler')
        
        runs_val = row.get('runs_batter', '0')
        runs = int(runs_val) if str(runs_val).isdigit() else 0
        
        # Match Stats
        stats = batter_match_stats[batter][match_id]
        stats['runs'] += runs
        stats['balls'] += 1
        if runs == 4: stats['fours'] += 1
        if runs == 6: stats['sixes'] += 1
        stats['date'] = row.get('date', '')
        stats['venue'] = row.get('venue', '')
        
        # Head to Head
        h2h = h2h_stats[batter][bowler]
        h2h['runs'] += runs
        h2h['balls'] += 1
        
        # Wicket check: player_dismissed contains the batter's name if they are out
        dismissed = row.get('player_dismissed', '')
        if dismissed == batter:
            # Check if it was a bowler's wicket (exclude run out, etc.)
            w_type = row.get('wicket_type', '').lower()
            if w_type and w_type not in ['run out', 'retired hurt', 'obstructing the field', 'retired out']:
                stats['is_out'] = 1
                stats['dismissal_bowler'] = bowler
                h2h['outs'] += 1
        
        if i % 100000 == 0:
            print(f"Processed {i} balls...")

print("Step 2: Processing match history for each batter...")

final_rows = []
for batter, matches in batter_match_stats.items():
    # Sort matches by date
    sorted_match_ids = sorted(matches.keys(), key=lambda x: matches[x]['date'])
    
    match_history = []
    for m_id in sorted_match_ids:
        m = matches[m_id]
        
        # Calculate Weighted Rolling Form (Last 5 innings)
        # Weights for [M-5, M-4, M-3, M-2, M-1] = [0.1, 0.15, 0.2, 0.25, 0.3]
        last_5 = match_history[-5:]
        n = len(last_5)
        if n > 0:
            weights = [0.1, 0.15, 0.2, 0.25, 0.3][-n:]
            # Normalize weights if less than 5 matches
            total_w = sum(weights)
            norm_weights = [w/total_w for w in weights]
            
            recent_avg = sum(x['runs'] * w for x, w in zip(last_5, norm_weights))
            recent_sr = sum((x['runs']/x['balls']*100 if x['balls'] > 0 else 0) * w for x, w in zip(last_5, norm_weights))
        else:
            recent_avg = 0
            recent_sr = 0
        
        row = {
            'player_name': batter,
            'match_id': m_id,
            'date': m['date'],
            'venue': m['venue'],
            'pitch_type': get_pitch_type(m['venue']),
            'runs': m['runs'],
            'balls': m['balls'],
            'strike_rate': round(m['runs'] / m['balls'] * 100, 2) if m['balls'] > 0 else 0,
            'is_out': m['is_out'],
            'dismissal_bowler': m['dismissal_bowler'],
            'recent_form_avg': round(recent_avg, 2),
            'recent_form_sr': round(recent_sr, 2),
            'innings_count': len(match_history) + 1
        }
        
        final_rows.append(row)
        match_history.append(m)

# Save
fieldnames = ['player_name', 'match_id', 'date', 'venue', 'pitch_type', 'runs', 'balls', 'strike_rate', 'is_out', 'dismissal_bowler', 'recent_form_avg', 'recent_form_sr', 'innings_count']
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_rows)

print(f"Dataset finished! {len(final_rows)} match-level records saved to {OUTPUT_FILE}")

# Also save H2H for easier lookup later
with open('batter_vs_bowler_h2h.json', 'w') as f:
    # Convert defaultdict to normal dict for JSON
    json.dump({k: dict(v) for k, v in h2h_stats.items()}, f)
