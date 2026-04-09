import csv

input_file = "ipl_2026_squads_raw.csv"
output_file = "ipl_2026_squads.csv"

# Comprehensive Player Styles (Common Knowledge Enrichment for IPL 2026 stars)
STYLES = {
    # CSK
    'Ruturaj Gaikwad': 'RHB', 'MS Dhoni': 'RHB', 'Sanju Samson': 'RHB', 'Shivam Dube': 'LHB / Pacer',
    'Khaleel Ahmed': 'L-Pacer', 'Noor Ahmad': 'L-Spinner', 'Rahul Chahar': 'R-Spinner',
    # MI
    'Rohit Sharma': 'RHB', 'Suryakumar Yadav': 'RHB', 'Hardik Pandya': 'RHB / Pacer', 'Jasprit Bumrah': 'R-Pacer',
    'Trent Boult': 'L-Pacer', 'Quinton de Kock': 'LHB', 'Tilak Varma': 'LHB', 'Shardul Thakur': 'R-Pacer',
    # RCB
    'Virat Kohli': 'RHB', 'Phil Salt': 'RHB', 'Rajat Patidar': 'RHB', 'Bhuvneshwar Kumar': 'R-Pacer',
    'Josh Hazlewood': 'R-Pacer', 'Krunal Pandya': 'LHB / Spinner', 'Tim David': 'RHB',
    # KKR
    'Cameron Green': 'RHB / Pacer', 'Rinku Singh': 'LHB', 'Sunil Narine': 'LHB / Spinner', 
    'Andre Russell': 'RHB / Pacer', 'Matheesha Pathirana': 'R-Pacer', 'Varun Chakravarthy': 'R-Spinner',
    # SRH
    'Heinrich Klaasen': 'RHB', 'Travis Head': 'LHB', 'Abhishek Sharma': 'LHB / Spinner',
    'Pat Cummins': 'R-Pacer', 'Harshal Patel': 'R-Pacer', 'Liam Livingstone': 'RHB / Spinner',
    # DC
    'KL Rahul': 'RHB', 'Axar Patel': 'LHB / Spinner', 'Tristan Stubbs': 'RHB', 'Kuldeep Yadav': 'L-Spinner',
    'Mitchell Starc': 'L-Pacer', 'T. Natarajan': 'L-Pacer',
    # LSG
    'Rishabh Pant': 'LHB', 'Nicholas Pooran': 'LHB', 'Mitchell Marsh': 'RHB / Pacer',
    'Mohammad Shami': 'R-Pacer', 'Mayank Yadav': 'R-Pacer', 'Wanindu Hasaranga': 'R-Spinner',
    # RR
    'Yashasvi Jaiswal': 'LHB', 'Riyan Parag': 'RHB / Spinner', 'Ravindra Jadeja': 'LHB / Spinner',
    'Sam Curran': 'LHB / Pacer', 'Jofra Archer': 'R-Pacer', 'Shimron Hetmyer': 'LHB',
    # GT
    'Shubman Gill': 'RHB', 'Jos Buttler': 'RHB', 'Rashid Khan': 'R-Spinner', 'Mohammed Siraj': 'R-Pacer',
    'Kagiso Rabada': 'R-Pacer', 'Washington Sundar': 'LHB / Spinner',
    # PBKS
    'Shreyas Iyer': 'RHB', 'Marcus Stoinis': 'RHB / Pacer', 'Arshdeep Singh': 'L-Pacer',
    'Yuzvendra Chahal': 'R-Spinner', 'Marco Jansen': 'L-Pacer', 'Lockie Ferguson': 'R-Pacer',
}

enriched_data = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['player_name']
        role = row['role']
        current_type = row['type']
        
        # Priority 1: Use our style dictionary
        if name in STYLES:
            row['type'] = STYLES[name]
        # Priority 2: Use existing type if it's not NA or Pacer (default)
        elif current_type != 'NA' and current_type != 'Pacer':
            pass
        # Priority 3: Sensible defaults based on Role
        else:
            if role in ['Batter', 'WK-Batter']:
                row['type'] = 'RHB' # Default to RHB
            elif role == 'Bowler':
                row['type'] = 'Pacer' # Maintain pacer default for bowlers
            elif role == 'All-Rounder':
                row['type'] = 'Balanced'

        enriched_data.append(row)

# Save Final
fieldnames = ['team', 'player_name', 'role', 'type']
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(enriched_data)

print(f"Dataset enriched! Saved to {output_file}")
