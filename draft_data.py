import os
import json

matches_dir = "matches"
output_file = "draft_results.json"

draft_data = {}

for filename in os.listdir(matches_dir):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(matches_dir, filename)

    with open(file_path, "r") as f:
        try:
            match = json.load(f)
            match_id = str(match.get("match_id"))  # use str key to keep JSON valid
            picks_bans = match.get("picks_bans", [])

            if picks_bans:  # only include matches that have drafting info
                draft_data[match_id] = picks_bans

        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

# Save the cleaned draft data
with open(output_file, "w") as out_f:
    json.dump(draft_data, out_f, indent=2)

print(f"✅ Saved draft data for {len(draft_data)} matches to {output_file}")




import json

# Path to your file (adjust this if needed)
file_path = 'draft_results.json'

# Load the data
with open(file_path, 'r') as f:
    draft_data = json.load(f)

# Show sample structure
for match_id, sequence in list(draft_data.items())[:3]:
    print(f"Match ID: {match_id}")
    for action in sequence[:5]:  # show first 5 actions
        print(action)
    print("-" * 40)

docs = [sequence for match_id, sequence in list(draft_data.items())]

