import draft_tokenizer

t = draft_tokenizer.DraftTokenizer.load_from_files()

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



# Just test the first document
sample_sequence = docs[0]

# Tokenize the sequence
encoded = t.encode_sequence(sample_sequence)

# Print tokenized results
for token in encoded[:10]:  # just first 10
    print(token)


output = t(docs[0], return_tensors='pt')
for k, v in output.items():
    print(f"{k}: {v.shape} -> {v}")
