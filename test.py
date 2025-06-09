from draft_tokenizer import DraftTokenizer

t  = DraftTokenizer.load_from_files()

import torch
from transformers import BertConfig
from draft_bert import DraftBertForMaskedLM 

config = BertConfig.from_pretrained("./checkpoints/checkpoint-85000")
model = DraftBertForMaskedLM(config)
model.load_state_dict(torch.load("./checkpoints/checkpoint-85000/pytorch_model.bin"))


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


encoded_games = [t(actions, return_tensors='pt') for actions in docs]


from draft_eval import evaluate_model_detailed

df = evaluate_model_detailed(model,encoded_games)

df
df.groupby("predicting_position")[["top1_correct", "top3_correct", "top5_correct", "top10_correct"]].mean()
