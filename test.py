import torch
import json
from transformers import BertConfig, GPT2Config
from draft_bert import DraftBertForMaskedLM
from draft_bert import DraftGPT2ForCausalLM  # assuming you created this
from draft_tokenizer import DraftTokenizer
from draft_eval import evaluate_model_detailed
from draft_dataset import split_tokenized_docs
import os

os.makedirs("eval", exist_ok=True)

# === 🧠 Load Tokenizer ===
t = DraftTokenizer.load_from_files()

# === 📜 Load Data ===
with open('draft_results.json', 'r') as f:
    draft_data = json.load(f)

docs = [seq for _, seq in draft_data.items()]
tokenized_docs = [t(doc, return_tensors='pt') for doc in docs]

# === ✂️ Split for Evaluation ===
train_docs, val_docs, test_docs = split_tokenized_docs(
    tokenized_docs,
    tokenizer=t,
    mid=True
)
btrain_docs, bval_docs, btest_docs = split_tokenized_docs(tokenized_docs, tokenizer=t, mlm=True)
gtrain_docs, gval_docs, gtest_docs = split_tokenized_docs(tokenized_docs, tokenizer=t, mlm=False)

# === 📦 Load Trained Models ===
# BERT (Masked Language Model)
bert_config = BertConfig.from_pretrained("./model/final_bert_model")
bmodel = DraftBertForMaskedLM(bert_config)
bmodel.load_state_dict(torch.load("./model/final_bert_model/pytorch_model.bin"))
bmodel.eval()

# GPT2 (Causal Language Model)
gpt_config = GPT2Config.from_pretrained("./model/final_gpt2_model")
gmodel = DraftGPT2ForCausalLM(gpt_config)
gmodel.load_state_dict(torch.load("./model/final_gpt2_model/pytorch_model.bin"))
gmodel.eval()

# === 🧪 Evaluate BERT ===
df = evaluate_model_detailed(bmodel, btrain_docs, mode='masked',tokenizer=t)
df.to_csv('eval/bmodel_train.csv')
df.groupby("predicting_position")[["top1_correct", "top3_correct", "top5_correct", "top10_correct"]].mean().to_csv('eval/btrain_rez.csv')

df = evaluate_model_detailed(bmodel, btest_docs, mode='masked',tokenizer=t)
df.to_csv('eval/bmodel_test.csv')
df.groupby("predicting_position")[["top1_correct", "top3_correct", "top5_correct", "top10_correct"]].mean().to_csv('eval/btest_rez.csv')

# === 🧪 Evaluate GPT2 ===
df = evaluate_model_detailed(gmodel, gtrain_docs, mode='causal',tokenizer=t)
df.to_csv('eval/gmodel_train.csv')
df.groupby("predicting_position")[["top1_correct", "top3_correct", "top5_correct", "top10_correct"]].mean().to_csv('eval/gtrain_rez.csv')

df = evaluate_model_detailed(gmodel, gtest_docs, mode='causal',tokenizer=t)
df.to_csv('eval/gmodel_test.csv')
df.groupby("predicting_position")[["top1_correct", "top3_correct", "top5_correct", "top10_correct"]].mean().to_csv('eval/gtest_rez.csv')


"""
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
"""