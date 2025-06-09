import os
import json
import pickle
from sklearn.model_selection import train_test_split
from draft_tokenizer import DraftTokenizer
from draft_dataset import DraftMLMDataset, DraftCLMDataset
from transformers import BertConfig
from draft_bert import DraftBertForMaskedLM
import torch

# === 📦 Setup Paths ===
os.makedirs("cache", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# === 🧠 Load Tokenizer ===
tokenizer = DraftTokenizer.load_from_files()

# === 📂 Load Draft Data ===
with open("draft_results.json", "r") as f:
    raw_docs = json.load(f)

docs = [sequence for _, sequence in raw_docs.items()]
tokenized_docs = [tokenizer(doc, return_tensors='pt') for doc in docs]

# === 🧪 Train/Val/Test Split ===
train_docs, test_docs = train_test_split(tokenized_docs, test_size=0.15, random_state=123)
train_docs, val_docs = train_test_split(train_docs, test_size=0.15, random_state=123)

# === 💾 Save as Pickle ===
with open("cache/train.pkl", "wb") as f: pickle.dump(train_docs, f)
with open("cache/val.pkl", "wb") as f: pickle.dump(val_docs, f)
with open("cache/test.pkl", "wb") as f: pickle.dump(test_docs, f)

# === 📚 Create Datasets ===
mlm_train_dataset = DraftMLMDataset(train_docs, tokenizer)
mlm_val_dataset = DraftMLMDataset(val_docs, tokenizer)
clm_train_dataset = DraftCLMDataset(train_docs)
clm_val_dataset = DraftCLMDataset(val_docs)

# === 🧱 Config Shared Model ===
config = BertConfig(
    vocab_size=tokenizer.vocab_size(),
    hidden_size=128,
    num_attention_heads=4,
    num_hidden_layers=2,
    intermediate_size=512,
    max_position_embeddings=32,
    pad_token_id=tokenizer.pad_token_id,
    cls_token_id=tokenizer.cls_token_id,
    sep_token_id=tokenizer.sep_token_id,
    mask_token_id=tokenizer.mask_token_id,
)
config.add_pooling_layer = True

# === 🧠 Init Models ===
mlm_model = DraftBertForMaskedLM(config, is_causal=False)
clm_model = DraftBertForMaskedLM(config, is_causal=True)

# === 💾 Save Init Models ===
torch.save(mlm_model.state_dict(), "checkpoints/mlm_model.pt")
torch.save(clm_model.state_dict(), "checkpoints/clm_model.pt")

print("✅ Setup complete. Datasets pickled. Models initialized and saved.")
