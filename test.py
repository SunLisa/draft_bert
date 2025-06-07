from draft_tokenizer import DraftTokenizer

t  = DraftTokenizer.load_from_files()


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
encoded = t(sample_sequence)


output = t(docs[0], return_tensors='pt')
for k, v in output.items():
    print(f"{k}: {v.shape} -> {v}")

from transformers import BertConfig

config = BertConfig(
    vocab_size=256,             # set to match your tokenizer.vocab_size()
    hidden_size=128,
    num_attention_heads=4,
    num_hidden_layers=2,
    intermediate_size=512,
    max_position_embeddings=32,
    pad_token_id=252,
    # Optional but good practice:
    cls_token_id=254,
    sep_token_id=255,
    mask_token_id=253,
)
config.add_pooling_layer = True

encoded_games = [t(actions, return_tensors='pt') for actions in docs]


from draft_dataset import DraftMLMDataset

dataset = DraftMLMDataset(encoded_games,t)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=t,
    mlm=True,
    mlm_probability=0.15
)

from draft_bert import DraftBertForMaskedLM
model = DraftBertForMaskedLM(config)




from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./checkpoints",                 # ✅ Save location
    overwrite_output_dir=True,                  # ✅ Overwrite if exists
    num_train_epochs=5000,                        # ✅ Your plan
    per_device_train_batch_size=8,              # ✅ Tweak based on memory
    logging_dir="./logs",                       # ✅ For TensorBoard etc.
    logging_steps=10,                           # ✅ Log every 10 steps
    save_strategy="epoch",                      # ✅ Save after each epoch
    save_total_limit=2,                         # ✅ Keep last 2 checkpoints
    #evaluation_strategy="no",                   # ✅ Skip eval for now
    save_safetensors=False,                     # ✅ Use `.bin` format
    report_to="none",                           # ✅ Disable wandb/hub
    seed=1991,                                    # ✅ For reproducibility
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # must return input_ids, labels, attention_mask, position_ids, team_ids, type_ids
    data_collator=None,
    tokenizer=None
)

trainer.train()