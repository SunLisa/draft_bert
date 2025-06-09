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

docs = [sequence for _, sequence in draft_data.items()]
tokenized_docs = [t(doc, return_tensors='pt') for doc in docs]

#def split_tokenized_docs(tokenized_docs, tokenizer=None, mlm=True, val_size=0.15, test_size=0.15, seed=123):
from draft_dataset import split_tokenized_docs
# === 🧪 Train/Val/Test Split ===

train_docs, val_docs, test_docs = split_tokenized_docs(
    tokenized_docs,
    tokenizer=t,
    mid=True
)

btrain_dataset, bval_dataset, btest_dataset = split_tokenized_docs(
    tokenized_docs,
    tokenizer=t,
    mlm=True
)

gtrain_dataset, gval_dataset, gtest_dataset = split_tokenized_docs(
    tokenized_docs,
    tokenizer=t,
    mlm=False
)

from transformers import BertConfig

bconfig = BertConfig(
    vocab_size=256,             # set to match your tokenizer.vocab_size()
    hidden_size=64,
    num_attention_heads=4,
    num_hidden_layers=4,
    intermediate_size=256,
    max_position_embeddings=32,
    pad_token_id=252,
    # Optional but good practice:
    cls_token_id=254,
    sep_token_id=255,
    mask_token_id=253,
)
bconfig.add_pooling_layer = True

encoded_games = [t(actions, return_tensors='pt') for actions in docs]


from transformers import GPT2Config

gconfig = GPT2Config(
    vocab_size=256,             # match tokenizer
    n_embd=64,                 # same as hidden_size
    n_layer=4,                  # same depth
    n_head=4,                   # same # of heads
    n_inner=256,                # same intermediate size
    n_positions=32,            # max sequence length
    n_ctx=32,                  # max context (same as n_positions)
    pad_token_id=252,          # you need to set this explicitly
    bos_token_id=254,          # you can map [CLS] to [BOS]
    eos_token_id=255           # optionally map [SEP] to [EOS]
)
gconfig.add_pooling_layer = True
# Optional: if using GPT2LMHeadModel directly, it needs to know vocab_size etc.


#print(t.special_tokens)  # e.g. { '[PAD]': 252, '[MASK]': 253, '[CLS]': 254, '[SEP]': 255 }

# Map for GPT2:
gconfig.pad_token_id = t.pad_token_id         # 252
gconfig.bos_token_id = t.cls_token_id         # 254
gconfig.eos_token_id = t.sep_token_id         # 255

from draft_dataset import DraftMLMDataset, DraftCLMDataset

dataset = DraftMLMDataset(encoded_games,t)

from draft_bert import DraftBertForMaskedLM,DraftGPT2ForCausalLM
bmodel = DraftBertForMaskedLM(bconfig)

gmodel = DraftGPT2ForCausalLM(gconfig)


from transformers import TrainingArguments

btraining_args = TrainingArguments(
    output_dir="./checkpoints",                 # ✅ Save location
    overwrite_output_dir=True,                  # ✅ Overwrite if exists
    num_train_epochs=500,                        # ✅ Your plan
    per_device_train_batch_size=8,              # ✅ Tweak based on memory
    logging_dir="./logs",                       # ✅ For TensorBoard etc.
    logging_steps=10,                           # ✅ Log every 10 steps
    save_strategy="epoch",                      # ✅ Save after each epoch
    save_total_limit=2,                         # ✅ Keep last 2 checkpoints
    eval_strategy="epoch", #evaluation_strategy="no",                   # ✅ Skip eval for now
    save_safetensors=False,                     # ✅ Use `.bin` format
    report_to="none",                           # ✅ Disable wandb/hub
    seed=1991,                                    # ✅ For reproducibility
    load_best_model_at_end=True, 
)
from transformers import Trainer

btrainer = Trainer(
    model=bmodel,
    args=btraining_args,
    train_dataset=btrain_dataset,  # must return input_ids, labels, attention_mask, position_ids, team_ids, type_ids
    eval_dataset=bval_dataset,
    data_collator=None,
    tokenizer=None
)

btrainer.train()
btrainer.save_model("./model/final_bert_model")



gtraining_args = TrainingArguments(
    output_dir="./checkpoints",                 # ✅ Save location
    overwrite_output_dir=True,                  # ✅ Overwrite if exists
    num_train_epochs=500,                        # ✅ Your plan
    per_device_train_batch_size=8,              # ✅ Tweak based on memory
    logging_dir="./logs",                       # ✅ For TensorBoard etc.
    logging_steps=10,                           # ✅ Log every 10 steps
    save_strategy="epoch",                      # ✅ Save after each epoch
    save_total_limit=2,                         # ✅ Keep last 2 checkpoints
    eval_strategy="epoch", #evaluation_strategy="no",                   # ✅ Skip eval for now
    save_safetensors=False,                     # ✅ Use `.bin` format
    report_to="none",                           # ✅ Disable wandb/hub
    seed=1991,                                    # ✅ For reproducibility
    load_best_model_at_end=True, 
)
from transformers import Trainer

gtrainer = Trainer(
    model=gmodel,
    args=gtraining_args,
    train_dataset=gtrain_dataset,  # must return input_ids, labels, attention_mask, position_ids, team_ids, type_ids
    eval_dataset=gval_dataset,
    data_collator=None,
    tokenizer=None
)

gtrainer.train()
gtrainer.save_model("./model/final_gpt2_model")



from draft_eval import evaluate_model_detailed

df = evaluate_model_detailed(gmodel,train_docs,mode='causal')

df.to_csv('eval/gmodel_train.csv')
df.groupby("predicting_position")[["top1_correct", "top3_correct", "top5_correct", "top10_correct"]].mean()


df = evaluate_model_detailed(gmodel,test_docs,mode='causal')

df.to_csv('eval/gmodel_test.csv')
df.groupby("predicting_position")[["top1_correct", "top3_correct", "top5_correct", "top10_correct"]].mean()

df = evaluate_model_detailed(bmodel,train_docs,mode='mask')

df.to_csv('eval/bmodel_train.csv')
df.groupby("predicting_position")[["top1_correct", "top3_correct", "top5_correct", "top10_correct"]].mean()

df = evaluate_model_detailed(bmodel,test_docs,mode='mask')

df.to_csv('eval/bmodel_test.csv')
df.groupby("predicting_position")[["top1_correct", "top3_correct", "top5_correct", "top10_correct"]].mean()