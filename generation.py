import torch
import random
from draft_gen import beam_generate_draft_from_cut
from draft_tokenizer import DraftTokenizer
#from draft_utils import get_token_name  # optional helper if you have a pretty formatter
# we have a vocab json which is gonnabe a dictionary#
#
""" def load_from_files(cls, vocab_dir="vocab"):
        with open(f"{vocab_dir}/vocab.json") as f:
            vocab = json.load(f)
        with open(f"{vocab_dir}/reverse_vocab.json") as f:
            reverse_vocab = json.load(f)
        reverse_vocab = {int(k): v for k, v in reverse_vocab.items()}
        with open(f"{vocab_dir}/dense_map.json") as f:
            dense_map = json.load(f)
        dense_map = {int(k): v for k, v in dense_map.items()}

        with open(f"{vocab_dir}/hero_id_to_name.json") as f:
            hero_id_to_name = json.load(f)
        hero_id_to_name = {int(k): v for k, v in hero_id_to_name.items()}

        return cls(vocab, reverse_vocab, dense_map, hero_id_to_name)
"""
from draft_dataset import split_tokenized_docs

# === ⚙️ Load Tokenizer ===
t = DraftTokenizer.load_from_files()
import json
# === 📜 Load Pre-tokenized Data (or re-tokenize) ===
with open('draft_results.json', 'r') as f:
    draft_data = json.load(f)

docs = [seq for _, seq in draft_data.items()]
tokenized_docs = [t(doc, return_tensors='pt') for doc in docs]

# === 🔍 Sample a Few Interesting Drafts ===
_, _, test_docs = split_tokenized_docs(tokenized_docs, tokenizer=t, mid=True)
test_cases = random.sample(test_docs, k=3)  # or manually choose slices

# === 📦 Load Model ===
from transformers import GPT2Config
from draft_bert import DraftGPT2ForCausalLM

gconfig = GPT2Config.from_pretrained("./model/final_gpt2_model")
gmodel = DraftGPT2ForCausalLM(gconfig)
gmodel.load_state_dict(torch.load("./model/final_gpt2_model/pytorch_model.bin"))
gmodel.eval().to('cuda' if torch.cuda.is_available() else 'cpu')

# === 🧪 Try Beam Generation ===
for i, sample in enumerate(test_cases):
    print(f"\n===== 🧠 Beam Search Generation: Case {i+1} =====")

    input_ids = sample['input_ids'][0]
    valid_len = (sample['attention_mask'][0] > 0).sum().item()

    print("👀 Original Sequence:")
    print(sample)
    print([t.decode_token_id(tok.item()) for tok in input_ids[:valid_len]])

    beams = beam_generate_draft_from_cut(
        model=gmodel,
        tokenizer=t,
        sequence=sample,
        start_pos=18,
        beam_width=3,
        depth=6,
        max_len=32,
        model_type='gpt'
    )

    top_beam = beams[0]
    final_tokens = top_beam["input_ids"][0]
    decoded = [t.id_to_token(tok.item()) for tok in final_tokens if tok.item() != t.pad_token_id]

    print("\n✅ Top Beam (Best Sequence):")
    print(" →", decoded)
    print(f"🧮 Score: {top_beam['score']:.2f}")

    print("\n📊 Generation Log:")
    for step in top_beam["log"]:
        tok = t.id_to_token(step["predicted_token"])
        print(f"  Pos {step['position']:2d}: {tok:25s} (p={step['prob']:.4f})")

    print("\n📈 Top-k Alternatives Per Step:")
    for step in top_beam["log"]:
        names = [t.id_to_token(tok_id) for tok_id in step.get("topk_ids", [])]
        probs = step.get("topk_probs", [])
        print(f"  Step {step['position']}:")
        for name, p in zip(names, probs):
            print(f"    - {name:25s}: {p:.4f}")

    print("="*60)



# === 📦 Load BERT Model ===
from transformers import BertConfig
from draft_bert import DraftBertForMaskedLM

bconfig = BertConfig.from_pretrained("./model/final_bert_model")
bmodel = DraftBertForMaskedLM(bconfig)
bmodel.load_state_dict(torch.load("./model/final_bert_model/pytorch_model.bin"))
bmodel.eval().to('cuda' if torch.cuda.is_available() else 'cpu')

# === 🧪 Beam Search for BERT ===
for i, sample in enumerate(test_cases):
    print(f"\n===== 🧠 Beam Search Generation (BERT): Case {i+1} =====")

    input_ids = sample['input_ids'][0]
    valid_len = (sample['attention_mask'][0] > 0).sum().item()

    print("👀 Original Sequence:")
    print([t.id_to_token(tok.item()) for tok in input_ids[:valid_len]])

    beams = beam_generate_draft_from_cut(
        model=bmodel,
        tokenizer=t,
        sequence=sample,
        start_pos=18,
        beam_width=3,
        depth=6,
        max_len=32,
        model_type='bert'
    )

    top_beam = beams[0]
    final_tokens = top_beam["input_ids"][0]
    decoded = [t.id_to_token(tok.item()) for tok in final_tokens if tok.item() != t.pad_token_id]

    print("\n✅ Top Beam (Best Sequence):")
    print(" →", decoded)
    print(f"🧮 Score: {top_beam['score']:.2f}")

    print("\n📊 Generation Log:")
    for step in top_beam["log"]:
        tok = t.id_to_token(step["predicted_token"])
        print(f"  Pos {step['position']:2d}: {tok:25s} (p={step['prob']:.4f})")

    print("\n📈 Top-k Alternatives Per Step:")
    for step in top_beam["log"]:
        names = [t.id_to_token(tok_id) for tok_id in step.get("topk_ids", [])]
        probs = step.get("topk_probs", [])
        print(f"  Step {step['position']}:")
        for name, p in zip(names, probs):
            print(f"    - {name:25s}: {p:.4f}")

    print("="*60)
