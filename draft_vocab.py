import requests
import json
import os

# Step 1: Fetch hero data
url = "https://api.opendota.com/api/heroes"
response = requests.get(url)

if response.status_code != 200:
    raise RuntimeError(f"Failed to get heroes: {response.status_code}")

heroes = response.json()
hero_id_name_map = {hero["id"]: hero["localized_name"] for hero in heroes}
hero_ids = sorted(hero_id_name_map.keys())

# Step 2: Build vocab, reverse_vocab, dense_map
dense_map = {hid: idx for idx, hid in enumerate(hero_ids)}
num_heroes = len(hero_ids)

vocab = {}
reverse_vocab = {}

for hero_id, dense_id in dense_map.items():
    hero_name = hero_id_name_map[hero_id]
    pick_token = f"pick_{hero_name}"
    ban_token = f"ban_{hero_name}"

    vocab[pick_token] = dense_id
    vocab[ban_token] = dense_id + num_heroes

    reverse_vocab[dense_id] = pick_token
    reverse_vocab[dense_id + num_heroes] = ban_token

# Special tokens
special_tokens = {
    '[PAD]': num_heroes * 2,
    '[MASK]': num_heroes * 2 + 1,
    '[CLS]': num_heroes * 2 + 2,
    '[SEP]': num_heroes * 2 + 3
}

vocab.update(special_tokens)
reverse_vocab.update({v: k for k, v in special_tokens.items()})

# Save to disk
os.makedirs("vocab", exist_ok=True)

with open("vocab/vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

with open("vocab/reverse_vocab.json", "w") as f:
    json.dump(reverse_vocab, f, indent=2)

with open("vocab/dense_map.json", "w") as f:
    json.dump({hid: dense_map[hid] for hid in dense_map}, f, indent=2)

with open("vocab/hero_id_to_name.json", "w") as f:
    json.dump(hero_id_name_map, f, indent=2)

"✅ Vocab files generated and saved."
