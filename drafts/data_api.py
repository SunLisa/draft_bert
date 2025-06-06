import requests

# Define the API endpoint for heroes
url = "https://api.opendota.com/api/heroes"

# Make the GET request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    heroes_data = response.json()

    # Optionally, print just the first few to get a feel
    print("Sample Heroes:")
    for hero in heroes_data[:5]:  # adjust the slice as needed
        print(f"Hero ID: {hero['id']}, Name: {hero['localized_name']}")
else:
    print(f"Failed to retrieve heroes. Status code: {response.status_code}")


hero_ids = set()
for hero in heroes_data:
    hero_ids.add(hero['id'])
len(hero_ids)


def build_dense_vocab(hero_ids):
    hero_ids = sorted(hero_ids)
    dense_map = {hid: idx for idx, hid in enumerate(hero_ids)}
    num_heroes = len(hero_ids)

    vocab = {}
    reverse_vocab = {}

    for hero_id, dense_id in dense_map.items():
        pick_token = f"pick_{hero_id}"
        ban_token = f"ban_{hero_id}"

        vocab[pick_token] = dense_id
        vocab[ban_token] = dense_id + num_heroes

        reverse_vocab[dense_id] = pick_token
        reverse_vocab[dense_id + num_heroes] = ban_token

    # Add special tokens
    special_tokens = {
        '[PAD]': num_heroes * 2,
        '[MASK]': num_heroes * 2 + 1,
        '[CLS]': num_heroes * 2 + 2,
        '[SEP]': num_heroes * 2 + 3
    }

    for token, token_id in special_tokens.items():
        vocab[token] = token_id
        reverse_vocab[token_id] = token

    return vocab, reverse_vocab, dense_map


vocab, reverse_vocab, dense_map = build_dense_vocab(hero_ids)


import requests

# Define the API endpoint for leagues
url = "https://api.opendota.com/api/leagues"

# Make the GET request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    leagues_data = response.json()

    # Optionally, print just the first few to get a feel
    print("Sample Leagues:")
    for league in leagues_data[:5]:  # adjust the slice as needed
        print(f"League ID: {league['leagueid']}, Name: {league['name']}, Tier: {league.get('tier', 'N/A')}")
else:
    print(f"Failed to retrieve leagues. Status code: {response.status_code}")





#8303157854

import requests

# Define the match ID and API endpoint
match_id = 8303157854
url = f"https://api.opendota.com/api/matches/{match_id}"

# Make the GET request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    match_data = response.json()
    print("Match Data:")
    print(match_data)
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")




import os
import time
import json
import requests

def fetch_data_from_league(league_id, save_dir="dota_matches"):
    os.makedirs(save_dir, exist_ok=True)

    match_list_url = f"https://api.opendota.com/api/leagues/{league_id}/matches"
    match_list_response = requests.get(match_list_url)

    if match_list_response.status_code != 200:
        print(f"Failed to retrieve matches for league {league_id}. Status code: {match_list_response.status_code}")
        return []

    matches = match_list_response.json()
    match_data_list = []

    for match in matches:
        match_id = match['match_id']
        match_url = f"https://api.opendota.com/api/matches/{match_id}"
        match_response = requests.get(match_url)

        if match_response.status_code == 200:
            match_data = match_response.json()
            match_data_list.append(match_data)

            with open(f"{save_dir}/match_{match_id}.json", "w") as f:
                json.dump(match_data, f, indent=2)
            print(f"Saved match {match_id}")
        else:
            print(f"Failed to retrieve match {match_id}. Status code: {match_response.status_code}")

        time.sleep(1.5)  # Be kind to OpenDota API limits

    return match_data_list
