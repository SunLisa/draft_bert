import json
import os

class DraftTokenizer:
    def __init__(self, vocab, reverse_vocab, dense_map, hero_id_to_name):
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.dense_map = dense_map
        self.hero_id_to_name = hero_id_to_name
        self.hero_ids = sorted(dense_map.keys())
        self.num_heroes = len(self.hero_ids)

        self.special_tokens = {
            '[PAD]': self.vocab['[PAD]'],
            '[MASK]': self.vocab['[MASK]'],
            '[CLS]': self.vocab['[CLS]'],
            '[SEP]': self.vocab['[SEP]'],
        }

        self.pad_token_id = self.special_tokens['[PAD]']
        self.mask_token_id = self.special_tokens['[MASK]']
        self.cls_token_id = self.special_tokens['[CLS]']
        self.sep_token_id = self.special_tokens['[SEP]']

    @classmethod
    def load_from_files(cls, vocab_dir="vocab"):
        with open(f"{vocab_dir}/vocab.json") as f:
            vocab = json.load(f)
        with open(f"{vocab_dir}/reverse_vocab.json") as f:
            reverse_vocab = json.load(f)
        with open(f"{vocab_dir}/dense_map.json") as f:
            dense_map = json.load(f)
        dense_map = {int(k): v for k, v in dense_map.items()}

        with open(f"{vocab_dir}/hero_id_to_name.json") as f:
            hero_id_to_name = json.load(f)
        hero_id_to_name = {int(k): v for k, v in hero_id_to_name.items()}

        return cls(vocab, reverse_vocab, dense_map, hero_id_to_name)


    def encode_action(self, action):
        #action_type = 'pick' if action['is_pick'] else 'ban'
        is_pick = action['is_pick']
        hero_id = action['hero_id']
        hero_name = self.hero_id_to_name.get(hero_id, 'UNKNOWN')
        #get hero name here
        token_str = f"{'pick' if is_pick else 'ban'}_{hero_name}"
        #token_str = f"{action_type}_{hero_name}"

        return {
        'token_str': token_str,
        'token_id': self.vocab.get(token_str),
        'team': action['team'],  # adjust here if team needs to be 0/1 instead of 2/3
        'order': action['order']+1,
        'is_pick': is_pick
        }

    def encode_sequence(self, actions):
        sorted_actions = sorted(actions, key=lambda x: x['order'])
        return [self.encode_action(a) for a in sorted_actions if self.encode_action(a)['token_id'] is not None]

    def decode_token_id(self, token_id):
        return self.reverse_vocab.get(token_id, '[UNK]')

    def vocab_size(self):
        return len(self.vocab)

    def __call__(self, actions, max_len=32, add_special_tokens=True, return_tensors=None):
        encoded = self.encode_sequence(actions)

        input_ids = [x['token_id'] for x in encoded]
        team_ids = [x['team'] for x in encoded]
        position_ids = [x['order'] for x in encoded]
        type_ids = [int(x['is_pick']) for x in encoded]

        if add_special_tokens:
            input_ids = [self.vocab['[CLS]']] + input_ids + [self.vocab['[SEP]']]
            team_ids = [0] + team_ids + [0]
            position_ids = [0] + position_ids + [0]
            type_ids = [0] + type_ids + [0]

        # Pad up to max_len
        pad_id = self.vocab['[PAD]']
        pad_len = max_len - len(input_ids)

        input_ids += [pad_id] * pad_len
        team_ids += [0] * pad_len
        position_ids += [0] * pad_len
        type_ids += [0] * pad_len

        attention_mask = [1 if token != pad_id else 0 for token in input_ids]

        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'team_ids': team_ids,
            'type_ids': type_ids
        }

        if return_tensors == 'pt':
            import torch
            output = {k: torch.tensor(v) for k, v in output.items()}

        return output

