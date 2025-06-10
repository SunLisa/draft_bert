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
        self.mask_token = '[MASK]'
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'

    @classmethod
    def load_from_files(cls, vocab_dir="vocab"):
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

    def get_vocab(self):
        return self.vocab

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
            output = {
                'input_ids': torch.tensor(output['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(output['attention_mask'], dtype=torch.bool),
                'position_ids': torch.tensor(output['position_ids'], dtype=torch.long),
                'team_ids': torch.tensor(output['team_ids'], dtype=torch.long),
                'type_ids': torch.tensor(output['type_ids'], dtype=torch.long),
            }

        return output

    def pad(self, encoded_inputs, padding=True, max_length=32, return_tensors="pt", **kwargs):
        import torch

        batch = {k: [example[k] for example in encoded_inputs] for k in encoded_inputs[0]}
        
        padded_batch = {}
        for k, sequences in batch.items():
            max_len = max(len(seq) for seq in sequences) if padding is True else max_length
            pad_id = self.pad_token_id if hasattr(self, "pad_token_id") else 0

            padded = [
                list(seq) + [pad_id] * (max_len - len(seq))
                for seq in sequences
            ]
            padded_batch[k] = torch.tensor(padded)

        return padded_batch


    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        """
        Returns a mask with 1 for special tokens, 0 otherwise.
        This is used by Hugging Face to avoid masking special tokens during MLM.
        """
        special_ids = set(self.special_tokens.values())
        return [1 if token_id in special_ids else 0 for token_id in token_ids]



