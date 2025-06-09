from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
import torch


class DraftMLMDataset(Dataset):
    def __init__(self, tokenized_docs, tokenizer, max_len=32, pad_token_id=252):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.docs = tokenized_docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        # This is how you generate input_ids
        #def __getitem__(self, idx):
        inputs = self.docs[idx]  # Already a dict of tensors (e.g., from tokenizer(actions, return_tensors='pt'))

        input_ids = inputs['input_ids'].clone()
        labels = input_ids.clone()

        # Apply 15% masking
        prob_matrix = torch.full(labels.shape, 0.15)
        mask = torch.bernoulli(prob_matrix).bool()

        # Replace with [MASK] token
        input_ids[mask] = self.tokenizer.mask_token_id

        # Only compute loss on masked positions
        labels = labels.masked_fill(~mask, -100)

        # Return the batch
        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'],
            'position_ids': inputs['position_ids'],
            'team_ids': inputs['team_ids'],
            'type_ids': inputs['type_ids'],
            'labels': labels
        }



class DraftCLMDataset(Dataset):
    def __init__(self, tokenized_docs, max_len=32, pad_token_id=252):
        self.samples = []
        self.pad_token_id = pad_token_id
        for doc in tokenized_docs:
            for i in range(1, min(len(doc), max_len)):
                sample = doc[:i+1]
                self.samples.append(self._process(sample))

    def _process(self, tokens):
        input_ids = [x['token_id'] for x in tokens]
        position_ids = [x['order'] for x in tokens]
        team_ids = [x['team'] for x in tokens]
        type_ids = [int(x['is_pick']) for x in tokens]

        # Labels is next token: shift
        labels = input_ids[1:] + [self.pad_token_id]

        pad_len = max(0, 32 - len(input_ids))
        input_ids += [self.pad_token_id] * pad_len
        labels += [self.pad_token_id] * pad_len
        position_ids += [0] * pad_len
        team_ids += [0] * pad_len
        type_ids += [0] * pad_len
        attention_mask = [1 if i != self.pad_token_id else 0 for i in input_ids]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'team_ids': team_ids,
            'type_ids': type_ids,
            'attention_mask': attention_mask
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.samples[idx].items()}
