from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split


def split_tokenized_docs(tokenized_docs, tokenizer=None, mlm=True, mid=False, val_size=0.15, test_size=0.15, seed=123):
    """
    Splits tokenized docs into train/val/test and wraps them in appropriate dataset class.

    Args:
        tokenized_docs: List of tokenized docs (dict of tensors).
        tokenizer: Required only for MLM dataset.
        mlm: True for MLM, False for CLM.
        val_size: Fraction of data for validation.
        test_size: Fraction of data for test.
        seed: Random seed for reproducibility.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from draft_dataset import DraftMLMDataset, DraftCLMDataset
    from sklearn.model_selection import train_test_split


    train_docs, test_docs = train_test_split(tokenized_docs, test_size=test_size, random_state=seed)
    train_docs, val_docs = train_test_split(train_docs, test_size=val_size, random_state=seed)

    if mid:
        return train_docs, val_docs, test_docs

    if mlm:
        train_dataset = DraftMLMDataset(train_docs, tokenizer)
        val_dataset = DraftMLMDataset(val_docs, tokenizer)
        test_dataset = DraftMLMDataset(test_docs, tokenizer)
    else:
        train_dataset = DraftCLMDataset(train_docs)
        val_dataset = DraftCLMDataset(val_docs)
        test_dataset = DraftCLMDataset(test_docs)

    return train_dataset, val_dataset, test_dataset



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
            seq_len = doc["attention_mask"].sum().item()  # Number of real tokens (before padding)
            for i in range(1, min(seq_len, max_len)):
                sample = {
                    "input_ids": doc["input_ids"][:i+1],
                    "attention_mask": doc["attention_mask"][:i+1],
                    "position_ids": doc["position_ids"][:i+1],
                    "team_ids": doc["team_ids"][:i+1],
                    "type_ids": doc["type_ids"][:i+1],}
                self.samples.append(self._process(sample))

    def _process(self, tokens):
        input_ids = tokens["input_ids"]
        position_ids = tokens["position_ids"]
        team_ids = tokens["team_ids"]
        type_ids = tokens["type_ids"]
        attention_mask = tokens["attention_mask"]

        # Shift input to create labels
        labels = input_ids[1:].tolist() + [self.pad_token_id]

        input_ids = input_ids.tolist()
        position_ids = position_ids.tolist()
        team_ids = team_ids.tolist()
        type_ids = type_ids.tolist()
        attention_mask = attention_mask.tolist()

        pad_len = max(0, 32 - len(input_ids))
        input_ids += [self.pad_token_id] * pad_len
        labels += [self.pad_token_id] * pad_len
        position_ids += [0] * pad_len
        team_ids += [0] * pad_len
        type_ids += [0] * pad_len
        attention_mask += [0] * pad_len

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
        #def __getitem__(self, idx):
        item = self.samples[idx]
        #print({k: len(v) for k, v in item.items()})  # quick length sanity check
        return {k: torch.tensor(v) for k, v in item.items()}

        #return {k: torch.tensor(v) for k, v in self.samples[idx].items()}
