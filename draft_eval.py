import torch

def get_blocked_token_ids(used_token_ids, pad_token_id=252, device='cpu'):
    """
    Given a list of used token IDs (input_ids), returns a tensor of blocked token IDs
    mapped to the pick-space [0–125].

    Args:
        used_token_ids (List[int]): List of token ids from the sequence.
        pad_token_id (int): The pad token ID to ignore.
        device (str or torch.device): Where to place the returned tensor.

    Returns:
        torch.LongTensor: A tensor of shape [1, N] listing blocked pick-token ids.
    """
    blocked_ids = set()
    for tok in used_token_ids:
        if tok >= pad_token_id:
            continue
        blocked_ids.add(tok)
        if 0 <= tok <= 125:  # pick
            blocked_ids.add(tok)
            blocked_ids.add(tok+126)
        elif 126 <= tok <= 251:  # ban
            blocked_ids.add(tok - 126)  # map ban_X -> pick_X
            blocked_ids.add(tok)

    return torch.tensor([list(blocked_ids)], dtype=torch.long, device=device)



def generate_causal_eval_samples(tokenized_doc, max_len=32):
    input_ids = tokenized_doc['input_ids']
    attention_mask = tokenized_doc['attention_mask']
    position_ids = tokenized_doc['position_ids']
    team_ids = tokenized_doc['team_ids']
    type_ids = tokenized_doc['type_ids']

    seq_len = attention_mask.sum().item()  # Only count non-pad tokens

    samples = []
    for i in range(1, seq_len):  # start from 1 to leave room for next token
        sample = {
            'input_ids': input_ids[:i].clone(),
            'attention_mask': attention_mask[:i].clone(),
            'position_ids': position_ids[:i].clone(),
            'team_ids': team_ids[:i].clone(),
            'type_ids': type_ids[:i].clone(),
            'label': input_ids[i].item(),
            'predicting_position':i  # the "next" token
        }
        samples.append(sample)

    return samples


def generate_masked_eval_samples(tokenized_doc, tokenizer):
    samples = []

    input_ids = tokenized_doc['input_ids']
    attention_mask = tokenized_doc['attention_mask']
    valid_len = attention_mask.sum().item()

    for i in range(valid_len):
        if input_ids[i].item() in tokenizer.special_tokens.values():
            continue  # skip [CLS], [SEP], [PAD]

        masked_ids = input_ids.clone()
        true_token = masked_ids[i].item()
        masked_ids[i] = tokenizer.mask_token_id

        sample = {
            'input_ids': masked_ids,
            'attention_mask': tokenized_doc['attention_mask'],
            'position_ids': tokenized_doc['position_ids'],
            'team_ids': tokenized_doc['team_ids'],
            'type_ids': tokenized_doc['type_ids'],
            'label': true_token,
            'predicting_position': i
        }
        samples.append(sample)

    return samples




import torch

def topk_accuracy(logits, true_labels, k=1):
    """
    Computes top-k accuracy.

    Args:
        logits (torch.Tensor): Model output logits of shape (batch_size, vocab_size)
        true_labels (torch.Tensor): True token ids of shape (batch_size,)
        k (int): Top-k level (e.g., 1, 3, 5)

    Returns:
        torch.Tensor: Top-k accuracy (scalar)
    """
    topk_preds = torch.topk(logits, k=k, dim=-1).indices  # (batch_size, k)
    match = topk_preds == true_labels.unsqueeze(1)        # (batch_size, k)
    correct = match.any(dim=1).float()                    # (batch_size,)
    return correct.mean()                                 # Scalar tensor

import pandas as pd




def evaluate_model_detailed(
    model,
    encoded_games,
    topks=(1, 3, 5, 10),
    tokenizer=None,
    mode="causal"  # or "masked"
):
    model.eval()
    records = []

    for doc_id, doc in enumerate(encoded_games):
        if mode == "causal":
            samples = generate_causal_eval_samples(doc)
        elif mode == "masked":
            samples = generate_masked_eval_samples(doc, tokenizer)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        for sample in samples:
            input_ids = sample['input_ids'].unsqueeze(0).to(model.device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(model.device)
            position_ids = sample['position_ids'].unsqueeze(0).to(model.device)
            team_ids = sample['team_ids'].unsqueeze(0).to(model.device)
            type_ids = sample['type_ids'].unsqueeze(0).to(model.device)
            ###
            ### we figure out who's banned and picked
            ### blocked out all the hero (both ban/picks)
            ### change logit to -inf
            ### output will be clean
            ###
            used_token_ids = input_ids[0].tolist()  # get list of tokens
            blocked_token_ids = get_blocked_token_ids(used_token_ids)
    # Remove padding tokens
           
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    team_ids=team_ids,
                    type_ids=type_ids,
                    blocked_token_ids=blocked_token_ids
                )
                logits = outputs['logits']
            if mode == "masked":
                pred_pos = sample['predicting_position']
                logits_at_pos = logits[0, pred_pos]
            elif mode == "causal":
                logits_at_pos = logits[0, -1]
                pred_pos = sample['predicting_position']
            true_label = sample['label']

            result = {
                "doc_id": doc_id,
                "predicting_position": pred_pos,
                "true_label": true_label,
            }

            for k in topks:
                topk_preds = torch.topk(logits_at_pos, k).indices.tolist()
                result[f"top{k}_correct"] = int(true_label in topk_preds)
                result[f"top{k}_preds"] = topk_preds
                if tokenizer:
                    result[f"top{k}_tokens"] = [tokenizer.decode_token_id(tid) for tid in topk_preds]
            if tokenizer:
                result["true_token"] = tokenizer.decode_token_id(true_label)

            records.append(result)

    return pd.DataFrame(records)

