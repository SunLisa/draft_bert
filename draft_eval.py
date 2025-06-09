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
            'label': input_ids[i].item()  # the "next" token
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

def evaluate_model_detailed(model, encoded_games, topks=(1, 3, 5, 10), tokenizer=None):
    model.eval()
    records = []

    for doc_id, doc in enumerate(encoded_games):
        samples = generate_causal_eval_samples(doc)

        for sample in samples:
            input_ids = sample['input_ids'].unsqueeze(0).to(model.device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(model.device)
            position_ids = sample['position_ids'].unsqueeze(0).to(model.device)
            team_ids = sample['team_ids'].unsqueeze(0).to(model.device)
            type_ids = sample['type_ids'].unsqueeze(0).to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    team_ids=team_ids,
                    type_ids=type_ids
                )
                logits = outputs['logits']

            next_token_logits = logits[0, -1]
            true_label = sample['label']
            
            result = {
                "doc_id": doc_id,
                "predicting_position": len(input_ids[0]),
                "true_label": true_label,
            }

            for k in topks:
                topk_preds = torch.topk(next_token_logits, k).indices.tolist()
                result[f"top{k}_correct"] = int(true_label in topk_preds)
                result[f"top{k}_preds"] = topk_preds
                if tokenizer:
                    result[f"top{k}_tokens"] = [tokenizer.decode_token_id(tid) for tid in topk_preds]
            
            if tokenizer:
                result["true_token"] = tokenizer.decode_token_id(true_label)

            records.append(result)

    df = pd.DataFrame(records)
    return df
