def prepare_draft_for_generation(sequence, tokenizer, start_generate_from=None):
    """
    Prepares a tokenized sequence for auto-regressive generation by zeroing out input_ids
    after a specified position. If no position is given, it finds the [SEP] token or falls
    back to attention_mask length.
    
    Returns a modified copy of the sequence dict.
    """
    seq = {k: v.clone() for k, v in sequence.items()}  # avoid modifying in-place
    input_ids = seq['input_ids'][0]  # shape: [seq_len]
    
    # Find SEP token
    sep_token_id = tokenizer.sep_token_id
    sep_idx = (input_ids == sep_token_id).nonzero(as_tuple=True)
    sep_pos = sep_idx[0].item() if len(sep_idx[0]) > 0 else input_ids.size(0)

    # Default to SEP token or attention_mask length
    max_valid_pos = start_generate_from if start_generate_from is not None else sep_pos

    # Zero out input_ids after max_valid_pos
    seq['input_ids'][0, max_valid_pos:] = 0

    return seq, max_valid_pos


def generate_draft_from_cut(model, tokenizer, starting_sequence, max_len=32, top_k=10):
    """
    Generate a draft from a cut sequence, preserving embeddings and collecting top-k logits at each step.
    
    Returns:
        - final generated input_ids
        - a list of dicts: each containing position, predicted token, top_k ids and their probabilities
    """
    model.eval()
    sequence = tokenizer(starting_sequence, return_tensors='pt')
    sequence, start_pos = prepare_draft_for_generation(sequence, tokenizer)
    sequence = {k: v.to(model.device) for k, v in sequence.items()}
    
    generation_log = []

    for i in range(start_pos, max_len):
        with torch.no_grad():
            blocked_ids = get_blocked_token_ids(sequence['input_ids'][0])
            blocked_token_ids = torch.tensor([list(blocked_ids)], dtype=torch.long).to(model.device)

            outputs = model(**sequence, blocked_token_ids=blocked_token_ids)
            logits = outputs["logits"][0, i]

            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, top_k)

            predicted_token = topk_ids[0].item()
            sequence['input_ids'][0, i] = predicted_token  # Greedy fill

            generation_log.append({
                'position': i,
                'predicted_token': predicted_token,
                'topk_ids': topk_ids.tolist(),
                'topk_probs': topk_probs.tolist()
            })

            if predicted_token == tokenizer.sep_token_id:
                break

    return sequence['input_ids'][0].tolist(), generation_log
