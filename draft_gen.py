import torch
from draft_eval import get_blocked_token_ids

def greedy_generate_draft_from_cut(model, tokenizer, sequence, start_pos, max_len=32, top_k=5, model_type='bert'):
    """
    sequence: a tokenized input dict with fields like input_ids, attention_mask, etc.
    model_type: 'bert' or 'gpt'
    """
    model.eval()
    sequence = {k: v.clone().to(model.device) for k, v in sequence.items()}
    input_ids = sequence["input_ids"]

    generation_log = []

    for i in range(start_pos, max_len):
        # Prepare model inputs
        if model_type == 'gpt':
    # Slice everything to only past + current token
            inputs = {
                "input_ids": input_ids[:, :i+1],
                "attention_mask": sequence["attention_mask"][:, :i+1],
                "position_ids": sequence["position_ids"][:, :i+1],
                "team_ids": sequence["team_ids"][:, :i+1],
                "type_ids": sequence["type_ids"][:, :i+1],
            }

        elif model_type == 'bert':
            # Keep full-length inputs, mask future tokens
            masked_input_ids = input_ids.clone()
            masked_input_ids[0, i:] = tokenizer.mask_token_id

            inputs = {
                "input_ids": masked_input_ids,
                "attention_mask": sequence["attention_mask"],  # full
                "position_ids": sequence["position_ids"],
                "team_ids": sequence["team_ids"],
                "type_ids": sequence["type_ids"],
            }
        
        else:
            raise ValueError("Unsupported model type")

        with torch.no_grad():
            blocked_ids = get_blocked_token_ids(input_ids[0, :i])
            blocked_token_ids = torch.tensor([list(blocked_ids)], dtype=torch.long).to(model.device)
            outputs = model(**inputs, blocked_token_ids=blocked_token_ids)

        logits = outputs["logits"][0, -1 if model_type == 'gpt' else i]
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, top_k)
        predicted_token = topk_ids[0].item()

        # Update sequence
        input_ids[0, i] = predicted_token
        generation_log.append({
            "position": i,
            "predicted_token": predicted_token,
            "topk_ids": topk_ids.tolist(),
            "topk_probs": topk_probs.tolist()
        })

        if predicted_token == tokenizer.sep_token_id:
            break

    return input_ids[0].tolist(), generation_log


def beam_generate_draft_from_cut(model, tokenizer, sequence, start_pos, beam_width=3, depth = 3, max_len=32, model_type='bert'):
    """
    Beam search generation for Dota draft modeling.
    Returns top sequences and their logs.
    """
    model.eval()
    device = model.device
    sequence = {k: v.clone().to(device) for k, v in sequence.items()}
    input_ids = sequence["input_ids"]

    # Initialize beams
    beams = [{
        "input_ids": input_ids.clone(),
        "score": 0.0,
        "log": [],
    }]

    for i in range(start_pos, min(start_pos + depth,max_len)):
        new_beams = []

        for beam in beams:
            beam_input_ids = beam["input_ids"]

            # Build model inputs
            if model_type == 'gpt':
                inputs = {
                    "input_ids": beam_input_ids[:, :i+1],
                    "attention_mask": sequence["attention_mask"][:, :i+1],
                    "position_ids": sequence["position_ids"][:, :i+1],
                    "team_ids": sequence["team_ids"][:, :i+1],
                    "type_ids": sequence["type_ids"][:, :i+1],
                }
            elif model_type == 'bert':
                masked_input_ids = beam_input_ids.clone()
                masked_input_ids[0, i:] = tokenizer.mask_token_id
                inputs = {
                    "input_ids": masked_input_ids,
                    "attention_mask": sequence["attention_mask"],
                    "position_ids": sequence["position_ids"],
                    "team_ids": sequence["team_ids"],
                    "type_ids": sequence["type_ids"],
                }
            else:
                raise ValueError("Unsupported model type")

            # Block used tokens
            with torch.no_grad():
                blocked_ids = get_blocked_token_ids(beam_input_ids[0, :i])
                blocked_token_ids = torch.tensor([list(blocked_ids)], dtype=torch.long).to(device)
                outputs = model(**inputs, blocked_token_ids=blocked_token_ids)

            logits = outputs["logits"][0, -1 if model_type == 'gpt' else i]
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, beam_width)

            # Expand beams
            for j in range(beam_width):
                new_input_ids = beam_input_ids.clone()
                new_input_ids[0, i] = topk_ids[j]

                new_beams.append({
                    "input_ids": new_input_ids,
                    "score": beam["score"] + topk_probs[j].log().item(),
                    "log": beam["log"] + [{
                        "position": i,
                        "predicted_token": topk_ids[j].item(),
                        "prob": topk_probs[j].item(),
                        "topk_ids": topk_ids.tolist(),
                        "topk_probs": topk_probs.tolist()
                        }],
                        
                })

        # Prune to top-k
        beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_width]

        # Early exit if all beams finished with SEP
        if all(beam["input_ids"][0, i].item() == tokenizer.sep_token_id for beam in beams):
            break

    return beams



from collections import defaultdict
import math

def analyze_beams_by_intervention(beams, round_index, target_token_id):
    """
    beams: List of beam dicts (from beam_generate_draft_from_cut)
    round_index: Which step in the generation you consider the intervention point
    target_token_id: ID of the hero you want to track in future rounds

    Returns:
        A dict mapping from picked hero at round_index -> prob of seeing target_token_id later
    """
    influence_map = defaultdict(lambda: {"total_prob": 0.0, "target_prob": 0.0})

    for beam in beams:
        log = beam["log"]
        score = beam["score"]
        prob = math.exp(score)  # convert log-prob back to probability

        if len(log) <= round_index:
            continue  # skip incomplete beams

        picked_token = log[round_index]["predicted_token"]

        future_tokens = [step["predicted_token"] for step in log[round_index+1:]]
        saw_target = target_token_id in future_tokens

        influence_map[picked_token]["total_prob"] += prob
        if saw_target:
            influence_map[picked_token]["target_prob"] += prob

    # Normalize: what % of beams with pick X led to target hero
    normalized = {
        t: round(data["target_prob"] / data["total_prob"], 4) if data["total_prob"] > 0 else 0.0
        for t, data in influence_map.items()
    }

    return normalized
