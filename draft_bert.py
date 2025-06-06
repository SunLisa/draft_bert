import torch.nn as nn
class DraftEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.team_embeddings = nn.Embedding(2, config.hidden_size)
        self.type_embeddings = nn.Embedding(2, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, team_ids, type_ids):
        x = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.team_embeddings(team_ids)
            + self.type_embeddings(type_ids)
        )
        return self.dropout(self.LayerNorm(x))





from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPooling
class DraftBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = DraftEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if getattr(config, "add_pooling_layer", False) else None
        self.is_causal = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None, team_ids=None, type_ids=None):
        if self.is_causal:  # hypothetical flag
            seq_len = input_ids.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
            attention_mask = causal_mask.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # broadcasted mask
        
        # 👇 Inject custom embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            team_ids=team_ids,
            type_ids=type_ids
        )

        # Standard BERT encoder behavior
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None



        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output
        )



from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class DraftBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = DraftBertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, position_ids=None, team_ids=None, type_ids=None, labels=None, blocked_token_ids=None):
        # Pass everything through the model
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            team_ids=team_ids,
            type_ids=type_ids
        )

        sequence_output = outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)

        if blocked_token_ids is not None:
            for i in range(prediction_scores.size(0)):  # batch size
                for blocked_id in blocked_token_ids[i]:
                    prediction_scores[i, :, blocked_id] = float('-inf')

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return {
            "loss": loss,
            "logits": prediction_scores
        }
config.add_pooling_layer = True


model = DraftBertForMaskedLM(config)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./checkpoints",
    save_strategy="no",  # or "epoch", "steps"
    num_train_epochs=50,
    #per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    #evaluation_strategy="no",
    save_safetensors=False,  # ✅ disables "safe serialization"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # must return input_ids, labels, attention_mask, position_ids, team_ids, type_ids
)

trainer.train()



def get_seen_token_list(input_ids, pad_token_id=252):
    """
    For each sequence in the batch, return a list of token IDs that were already used
    (excluding [PAD] tokens).
    """
    blocked = []
    for seq in input_ids:
        seen = set()
        for token_id in seq:
            if token_id != pad_token_id:
                seen.add(token_id)
        blocked.append(list(seen))
    return blocked
