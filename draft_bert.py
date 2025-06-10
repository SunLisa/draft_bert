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




import torch
from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPooling
class DraftBertModel(BertModel):
    def __init__(self, config, is_causal):
        super().__init__(config)
        self.embeddings = DraftEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if getattr(config, "add_pooling_layer", False) else None
        self.is_causal = is_causal

    def forward(self, input_ids, attention_mask=None, position_ids=None, team_ids=None, type_ids=None):
        if self.is_causal:  # hypothetical flag
            seq_len = input_ids.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
            attention_mask = causal_mask.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # broadcasted mask
        """
        print("input_ids:", input_ids.shape)
        print("attention_mask:", attention_mask.shape)
        print("position_ids:", position_ids.shape)
        print("team_ids:", team_ids.shape)
        print("type_ids:", type_ids.shape)
        """

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
    def __init__(self, config, is_causal=0):
        super().__init__(config)
        self.bert = DraftBertModel(config, is_causal)
        #self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
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




from transformers import GPT2PreTrainedModel, GPT2Model
import torch.nn as nn

class DraftGPT2ForCausalLM(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)

        self.team_embed = nn.Embedding(2, config.hidden_size)
        self.type_embed = nn.Embedding(2, config.hidden_size)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pad_token_id = config.pad_token_id

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, position_ids=None, team_ids=None, type_ids=None, labels=None,
                 blocked_token_ids=None):
        team_embeddings = self.team_embed(team_ids)
        type_embeddings = self.type_embed(type_ids)

        inputs_embeds = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        inputs_embeds = inputs_embeds + team_embeddings + type_embeddings

        outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.lm_head(outputs.last_hidden_state)

        if blocked_token_ids is not None:
            for i in range(logits.size(0)):  # batch size
                for blocked_id in blocked_token_ids[i]:
                    logits[i, -1, blocked_id] = float('-inf')

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {'loss': loss, 'logits': logits}
