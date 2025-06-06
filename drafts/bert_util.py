class DraftTokenizer:
    def __init__(self, hero_ids):
        # Build dense map from hero_id to 0...N-1
        self.hero_ids = sorted(hero_ids)
        self.hero_id_to_dense = {hid: idx for idx, hid in enumerate(self.hero_ids)}
        self.num_heroes = len(self.hero_ids)

        # Build vocab and reverse vocab
        self.vocab = {}
        self.reverse_vocab = {}

        for hero_id, dense_id in self.hero_id_to_dense.items():
            pick_token = f"pick_{hero_id}"
            ban_token = f"ban_{hero_id}"

            self.vocab[pick_token] = dense_id
            self.vocab[ban_token] = dense_id + self.num_heroes

            self.reverse_vocab[dense_id] = pick_token
            self.reverse_vocab[dense_id + self.num_heroes] = ban_token

        # Special tokens
        self.special_tokens = {
            '[PAD]': self.num_heroes * 2,
            '[MASK]': self.num_heroes * 2 + 1,
            '[CLS]': self.num_heroes * 2 + 2,
            '[SEP]': self.num_heroes * 2 + 3
        }

        for token, token_id in self.special_tokens.items():
            self.vocab[token] = token_id
            self.reverse_vocab[token_id] = token

    def encode_action(self, action):
        """Convert a single action dict to its token ID."""
        action_type = 'pick' if action['is_pick'] else 'ban'
        hero_id = action['hero_id']
        token_str = f"{action_type}_{hero_id}"

        return {
            'token_str': token_str,
            'token_id': self.vocab.get(token_str),
            'team': action['team'],
            'order': action['order'],
            'is_pick': action['is_pick']
        }

    def encode_sequence(self, actions):
        """Convert a list of actions to a tokenized sequence."""
        sorted_actions = sorted(actions, key=lambda x: x['order'])
        return [self.encode_action(a) for a in sorted_actions if self.encode_action(a)['token_id'] is not None]

    def decode_token_id(self, token_id):
        """Convert a token ID back to its string form."""
        return self.reverse_vocab.get(token_id, '[UNK]')

    def vocab_size(self):
        return len(self.vocab)



tokenizer = DraftTokenizer(hero_ids)


tokens = tokenizer.encode_sequence(match_data['picks_bans'])



def make_training_example(token_ids, max_len=32, pad_token_id=252):
    # Pad or truncate to max_len
    input_ids = token_ids[:max_len]
    input_ids += [pad_token_id] * (max_len - len(input_ids))

    # Shifted labels (next token prediction)
    labels = input_ids[1:] + [pad_token_id]

    attention_mask = [1 if token != pad_token_id else 0 for token in input_ids]

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


from torch.utils.data import Dataset

class DraftSequenceDataset(Dataset):
    def __init__(self, list_of_token_ids, max_len=32, pad_token_id=252):
        self.examples = [make_training_example(seq, max_len, pad_token_id)
                         for seq in list_of_token_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}

from transformers import BertConfig

config = BertConfig(
    vocab_size=256,             # set to match your tokenizer.vocab_size()
    hidden_size=128,
    num_attention_heads=4,
    num_hidden_layers=2,
    intermediate_size=512,
    max_position_embeddings=32,
    pad_token_id=252
)

from transformers import BertForMaskedLM

model = BertForMaskedLM(config)


from transformers import Trainer, TrainingArguments
import torch
training_args = TrainingArguments(
    output_dir='./draftbert',
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()




from transformers import BertConfig

config = BertConfig(
    vocab_size=256,               # from your tokenizer
    hidden_size=128,
    num_attention_heads=4,
    num_hidden_layers=4,
    intermediate_size=512,
    max_position_embeddings=32,   # max length of draft sequence
    pad_token_id=252,               # match your tokenizer's [PAD]
)




class DraftDataset(Dataset):
    def __init__(self, sequences, max_len=32, pad_token_id=252):
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.examples = [self.process(seq) for seq in sequences]

    def process(self, token_objs):
        input_ids = [x['token_id'] for x in token_objs]
        position_ids = [x['order'] for x in token_objs]
        team_ids = [x['team'] for x in token_objs]
        type_ids = [int(x['is_pick']) for x in token_objs]

        input_ids += [self.pad_token_id] * (self.max_len - len(input_ids))
        position_ids += [0] * (self.max_len - len(position_ids))
        team_ids += [0] * (self.max_len - len(team_ids))
        type_ids += [0] * (self.max_len - len(type_ids))

        labels = input_ids[1:] + [self.pad_token_id]
        attention_mask = [1 if i != self.pad_token_id else 0 for i in input_ids]
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'team_ids': team_ids,
            'type_ids': type_ids,
            'attention_mask': attention_mask
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}



dataset = DraftDataset([tokens])



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

    def forward(self, input_ids, attention_mask=None, position_ids=None, team_ids=None, type_ids=None):
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

    def forward(self, input_ids, attention_mask=None, position_ids=None, team_ids=None, type_ids=None, labels=None):
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
