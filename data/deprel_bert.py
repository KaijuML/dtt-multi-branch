import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertForTokenClassification

from biaffine import DeepBiaffineScorer


class LogitsSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.to_single_head = nn.Linear(self.num_attention_heads, 1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if attention_mask is not None:
            dep_mask = -10000 * (1 - attention_mask).unsqueeze(1).unsqueeze(1)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + dep_mask

        attention_scores = attention_scores.permute(0, 2, 3, 1)
        attention_scores = self.to_single_head(attention_scores).squeeze()

        return attention_scores


class BertForDependencyRelations(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        # For labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # For relations
        self.attention = DeepBiaffineScorer(config.hidden_size, config.hidden_size, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # Dependency relations
        dependencies = self.attention(sequence_output)

        # Relations' tags
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = ([logits, dependencies],) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels_tags, labels_dep = labels.chunk(2, dim=-1)
            seq_len = input_ids.size(1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_dependencies = dependencies.view(-1, seq_len)[active_loss]
                active_labels_tags = labels_tags.view(-1)[active_loss]
                active_labels_dep = labels_dep.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels_tags) + loss_fct(active_dependencies, active_labels_dep)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels_tags.view(-1)) + \
                       loss_fct(dependencies.view(-1, seq_len), labels_dep.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
