import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from nltk.util import pr

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import LayerNorm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.models.mobilebert import MobileBertForMaskedLM
from transformers.utils import logging

from .configuration_elasticbert import ElasticBertConfig

logger = logging.get_logger(__name__)


# _CHECKPOINT_FOR_DOC = "bert-base-uncased"
# _CONFIG_FOR_DOC = "BertConfig"
# _TOKENIZER_FOR_DOC = "BertTokenizer"


def shift_nonzero(remain_layers):
    bsz, seq_len = remain_layers.shape
    shifted_remain_layers = []
    for r in remain_layers:
        tmp = []
        for v in r:
            if v > 0:
                tmp.append(v)
        for _ in range(seq_len - len(tmp)):
            tmp.append(0)
        shifted_remain_layers.append(tmp)
    return torch.tensor(shifted_remain_layers).cuda()


def get_reduced_hidden(hidden, remain_layers):
    '''
    hidden: batch_size x seq_len x hidden_dim
    remain_layers: batch_size x seq_len
    '''
    indices = []
    max_tokens = 0
    for r in remain_layers:
        remain_index = torch.nonzero(r > 0).reshape(-1)
        if len(remain_index) > max_tokens:
            max_tokens = len(remain_index)
        indices.append(remain_index)

    reduced_hidden = []
    for i, h in enumerate(hidden):
        hidden_dim = h.shape[-1]
        reduced_hidden.append(
            torch.cat([h[indices[i]].reshape(1, -1, hidden_dim),
                      torch.zeros(1, max_tokens-len(indices[i]), hidden_dim).cuda()], dim=1)
        )
    return torch.cat(reduced_hidden, dim=0)


def get_dropped_hidden(hidden, remain_layers):
    '''
    hidden: batch_size x seq_len x hidden_dim
    remain_layers: batch_size x seq_len
    '''
    indices = []
    num_drop_tokens = []
    for r in remain_layers:
        drop_index = torch.nonzero(r.le(0)).reshape(-1)
        num_drop_tokens.append(len(drop_index))
        indices.append(drop_index)

    dropped_hidden = []
    for i, h in enumerate(hidden):
        dropped_hidden.append(h[indices[i]])
    return dropped_hidden, num_drop_tokens


def copy_hidden(hidden, prev_hidden, remain_layers):
    dropped_hidden, num_drop_tokens = get_dropped_hidden(
        prev_hidden, remain_layers)
    bsz, seq_len, hidden_dim = prev_hidden.shape
    new_hidden = []
    for i in range(bsz):
        new_hidden.append(
            torch.cat([hidden[i][:seq_len-num_drop_tokens[i]], dropped_hidden[i]], dim=0).reshape(1, seq_len, hidden_dim)
        )
    return torch.cat(new_hidden, dim=0)


def get_start_piece_outputs(tensor, index):
    errors = (index >= tensor.size(1)).sum().item()
    assert errors == 0, errors
    batch_idx = torch.arange(0, tensor.size(0), device=tensor.device)
    batch_idx = batch_idx.view(-1, 1).expand_as(index)
    return tensor[batch_idx, index]


def get_extended_attention_mask(mask: torch.Tensor, dtype: torch.dtype):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()

    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, src_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def attention_mask_func(attention_scores, attention_mask):
    return attention_scores + attention_mask


class GradientRescaleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        ctx.gd_scale_weight = weight
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_outputs

        return grad_input, grad_weight


gradient_rescale = GradientRescaleFunction.apply


class ElasticBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # self.layernorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(
            config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ElasticBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        # self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            remain_layers=None,
    ):
        reduced_hidden = get_reduced_hidden(hidden_states, remain_layers)
        mixed_query_layer = self.query(reduced_hidden)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + \
                    relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ElasticBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)

        return outputs


class ElasticBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ElasticBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ElasticBertSelfAttention(config)
        self.output = ElasticBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - \
            len(heads)
        self.self.all_head_size = self.self.attention_head_size * \
            self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            remain_layers=None,
    ):
        self_outputs = self.self(
            hidden_states,
            # attention_mask,
            output_attentions,
            remain_layers=remain_layers
        )
        attention_output = self.output(self_outputs[0], get_reduced_hidden(hidden_states, remain_layers))
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ElasticBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ElasticBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ElasticBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ElasticBertAttention(config)
        self.intermediate = ElasticBertIntermediate(config)
        self.output = ElasticBertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            remain_layers=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            remain_layers=remain_layers,
        )
        attention_output = self_attention_outputs[0]

        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ElasticBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ElasticBertEncoder(nn.Module):
    def __init__(self, config, add_pooling_layer=None):
        super().__init__()
        self.config = config
        self.add_pooling_layer = add_pooling_layer
        self.num_base_layers = config.num_base_layers
        self.num_output_layers = config.num_output_layers
        self.num_hidden_layers = config.num_hidden_layers
        self.max_output_layers = config.max_output_layers

        self.layer = nn.ModuleList([ElasticBertLayer(config)
                                   for _ in range(config.num_hidden_layers)])

        assert self.num_base_layers + self.num_output_layers <= self.num_hidden_layers, \
            "The total number of layers must be be greater than or equal to the sum of the number of the base layers and the output layers. "

        assert self.num_output_layers <= self.max_output_layers, \
            "The number of output layers set by the user must be smaller than or equal to the maximum number of output layers."

        self.start_output_layer = None
        self.current_pooler_num = None
        if self.num_output_layers > 1:
            self.start_output_layer = self.num_hidden_layers - self.num_output_layers
            start_pooler_num = self.start_output_layer - self.num_base_layers
            end_pooler_num = self.num_hidden_layers - self.num_base_layers - 1
            if add_pooling_layer:
                self.pooler = nn.ModuleList([ElasticBertPooler(config) if i >= start_pooler_num and
                                             i <= end_pooler_num else None for i in
                                             range(self.max_output_layers)])
        elif self.num_output_layers == 1:
            self.current_pooler_num = self.num_hidden_layers - self.num_base_layers - 1
            if add_pooling_layer:
                self.pooler = nn.ModuleList([ElasticBertPooler(config) if i == self.current_pooler_num
                                             else None for i in range(self.max_output_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            group_output_layers=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        output_sequence_outputs = () if self.num_output_layers > 1 else None
        output_pooled_outputs = () if self.num_output_layers > 1 else None

        final_pooled_output = None

        for i, layer_module in enumerate(self.layer):

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                )
                # layer_outputs = mpu.checkpoint(
                #     create_custom_forward(layer_module),
                #     hidden_states,
                #     attention_mask,
                # )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

            if self.num_output_layers > 1:
                if group_output_layers is None:
                    if i >= self.start_output_layer:
                        if self.training:
                            hidden_states = gradient_rescale(
                                hidden_states, 1.0 / (self.num_hidden_layers - i))
                        output_sequence_outputs += (hidden_states,)
                        if self.add_pooling_layer:
                            pooled_output = self.pooler[i -
                                                        self.start_output_layer](hidden_states)
                            output_pooled_outputs += (pooled_output,)
                        else:
                            output_pooled_outputs += (hidden_states[:, 0],)
                        if self.training:
                            hidden_states = gradient_rescale(
                                hidden_states, (self.num_hidden_layers - i - 1))
                else:
                    if i in group_output_layers:
                        curr_num_output_layers = len(group_output_layers)
                        if self.training:
                            hidden_states = gradient_rescale(hidden_states,
                                                             1.0 / (curr_num_output_layers - group_output_layers.index(
                                                                 i)))
                        output_sequence_outputs += (hidden_states,)
                        if self.add_pooling_layer:
                            pooled_output = self.pooler[i -
                                                        self.start_output_layer](hidden_states)
                            output_pooled_outputs += (pooled_output,)
                        else:
                            output_pooled_outputs += (hidden_states[:, 0],)
                        if self.training:
                            hidden_states = gradient_rescale(hidden_states,
                                                             (curr_num_output_layers - group_output_layers.index(
                                                                 i) - 1))
            elif self.num_output_layers == 1:
                if i == self.num_hidden_layers - 1:
                    if self.add_pooling_layer:
                        final_pooled_output = self.pooler[self.current_pooler_num](
                            hidden_states)
                    else:
                        final_pooled_output = hidden_states[:, 0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                output_sequence_outputs,
                output_pooled_outputs,
                final_pooled_output,
                all_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )

    def adaptive_forward(
            self,
            hidden_states=None,
            current_layer=None,
            attention_mask=None,
            remain_layers=None,
    ):
        layer_outputs = self.layer[current_layer](
            hidden_states,
            attention_mask,
            output_attentions=False,
            remain_layers=remain_layers,
        )

        hidden_states = layer_outputs[0]

        # if self.training:
        #     hidden_states = gradient_rescale(hidden_states, 1.0 / (self.num_hidden_layers-current_layer))

        pooled_output = None
        if self.add_pooling_layer:
            pooled_output = self.pooler[current_layer - self.num_base_layers](
                hidden_states,
            )

        return hidden_states, pooled_output


class ElasticBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ElasticBertConfig
    base_model_prefix = "elasticbert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ElasticBertModel(ElasticBertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.add_pooling_layer = add_pooling_layer
        self.num_base_layers = config.num_base_layers
        self.num_output_layers = config.num_output_layers
        self.num_hidden_layers = config.num_hidden_layers
        self.max_output_layers = config.max_output_layers

        self.embeddings = ElasticBertEmbeddings(config)
        self.encoder = ElasticBertEncoder(
            config, add_pooling_layer=add_pooling_layer)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def update_mask(self, hidden_mask, remain_layers):
        # hidden_mask: batch_size x seq_len x hidden_size, 1 for exited tokens and 0 for remained tokens
        # remain_layers: batch_size x seq_len, indicate the exiting layer for each token
        # TODO: make sure that the padding tokens are zero for param:exit_layers
        batch_size, seq_len, hidden_size = hidden_mask.size()
        hidden_mask = remain_layers.le(0).view(
            batch_size, seq_len, 1).expand_as(hidden_mask).float()
        remain_layers = remain_layers - 1
        return hidden_mask, remain_layers

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            exit_layers=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            group_output_layers=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        extended_attention_mask = get_extended_attention_mask(
            attention_mask, embedding_output.dtype)

        hidden, pooled_output = None, None
        prev_hidden = embedding_output
        # hidden_mask = torch.ones(
        # batch_size, seq_length, self.config.hidden_size)
        # hidden_mask, remain_layers = self.update_mask(
        # hidden_mask, exit_layers)  # mask the padding tokens
        remain_layers = exit_layers - 1
        for i in range(self.num_hidden_layers):
            hidden, pooled_output = self.encoder.adaptive_forward(
                prev_hidden,
                current_layer=i,
                attention_mask=extended_attention_mask,
                remain_layers=remain_layers,
            )
            prev_hidden = copy_hidden(hidden, prev_hidden, remain_layers)
            # prev_hidden = prev_hidden * hidden_mask + \
            # hidden * (1 - hidden_mask)
            remain_layers = shift_nonzero(remain_layers)
            remain_layers = remain_layers - 1
            # hidden_mask, remain_layers = self.update_mask(
            # hidden_mask, remain_layers)
        return (hidden, pooled_output)

        # encoder_outputs = self.encoder(
        #     embedding_output,
        #     attention_mask=extended_attention_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     group_output_layers=group_output_layers,
        # )
        #
        # if self.num_output_layers > 1:
        #     sequence_outputs = encoder_outputs[1]
        #     pooled_output = encoder_outputs[2]
        #
        #     return (sequence_outputs, pooled_output)
        #
        # elif self.num_output_layers == 1:
        #     sequence_outputs = encoder_outputs[0]
        #     pooled_output = encoder_outputs[1]
        #
        #     return (sequence_outputs, pooled_output)


class ElasticBertHeeForSequenceClassification(ElasticBertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.add_pooling_layer = add_pooling_layer

        self.elasticbert = ElasticBertModel(
            config, add_pooling_layer=add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            exit_layers=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.elasticbert(
            input_ids,
            attention_mask=None,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            exit_layers=exit_layers,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        output = outputs[1]

        output = self.dropout(output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'pred': logits
        }

# class ElasticBertForSequenceClassification(ElasticBertPreTrainedModel):
#     def __init__(self, config, add_pooling_layer=True):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config
#         self.add_pooling_layer = add_pooling_layer
#
#         self.elasticbert = ElasticBertModel(config, add_pooling_layer=add_pooling_layer)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#
#         self.init_weights()
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             inputs_embeds=None,
#             labels=None,
#             output_attentions=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
#             config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         outputs = self.elasticbert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#         )
#
#         output = outputs[1][-1]
#
#         output = self.dropout(output)
#         logits = self.classifier(output)
#
#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = nn.MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#
#         return {
#             'loss': loss,
#             'pred': logits
#         }
