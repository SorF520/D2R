import math
import copy
import torch
from torch import nn, Tensor, device

from transformers.activations import ACT2FN

from transformers.modeling_utils import (
    apply_chunking_to_forward,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling
)
import torch.nn.functional as F

from models.XModules import SelfEncoder, CrossModalAlignment, GraphReasoning, SELayer, l2norm, SoftContrastiveLoss, AmbiguityLearning, js_div, Block, DiffLoss

from models.InteractionModule import InteractionModule, Reversed_InteractionModule



def get_extended_attention_mask(attention_mask, input_shape, device):
    """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.long)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def get_head_mask(
        head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
):
    """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `argsional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `argsional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
    head_mask = [None] * num_hidden_layers

    return head_mask


# Vision
class CLIPVisionEmbeddings(nn.Module):
    """
    CLIP Embedding from transformers (original version)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        # default num-patches: 7
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
            past_key_values: torch.Tensor = None,
    ):
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
            past_key_values: torch.Tensor = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `argsional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Bert
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

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


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fusion_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# CrossAttention Part
class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        # BertCoAttention is the same as BertSelfAttention
        self.self = BertCoAttention(config)

        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num=1):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, attention_mask):
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, attention_mask)
        return s1_hidden_states


class UnimoEncoder(nn.Module):
    def __init__(self, vision_config, text_config):
        super(UnimoEncoder, self).__init__()
        self.vision_config = vision_config
        self.text_config = text_config

        self.vision_layers = nn.ModuleList(
            [CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.ModuleList([BertLayer(text_config) for _ in range(text_config.num_hidden_layers)])

    def forward(self,
                vision_embeds=None,
                text_embeds=None,
                attention_mask=None,
                head_mask=None,

                output_attentions=None,
                output_hidden_states=None,
                return_dict=None
                ):

        assert self.vision_config.num_hidden_layers == self.text_config.num_hidden_layers

        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None

        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds
        attention_mask = attention_mask

        # Vision
        for idx in range(self.vision_config.num_hidden_layers):
            if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states,)

            vision_layer_module = self.vision_layers[idx]
            vision_layer_output = vision_layer_module(
                vision_hidden_states,
                output_attentions=output_attentions,
                past_key_values=None,
            )
            vision_hidden_states = vision_layer_output[0]

            if output_attentions:
                all_vision_attentions = all_vision_attentions + (vision_layer_output[1],)

        # Text
        for idx in range(self.text_config.num_hidden_layers):
            if output_hidden_states:
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states,)

            text_layer_module = self.text_layer[idx]
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            text_layer_output = text_layer_module(
                text_hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions
            )
            text_hidden_states = text_layer_output[0]
            if output_attentions:
                all_text_attentions = all_text_attentions + (text_layer_output[1],)

        if not return_dict:
            return tuple(
                v for v in [
                    text_hidden_states,
                    all_text_hidden_states,
                    all_text_attentions,
                ] if v is not None)

        return BaseModelOutput(
            last_hidden_state=text_hidden_states,
            hidden_states=all_text_hidden_states,
            attentions=all_text_attentions), \
            BaseModelOutput(
                last_hidden_state=vision_hidden_states,
                hidden_states=all_vision_hidden_states,
                attentions=all_vision_attentions)



class UnimoModel(nn.Module):
    def __init__(self, args, vision_config, text_config, add_pooling_layer=True, num_self_layer=1):
        super().__init__()
        self.args = args
        self.vision_config = vision_config
        self.text_config = text_config

        # vision
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

        # text
        self.text_embeddings = BertEmbeddings(text_config)

        self.encoder = UnimoEncoder(vision_config, text_config)

        # self-attention
        self.self_text = nn.ModuleList([BertLayer(text_config) for _ in range(num_self_layer)])
        self.text_cls_pool = BertPooler(text_config)
        self.self_vision = nn.ModuleList([CLIPEncoderLayer(vision_config) for _ in range(num_self_layer)])
        self.vision_cls_pool = BertPooler(vision_config)

        # # MildTriple Loss
        # self.mild_loss = SoftContrastiveLoss(alpha=args.beta, margin=args.mild_margin, max_violation=True,
        #                                      threshold_hetero=args.hetero, threshold_homo=args.homo)
        # self.fc_1 = nn.Linear(text_config.hidden_size, text_config.hidden_size)
        # self.fc_2 = nn.Linear(text_config.hidden_size, text_config.hidden_size)

        # self.diff_loss = DiffLoss(args)
        # self.fc_text = nn.Sequential(
        #     nn.Linear(args.max_seq, 50),
        #     nn.ReLU(True),
        #     nn.Linear(50, 50)
        # )
        # self.affine1 = nn.Parameter(torch.empty(size=(768, 768)))
        # nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        # self.affine2 = nn.Parameter(torch.empty(size=(768, 768)))
        # nn.init.xavier_uniform_(self.affine2.data, gain=1.414)

        # # SE-block
        # self.SE_fusion = SELayer(channel=2, reduction=16)
        # Bilinear Pooling
        self.block_fusion = Block([768, 768], 768)

        self.text_pool = BertPooler(text_config)
        self.vision_pool = BertPooler(text_config)
        
        self.itr_module = InteractionModule(args, num_layer_routing=args.DR_step, num_cells=6, path_hid=128)
        self.Reversed_itr_module = Reversed_InteractionModule(args, num_layer_routing=args.DR_step, num_cells=6, path_hid=128)

        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,

                pixel_values=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        # pre vision
        vision_embedding_output = self.vision_embeddings(pixel_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # pre text
        input_shape = input_ids.shape
        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            raise ValueError("token_type_ids is None!")


        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, device)

        text_embedding_output = self.text_embeddings(input_ids=input_ids,
                                                     position_ids=position_ids,
                                                     token_type_ids=token_type_ids)

        encoder_text_out, encoder_vision_out = self.encoder(
            vision_embeds=vision_embedding_output,
            text_embeds=text_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        text_encode_out = encoder_text_out.last_hidden_state
        vision_encode_out = encoder_vision_out.last_hidden_state

        # Self-Attention
        text_output = text_encode_out
        vision_output = vision_encode_out

        for text_layer_module in self.self_text:
            text_output = text_layer_module(text_output, extended_attention_mask, output_attentions=False)[0]
        text_cls_output = self.text_cls_pool(text_output)      # (32, 768)

        for vision_layer_module in self.self_vision:
            vision_output = vision_layer_module(vision_output, output_attentions=False)[0]
        vision_cls_output = self.vision_cls_pool(vision_output)       # (32, 768)

        sim_mat, sim_paths = self.itr_module(text_encode_out, vision_encode_out)
        Reversed_sim_mat, Reversed_sim_paths = self.Reversed_itr_module(text_encode_out, vision_encode_out)

        sim_text = torch.matmul(text_cls_output, text_cls_output.transpose(-1, -2))   # (bsz, bsz)
        sim_vision = torch.matmul(vision_cls_output, vision_cls_output.transpose(-1, -2))  # (bsz, bsz)

        # JS散度 -> 双向KL散度    (bsz, bsz)
        js_loss = - self.args.weight_js_1 * js_div(sim_paths, sim_text) - self.args.weight_js_2 * js_div(Reversed_sim_paths, sim_vision)

        # # MildTriple Loss     文本、图片  false negative
        # mild_loss = self.mild_loss(torch.nn.functional.normalize(self.fc_1(text_cls_output)),
        #                      torch.nn.functional.normalize(self.fc_2(sim_paths.squeeze(-2))))

        # # (bsz, 50, 768)
        # diff_loss = self.args.weight_diff * self.diff_loss(self.fc_text(sim_mat[0].transpose(-1, -2)).transpose(-1, -2),
        #                                                    Reversed_sim_mat[0])
        #
        # # BiAffine
        # A1 = F.softmax(torch.bmm(torch.matmul(sim_mat[0], self.affine1), torch.transpose(Reversed_sim_mat[0], 1, 2)),
        #                dim=-1)
        # A2 = F.softmax(torch.bmm(torch.matmul(Reversed_sim_mat[0], self.affine2), torch.transpose(sim_mat[0], 1, 2)),
        #                dim=-1)
        #
        # sim_mat_new = torch.bmm(A1, Reversed_sim_mat[0])
        # Reversed_sim_mat_new = torch.bmm(A2, sim_mat[0])

        # diff_loss = 0

        # Fusion
        text_pooled_output = self.text_pool(sim_mat[0])  # (32, 768)
        image_pooled_output = self.vision_pool(Reversed_sim_mat[0])  # (32, 768)

        # # SE-block    modality-wise attention
        # combine_representation = torch.cat((text_pooled_output.unsqueeze(1), image_pooled_output.unsqueeze(1)), dim=1)  # (bsz, 2, 768)
        #
        # # (bsz, 2, 768)     (bsz, 2)
        # combine_output, channel_weight = self.SE_fusion(combine_representation)
        #
        # # (bsz, 768)
        # output = torch.sum(combine_output, dim=1)

        # (bsz, 768)
        output = self.block_fusion([text_pooled_output, image_pooled_output])

        # # Pooler for classification
        # pooled_out_text = self.text_pooler(sim_mat[0])

        return BaseModelOutputWithPooling(
            last_hidden_state=encoder_text_out.last_hidden_state,
            pooler_output=output,
            hidden_states=encoder_text_out.hidden_states,
            attentions=encoder_text_out.attentions
        ), js_loss
