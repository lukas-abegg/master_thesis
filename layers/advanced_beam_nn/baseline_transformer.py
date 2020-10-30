from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.nn as nn

import math

import numpy as np


class Embedding(nn.Module):
    """ Implement input and output embedding """

    def __init__(self, vocab_size, ninp, embedding_layer=None):
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.ninp = ninp

        if embedding_layer is None:
            self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=1)
        else:
            self.encoder = embedding_layer

        self.decoder = nn.Linear(ninp, vocab_size, bias=False)

        """ only for language models """
        # self.decoder.weight = self.encoder.weight

    def forward(self, x, inverse=False):
        if inverse:
            return self.decoder(x)

        return self.encoder(x) * np.sqrt(self.ninp)


class PositionalEncoding(nn.Module):

    def __init__(self, ninp, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, ninp)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, ninp, 2).float() * (-math.log(10000.0) / ninp))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, ninp, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(ninp, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, ninp)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, ninp, nheads, nhidden, depth, dropout, embedding_layer, max_len):
        super(TransformerEncoder, self).__init__()

        self.embedding = Embedding(vocab_size, ninp, embedding_layer)
        self.pos_enc = PositionalEncoding(ninp, dropout, max_len)

        self.encoder_layers = nn.TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(ninp=ninp, nhead=nheads, dim_feedforward=nhidden, dropout=dropout),
            num_layers=depth
        )

    def forward(self, sources, src_msk=None):
        """
        args:
           sources: input_sequence, (batch_size, seq_len, embed_size)
        """
        src_pe = self.pos_enc(self.embedding(sources))
        src_pe = src_pe.transpose(0, 1)

        enc_output = self.encoder_layers(src_pe, src_msk)

        return enc_output


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        ninp: the number of expected features in the input (required).
        nhead: the number of heads in the multi head attention (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    """

    def __init__(self, ninp, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(ninp, nhead, dropout=dropout)

        self.ff = FeedForward(ninp, d_ff=dim_feedforward, dropout=dropout)

        self.norm1 = nn.LayerNorm(ninp)
        self.norm2 = nn.LayerNorm(ninp)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.ff(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size, ninp, nheads, nhidden, depth, dropout, embedding_layer, max_len):
        super(TransformerDecoder, self).__init__()

        self.embedding = Embedding(vocab_size, ninp, embedding_layer)
        self.pos_enc = PositionalEncoding(ninp, dropout, max_len)

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(ninp=ninp, nhead=nheads, dim_feedforward=nhidden, dropout=dropout)
             for _ in range(depth)]
        )

    def forward(self, targets, memory, memory_mask=None, trg_mask=None, state=None):

        trg_pe = self.pos_enc(self.embedding(targets))
        trg_pe = trg_pe.transpose(0, 1)

        dec = None

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            if state is None:
                dec = decoder_layer(trg_pe, memory, trg_mask=trg_mask, memory_mask=memory_mask)
            else:  # Use cache
                layer_cache = state.layer_caches[layer_index]
                # print('inputs_mask', inputs_mask)
                dec = decoder_layer(trg_pe, memory, trg_mask=trg_mask, memory_mask=memory_mask, layer_cache=layer_cache)

                state.update_state(
                    layer_index=layer_index,
                    layer_mode='self-attention',
                    key_projected=decoder_layer.self_attention_layer.key_projected,
                    value_projected=decoder_layer.self_attention_layer.value_projected,
                )
                state.update_state(
                    layer_index=layer_index,
                    layer_mode='memory-attention',
                    key_projected=decoder_layer.memory_attention_layer.key_projected,
                    value_projected=decoder_layer.memory_attention_layer.value_projected,
                )

        generated = self.embedding(dec, inverse=True)  # (batch_size, seq_len, vocab_size)
        return generated, state

    @staticmethod
    def init_decoder_state():
        return DecoderState()


class DecoderState:

    def __init__(self):
        self.previous_inputs = torch.tensor([])
        self.layer_caches = defaultdict(lambda: {'self-attention': None, 'memory-attention': None})

    def update_state(self, layer_index, layer_mode, key_projected, value_projected):
        self.layer_caches[layer_index][layer_mode] = {
            'key_projected': key_projected,
            'value_projected': value_projected
        }

    def beam_update(self, positions):
        for layer_index in self.layer_caches:
            for mode in ('self-attention', 'memory-attention'):
                if self.layer_caches[layer_index][mode] is not None:
                    for projection in self.layer_caches[layer_index][mode]:
                        cache = self.layer_caches[layer_index][mode][projection]
                        if cache is not None:
                            cache.data.copy_(cache.data.index_select(0, positions))


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        ninp: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention zzz_bert_models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    """

    def __init__(self, ninp, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(ninp, nhead, dropout=dropout, mode='self-attention')
        self.multihead_attn = MultiheadAttention(ninp, nhead, dropout=dropout, mode='memory-attention')

        self.ff = FeedForward(ninp, d_ff=dim_feedforward, dropout=dropout)

        self.norm1 = nn.LayerNorm(ninp)
        self.norm2 = nn.LayerNorm(ninp)
        self.norm3 = nn.LayerNorm(ninp)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, trg, memory, trg_mask=None, memory_mask=None, layer_cache=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            trg: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            trg_mask: the mask for the trg sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            layer_cache: layer cache
        Shape:
            see the docs in Transformer class.
        """
        trg2 = self.self_attn(trg, trg, trg, trg_mask, layer_cache)
        trg = trg + self.dropout1(trg2)
        trg = self.norm1(trg)

        trg2 = self.multihead_attn(trg, memory, memory, memory_mask, layer_cache)

        trg = trg + self.dropout2(trg2)
        trg = self.norm2(trg)

        trg2 = self.ff(trg)
        trg = trg + self.dropout3(trg2)
        trg = self.norm3(trg)

        return trg


class MultiheadAttention(nn.Module):
    def __init__(self, ninp, nhead, dropout, mode='self-attention'):
        super(MultiheadAttention, self).__init__()

        assert ninp % nhead == 0
        assert mode in ('self-attention', 'memory-attention')

        self.d_head = ninp // nhead
        self.heads_count = nhead
        self.mode = mode
        self.query_projection = nn.Linear(ninp, nhead * self.d_head)
        self.key_projection = nn.Linear(ninp, nhead * self.d_head)
        self.value_projection = nn.Linear(ninp, nhead * self.d_head)
        self.final_projection = nn.Linear(ninp, nhead * self.d_head)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=3)

        self.attention = None
        # For cache
        self.key_projected = None
        self.value_projected = None

    def forward(self, query, key, value, mask=None, layer_cache=None):
        """

        Args:
            query: (batch_size, query_len, model_dim)
            key: (batch_size, key_len, model_dim)
            value: (batch_size, value_len, model_dim)
            mask: (batch_size, query_len, key_len)
            layer_cache: layer cache for predictions
        """
        # print('attention mask', mask)
        batch_size, query_len, d_model = query.size()

        d_head = d_model // self.heads_count

        query_projected = self.query_projection(query)
        # print('query_projected', query_projected.shape)
        if layer_cache is None or layer_cache[self.mode] is None:  # Don't use cache
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        else:  # Use cache
            if self.mode == 'self-attention':
                key_projected = self.key_projection(key)
                value_projected = self.value_projection(value)

                key_projected = torch.cat([key_projected, layer_cache[self.mode]['key_projected']], dim=1)
                value_projected = torch.cat([value_projected, layer_cache[self.mode]['value_projected']], dim=1)
            else:
                key_projected = layer_cache[self.mode]['key_projected']
                value_projected = layer_cache[self.mode]['value_projected']

        # For cache
        self.key_projected = key_projected
        self.value_projected = value_projected

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(batch_size, query_len, self.heads_count, d_head).transpose(1,
                                                                                                      2)  # (batch_size, heads_count, query_len, d_head)
        # print('query_heads', query_heads.shape)
        # print(batch_size, key_len, self.heads_count, d_head)
        # print(key_projected.shape)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count, d_head).transpose(1,
                                                                                                2)  # (batch_size, heads_count, key_len, d_head)
        value_heads = value_projected.view(batch_size, value_len, self.heads_count, d_head).transpose(1,
                                                                                                      2)  # (batch_size, heads_count, value_len, d_head)

        attention_weights = self.scaled_dot_product(query_heads,
                                                    key_heads)  # (batch_size, heads_count, query_len, key_len)

        if mask is not None:
            # print('mode', self.mode)
            # print('mask', mask.shape)
            # print('attention_weights', attention_weights.shape)
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_expanded, -1e18)

        self.attention = self.softmax(attention_weights)  # Save attention to the object
        # print('attention_weights', attention_weights.shape)
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads)  # (batch_size, heads_count, query_len, d_head)
        # print('context_heads', context_heads.shape)
        context_sequence = context_heads.transpose(1, 2).contiguous()  # (batch_size, query_len, heads_count, d_head)
        context = context_sequence.view(batch_size, query_len, d_model)  # (batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        # print('final_output', final_output.shape)

        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        """

        Args:
             query_heads: (batch_size, heads_count, query_len, d_head)
             key_heads: (batch_size, heads_count, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)  # (batch_size, heads_count, query_len, key_len)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class Transformer(nn.Module):
    """Implement transformer model.
    Args:
        vocab_size: number of unique tokens in vocabulary.
        ninp: dimension of embedding.
        nhidden: size of hidden layer in feed forward sub-layers.
        nheads: number of attention heads.
        max_len: maximum sentence length used to pre-compute positional encoder.
        depth: number of encoder and decoder sub-layers.
    """

    def __init__(
            self,
            vocab_size,
            ninp,
            nhidden,
            nheads,
            depth,
            dropout=0.1,
            embedding_layer=None,
            max_len=512
    ):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(vocab_size, ninp, nheads, nhidden, depth, dropout, embedding_layer, max_len)

        self.decoder = TransformerDecoder(vocab_size, ninp, nheads, nhidden, depth, dropout, embedding_layer, max_len)

        self._reset_parameters()

        self.ninp = ninp
        self.nhead = nheads
        self.trg_mask = None

    def forward(self, sources, targets):

        if self.trg_mask is None or self.trg_mask.size(0) != len(targets):
            self.trg_mask = self.generate_square_subsequent_mask(len(targets)).to(targets.device)

        # Encoder Part
        memory = self.encoder(sources)

        # Decoder Part
        return self.decoder(targets, memory, trg_mask=self.trg_mask)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
