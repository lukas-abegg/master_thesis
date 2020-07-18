import copy
from collections import OrderedDict
from torch.autograd import Variable

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from layers.utils.gru import GRUGate


class Embedding(nn.Module):
    """Implement input and output embedding with tied weights."""

    def __init__(self, vocab_size, model_dim):
        super(Embedding).__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim

        self.encoder = nn.Embedding(vocab_size, model_dim)
        self.decoder = nn.Linear(model_dim, vocab_size, bias=False)

        self.decoder.weight = self.encoder.weight

    def forward(self, x, inverse=False):
        if inverse:
            return self.decoder(x)

        return self.encoder(x) * np.sqrt(self.model_dim)


class PositionalEncoder(nn.Module):
    """Implement the Positional Encoder function.
    Based on https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """

    def __init__(self, d_model, max_len=80, p=0.1):
        super(PositionalEncoder).__init__()

        # Compute the positional encodings once in log space.
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pe', pos_enc)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = x + Variable(self.pos_enc[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward).__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, use_gate=False):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.ff = FeedForward(d_model, d_ff=dim_feedforward, dropout=dropout)

        if self.use_gate:
            self.gate_mha = GRUGate(d_model)
            self.gate_mlp = GRUGate(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        if not self.use_gate:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            src2 = self.ff(src)
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        else:
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = self.gate_mha(src, F.relu(self.dropout1(src2)))

            src2 = self.norm2(src)
            src2 = self.ff(src2)
            src = self.gate_mlp(src, F.relu(self.dropout2(src2)))

        return src


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, use_gate=False):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.ff = FeedForward(d_model, d_ff=dim_feedforward, dropout=dropout)

        if self.use_gate:
            self.gate_mha = GRUGate(d_model)
            self.gate_mha2 = GRUGate(d_model)
            self.gate_mlp = GRUGate(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, trg, memory, trg_mask=None, memory_mask=None,
                trg_key_padding_mask=None, memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            trg: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            trg_mask: the mask for the trg sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            trg_key_padding_mask: the mask for the trg keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if not self.use_gate:
            trg2 = self.self_attn(trg, trg, trg, attn_mask=trg_mask,
                                  key_padding_mask=trg_key_padding_mask)[0]
            trg = trg + self.dropout1(trg2)
            trg = self.norm1(trg)

            trg2 = self.multihead_attn(trg, memory, memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            trg = trg + self.dropout2(trg2)
            trg = self.norm2(trg)

            trg2 = self.ff(trg)
            trg = trg + self.dropout3(trg2)
            trg = self.norm3(trg)
        else:
            trg2 = self.norm1(trg)
            trg2 = self.self_attn(trg2, trg2, trg2, attn_mask=trg_mask,
                                  key_padding_mask=trg_key_padding_mask)[0]
            trg = self.gate_mha(trg, F.relu(self.dropout1(trg2)))

            trg2 = self.norm2(trg)
            trg2 = self.multihead_attn(trg2, memory, memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            trg = self.gate_mha2(trg, F.relu(self.dropout2(trg2)))

            trg2 = self.norm3(trg)
            trg2 = self.ff(trg2)
            trg = self.gate_mlp(trg, F.relu(self.dropout3(trg2)))

        return trg


class Transformer(nn.Module):
    """Implement transformer model.
    Args:
        vocab_size: number of unique tokens in vocabulary.
        model_dim: dimension of embedding.
        hidden_dim: size of hidden layer in feed forward sub-layers.
        nheads: number of attention heads.
        max_len: maximum sentence length used to pre-compute positional encoder.
        depth: number of encoder and decoder sub-layers.
        device: CPU or GPU as device
    """

    def __init__(
            self,
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            depth,
            use_gate,
            p=0.1,
            max_len=5000,
            device="cpu"
    ):
        super(Transformer).__init__()
        self.embedding = Embedding(vocab_size, model_dim)
        self.pos_enc = PositionalEncoder(model_dim, max_len, p)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(d_model=model_dim, nhead=nheads, dim_feedforward=hidden_dim, dropout=p, use_gate=use_gate),
            num_layers=depth,
            norm=nn.LayerNorm(model_dim)
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(d_model=model_dim, nhead=nheads, dim_feedforward=hidden_dim, dropout=p, use_gate=use_gate),
            num_layers=depth,
            norm=nn.LayerNorm(model_dim)
        )

        self.apply(self._init_weights)

        self.src_embedding = None
        self.trg_embedding = None

        self.device = device

    def forward(self, src, trg, src_mask, trg_mask):
        right_shift = torch.zeros((trg.shape[0], 1), dtype=torch.long, device=self.device)
        trg_rs = torch.cat([right_shift, trg], dim=1)[:, :-1]

        self.src_embedding = self.embedding(src)
        self.trg_embedding = self.embedding(trg_rs)

        src_pe = self.pos_enc(self.src_embedding)
        trg_pe = self.pos_enc(self.trg_embedding)

        enc_output = self.encoder(src_pe, mask=src_mask)

        dec = self.decoder(tgt=trg_pe, memory=enc_output, tgt_mask=trg_mask, memory_mask=src_mask)

        return self.embedding(dec, inverse=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.1)

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

