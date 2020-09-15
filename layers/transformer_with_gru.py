import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.utils.gru import GRUGate


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512):
        super(TransformerModel, self).__init__()
        self.ninp = ninp

        self.encoder = Embedding(ntoken, ninp, embedding_layer)
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len)
        self.decoder = nn.Linear(ninp, ntoken)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)

        self.trg_mask = None

        self.init_weights()

    @staticmethod
    def generate_key_padding_mask(seq):
        # x: (batch_size, seq_len)
        batch_size, seq_len = seq.size()
        pad_mask = seq == 0  # (batch_size, seq_len)
        return pad_mask

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, targets):
        if self.trg_mask is None or self.trg_mask.size(0) != len(targets):
            self.trg_mask = self.generate_square_subsequent_mask(len(targets)).to(targets.device)

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        memory = self.transformer_encoder(src)

        output = self.transformer_decoder(targets, memory, tgt_mask=self.trg_mask)

        output = self.decoder(output)

        return output


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

        self.gate_mha = GRUGate(ninp)
        self.gate_mlp = GRUGate(ninp)

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
        ninp: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention zzz_bert_models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    """

    def __init__(self, ninp, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(ninp, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(ninp, nhead, dropout=dropout)

        self.ff = FeedForward(ninp, d_ff=dim_feedforward, dropout=dropout)

        self.gate_mha = GRUGate(ninp)
        self.gate_mha2 = GRUGate(ninp)
        self.gate_mlp = GRUGate(ninp)

        self.norm1 = nn.LayerNorm(ninp)
        self.norm2 = nn.LayerNorm(ninp)
        self.norm3 = nn.LayerNorm(ninp)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.memory_attention_weights = None

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

        trg2 = self.norm1(trg)
        trg2 = self.self_attn(trg2, trg2, trg2, trg_mask, layer_cache)
        trg = self.gate_mha(trg, F.relu(self.dropout1(trg2)))

        trg2 = self.norm2(trg)
        trg2, attn_output_weights = self.multihead_attn(trg2, memory, memory, memory_mask, layer_cache)

        trg = self.gate_mha2(trg, F.relu(self.dropout2(trg2)))

        trg2 = self.norm3(trg)
        trg2 = self.ff(trg2)
        trg = self.gate_mlp(trg, F.relu(self.dropout3(trg2)))

        return trg


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


class Embedding(nn.Module):
    """ Implement input and output embedding """

    def __init__(self, vocab_size, ninp, embedding_layer=None):
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.ninp = ninp

        if embedding_layer is None:
            self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=0)
        else:
            self.encoder = embedding_layer

        """ only for language models """
        # self.decoder.weight = self.encoder.weight

    def forward(self, x):
        return self.encoder(x)


class PositionalEncoding(nn.Module):
    """ Implement the PE function. """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
