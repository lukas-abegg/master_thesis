import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, bert_model, d_model=512, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", source_vocab_length=60000, target_vocab_length=60000, load_embedding_weights=False):
        super(Transformer, self).__init__()

        self.source_embedding = nn.Embedding(source_vocab_length, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.target_embedding = nn.Embedding(target_vocab_length, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.out = nn.Linear(d_model, target_vocab_length)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.load_embedding_weights = load_embedding_weights
        self.bert_model = bert_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        src = src.transpose(0, 1)
        src = self.source_embedding(src)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        tgt = tgt.transpose(0, 1)
        tgt = self.target_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        tgt = tgt.transpose(0, 1)

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        output = self.out(output)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        if self.load_embedding_weights:
            self.source_embedding.weight = self.bert_model.embeddings.word_embeddings.weight
            self.target_embedding.weight = self.bert_model.embeddings.word_embeddings.weight

        self.source_embedding.weight.requires_grad = False
        self.target_embedding.weight.requires_grad = False
        self.out.weight.requires_grad = False
        self.out.bias.requires_grad = False