import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class TransformerModel(nn.Module):

    def __init__(self, src_vocab_length, trg_vocab_length, ninp, nhead, nhid, nlayers, dropout=0.1,
                 embedding_layer=None, max_len=512,
                 trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=False):
        super(TransformerModel, self).__init__()

        self.source_embedding = nn.Embedding(src_vocab_length, ninp)
        self.pos_encoder = PositionalEncoding(ninp)
        encoder_layer = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout, "relu")
        encoder_norm = nn.LayerNorm(ninp)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        self.target_embedding = nn.Embedding(trg_vocab_length, ninp)
        decoder_layer = nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout, "relu")
        decoder_norm = nn.LayerNorm(ninp)
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)
        self.out = nn.Linear(ninp, trg_vocab_length)
        self._reset_parameters()
        self.d_model = ninp
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        src = self.source_embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        tgt = self.target_embedding(tgt)
        tgt = self.pos_encoder(tgt)
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

#         self.ninp = ninp
#
#         self.encoder = Encoder(src_vocab_length, ninp, nhead, nhid, nlayers, dropout, embedding_layer, max_len)
#         self.decoder = Decoder(trg_vocab_length, ninp, nhead, nhid, nlayers, dropout, embedding_layer, max_len)
#
#         self.trg_word_prj = nn.Linear(ninp, trg_vocab_length)
#
#         self.init_weights()
#
#         if trg_emb_prj_weight_sharing:
#             # Share the weight between target word embedding & last dense layer
#             self.trg_word_prj.weight = self.decoder.trg_word_emb.encoder.weight
#
#         if emb_src_trg_weight_sharing:
#             self.encoder.src_word_emb.encoder.weight = self.decoder.trg_word_emb.encoder.weight
#
#     def init_weights(self):
#         r"""Initiate parameters in the transformer model."""
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#         initrange = 0.01
#         self.encoder.src_word_emb.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.trg_word_emb.encoder.weight.data.uniform_(-initrange, initrange)
#         self.trg_word_prj.bias.data.zero_()
#         self.trg_word_prj.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, src, tgt, tgt_mask=None,
#                 memory_mask=None, src_key_padding_mask=None,
#                 tgt_key_padding_mask=None,
#                 memory_key_padding_mask=None):
#
#         if src.size(1) != tgt.size(1):
#             raise RuntimeError("the batch number of src and tgt must be equal")
#
#         memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
#
#         dec_output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
#                                   tgt_key_padding_mask=tgt_key_padding_mask,
#                                   memory_key_padding_mask=memory_key_padding_mask)
#
#         return self.trg_word_prj(dec_output)
#
#
# class Encoder(nn.Module):
#     """ A encoder model with self attention mechanism. """
#
#     def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512):
#         super(Encoder, self).__init__()
#
#         self.ninp = ninp
#         self.src_word_emb = Embedding(vocab_size, ninp, embedding_layer)
#         self.pos_encoder = PositionalEncoding(ninp, dropout, max_len)
#
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
#
#         self.layer_norm = nn.LayerNorm(ninp)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, self.layer_norm)
#
#     def forward(self, src_seq, src_key_padding_mask=None):
#         src = self.src_word_emb(src_seq) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#
#         memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
#
#         return memory
#
#
# class Decoder(nn.Module):
#     """ A decoder model with self attention mechanism. """
#
#     def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512):
#         super(Decoder, self).__init__()
#
#         self.ninp = ninp
#         self.trg_word_emb = Embedding(vocab_size, ninp, embedding_layer)
#         self.pos_encoder = PositionalEncoding(ninp, dropout, max_len)
#
#         decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
#
#         self.layer_norm = nn.LayerNorm(ninp)
#         self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers, self.layer_norm)
#
#     def forward(self, tgt, memory, tgt_mask=None,
#                 memory_mask=None, tgt_key_padding_mask=None,
#                 memory_key_padding_mask=None):
#         tgt = self.trg_word_emb(tgt) * math.sqrt(self.ninp)
#         tgt = self.pos_encoder(tgt)
#
#         output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask,
#                                           memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
#                                           memory_key_padding_mask=memory_key_padding_mask)
#
#         return output
#
#
# class Embedding(nn.Module):
#     """ Implement input and output embedding """
#
#     def __init__(self, vocab_size, ninp, embedding_layer=None):
#         super(Embedding, self).__init__()
#
#         self.vocab_size = vocab_size
#         self.ninp = ninp
#
#         if embedding_layer is None:
#             self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=1)
#         else:
#             self.encoder = embedding_layer
#
#     def forward(self, x):
#         return self.encoder(x)
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.d_model = d_model
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x * math.sqrt(self.d_model)
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
