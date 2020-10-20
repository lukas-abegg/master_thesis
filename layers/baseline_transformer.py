import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512,
                 trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=True):
        super(TransformerModel, self).__init__()
        self.ninp = ninp

        self.encoder = Encoder(vocab_size, ninp, nhead, nhid, nlayers, dropout, embedding_layer, max_len)
        self.decoder = Decoder(vocab_size, ninp, nhead, nhid, nlayers, dropout, embedding_layer, max_len)

        self.trg_word_prj = nn.Linear(ninp, vocab_size)

        self.init_weights()

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.encoder.weight
            self.x_logit_scale = (ninp ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.encoder.weight = self.decoder.trg_word_emb.encoder.weight

    @staticmethod
    def generate_key_padding_mask(seq):
        # x: (batch_size, seq_len)
        pad_mask = seq == 0  # (batch_size, seq_len)
        return pad_mask

    @staticmethod
    def generate_square_subsequent_mask(seq):
        sz_b, len_s = seq.size()
        mask = (torch.triu(torch.ones(len_s, len_s)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        initrange = 0.1
        self.encoder.src_word_emb.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.trg_word_emb.encoder.weight.data.uniform_(-initrange, initrange)
        self.trg_word_prj.bias.data.zero_()
        self.trg_word_prj.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, targets):
        src_key_padding_mask = self.generate_key_padding_mask(src).to(src.device)
        trg_key_padding_mask = self.generate_key_padding_mask(targets).to(targets.device)

        trg_mask = self.generate_square_subsequent_mask(targets).to(targets.device)

        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        dec_output = self.decoder(tgt=targets, memory=memory, tgt_mask=trg_mask, memory_mask=None,
                                  tgt_key_padding_mask=trg_key_padding_mask,
                                  memory_key_padding_mask=src_key_padding_mask)

        return self.trg_word_prj(dec_output) * self.x_logit_scale


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512):
        super(Encoder, self).__init__()

        self.ninp = ninp
        self.src_word_emb = Embedding(vocab_size, ninp, embedding_layer)
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)

        self.layer_norm = nn.LayerNorm(ninp)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, self.layer_norm)

    def forward(self, src_seq, src_key_padding_mask=None):
        src = self.src_word_emb(src_seq) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        enc_output = src.transpose(0, 1)

        memory = self.transformer_encoder(enc_output, src_key_padding_mask=src_key_padding_mask)

        return memory


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512):
        super(Decoder, self).__init__()

        self.ninp = ninp
        self.trg_word_emb = Embedding(vocab_size, ninp, embedding_layer)
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)

        self.layer_norm = nn.LayerNorm(ninp)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers, self.layer_norm)

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt = self.trg_word_emb(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)

        dec_output = tgt.transpose(0, 1)

        output = self.transformer_decoder(tgt=dec_output, memory=memory, tgt_mask=tgt_mask,
                                          memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)

        output = output.transpose(0, 1)

        return output


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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
