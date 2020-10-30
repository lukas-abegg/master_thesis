import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512,
                 trg_emb_prj_weight_sharing=True):
        super(TransformerModel, self).__init__()
        self.ninp = ninp

        self.encoder = Encoder(vocab_size, ninp, nhead, nhid, nlayers, dropout, max_len)
        self.decoder = Decoder(vocab_size, ninp, nhead, nhid, nlayers, dropout, max_len)

        self.trg_word_prj = nn.Linear(ninp, vocab_size)

        self.init_weights(embedding_layer)

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.encoder.weight
            self.x_logit_scale = (ninp ** -0.5)

    def init_weights(self, embedding_layer):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.encoder.src_word_emb.encoder.weight = embedding_layer.embeddings.word_embeddings.weight
        self.decoder.trg_word_emb.encoder.weight = embedding_layer.embeddings.word_embeddings.weight

        for p in self.encoder.src_word_emb.encoder.parameters():
            p.requires_grad = False

        for p in self.decoder.trg_word_emb.encoder.parameters():
            p.requires_grad = False

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

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, max_len=512):
        super(Encoder, self).__init__()

        self.ninp = ninp
        self.src_word_emb = Embedding(vocab_size, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(ninp, eps=1e-6)

    def forward(self, src_seq, src_key_padding_mask=None):
        src = self.src_word_emb(src_seq) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        enc_output = self.dropout(src)
        enc_output = self.layer_norm(enc_output)

        enc_output = enc_output.transpose(0, 1)

        memory = self.transformer_encoder(enc_output, src_key_padding_mask=src_key_padding_mask)

        return memory


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, max_len=512):
        super(Decoder, self).__init__()

        self.ninp = ninp
        self.trg_word_emb = Embedding(vocab_size, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(ninp, eps=1e-6)

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt = self.trg_word_emb(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        dec_output = self.dropout(tgt)
        dec_output = self.layer_norm(dec_output)

        dec_output = dec_output.transpose(0, 1)

        output = self.transformer_decoder(tgt=dec_output, memory=memory, tgt_mask=tgt_mask,
                                          memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)

        output = output.transpose(0, 1)

        return output


class Embedding(nn.Module):
    """ Implement input and output embedding """

    def __init__(self, vocab_size, ninp):
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.ninp = ninp

        self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=1)

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
