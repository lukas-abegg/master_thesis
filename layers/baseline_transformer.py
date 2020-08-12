import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512):
        super(TransformerModel, self).__init__()
        self.ninp = ninp

        self.encoder = Embedding(vocab_size, ninp, embedding_layer)
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len)
        self.decoder = nn.Linear(ninp, vocab_size)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        self.trg_mask = None

        self.init_weights()

    @staticmethod
    def _generate_square_subsequent_mask(sz):
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

        output = self.transformer_decoder(targets, memory, self.trg_mask)

        output = self.decoder(output)

        return output


class Embedding(nn.Module):
    """ Implement input and output embedding """

    def __init__(self, vocab_size, ninp, embedding_layer=None):
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.ninp = ninp

        if embedding_layer is None:
            self.encoder = nn.Embedding(vocab_size, ninp)
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
