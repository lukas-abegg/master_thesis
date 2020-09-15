import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512):
        super(TransformerModel, self).__init__()
        self.ninp = ninp

        self.encoder = Encoder(vocab_size, ninp, nhead, nhid, nlayers, dropout, embedding_layer, max_len)
        self.decoder = Decoder(vocab_size, ninp, nhead, nhid, nlayers, dropout, embedding_layer, max_len)

        self.trg_word_prj = nn.Linear(ninp, vocab_size)

        self.init_weights(embedding_layer)

    @staticmethod
    def generate_key_padding_mask(seq):
        # x: (batch_size, seq_len)
        pad_mask = seq == 0  # (batch_size, seq_len)
        attn_mask = pad_mask.int().masked_fill(pad_mask == 0, 1).masked_fill(pad_mask == 1, 0)
        return pad_mask, attn_mask

    @staticmethod
    def generate_square_subsequent_mask(seq):
        sz_b, len_s = seq.size()
        mask = (torch.triu(torch.ones(len_s, len_s)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self, embedding_layer):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.encoder.src_word_emb = embedding_layer

        for p in self.encoder.src_word_emb.parameters():
            p.requires_grad = False

        self.decoder.trg_word_emb = embedding_layer

        for p in self.decoder.trg_word_emb.parameters():
            p.requires_grad = False

    def forward(self, src, targets):
        src_key_padding_mask, src_attn_mask = self.generate_key_padding_mask(src)
        src_key_padding_mask.to(src.device)
        src_attn_mask.to(src.device)

        trg_key_padding_mask, trg_attn_mask = self.generate_key_padding_mask(targets)
        trg_key_padding_mask.to(targets.device)
        trg_attn_mask.to(targets.device)

        trg_mask = self.generate_square_subsequent_mask(targets).to(targets.device)

        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask, src_attn_mask=src_attn_mask)

        dec_output = self.decoder(tgt=targets, memory=memory, tgt_mask=trg_mask, memory_mask=None,
                                  tgt_key_padding_mask=trg_key_padding_mask,
                                  memory_key_padding_mask=src_key_padding_mask, tgt_attn_mask=trg_attn_mask)

        return self.trg_word_prj(dec_output)


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512):
        super(Encoder, self).__init__()

        self.ninp = ninp
        self.src_word_emb = embedding_layer

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(ninp, eps=1e-6)

    def forward(self, src_seq, src_key_padding_mask=None, src_attn_mask=None):
        src = self.src_word_emb(src_seq, src_attn_mask)[0]

        enc_output = self.layer_norm(src)

        enc_output = enc_output.transpose(0, 1)

        memory = self.transformer_encoder(enc_output, src_key_padding_mask=src_key_padding_mask)

        return memory


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(self, vocab_size, ninp, nhead, nhid, nlayers, dropout=0.1, embedding_layer=None, max_len=512):
        super(Decoder, self).__init__()

        self.ninp = ninp
        self.trg_word_emb = embedding_layer

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(ninp, eps=1e-6)

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_attn_mask=None):
        tgt = self.trg_word_emb(tgt, tgt_attn_mask)[0]

        dec_output = self.layer_norm(tgt)

        dec_output = dec_output.transpose(0, 1)

        output = self.transformer_decoder(tgt=dec_output, memory=memory, tgt_mask=tgt_mask,
                                          memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)

        output = output.transpose(0, 1)

        return output