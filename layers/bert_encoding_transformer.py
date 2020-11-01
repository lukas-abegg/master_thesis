import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_


class TransformerModel(nn.Module):

    def __init__(self, src_vocab_length, trg_vocab_length, ninp, nhead, nhid, nlayers, dropout=0.1,
                 embedding_layer=None, max_len=512, SRC=None, TRG=None, tokenizer=None):
        super(TransformerModel, self).__init__()

        self.source_embedding = Embedding(src_vocab_length, ninp, embedding_layer, SRC, tokenizer)
        self.pos_encoder = PositionalEncoding(ninp)
        encoder_layer = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout, "relu")
        encoder_norm = nn.LayerNorm(ninp)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        self.target_embedding = Embedding(trg_vocab_length, ninp, embedding_layer, TRG, tokenizer)
        decoder_layer = nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout, "relu")
        decoder_norm = nn.LayerNorm(ninp)
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)
        self.out = nn.Linear(ninp, trg_vocab_length)
        self._reset_parameters(embedding_layer)
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

    def _reset_parameters(self, embedding_layer):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        self.source_embedding.bert = embedding_layer
        self.target_embedding.bert = embedding_layer

        for p in self.source_embedding.bert.parameters():
            p.requires_grad = False

        for p in self.target_embedding.bert.parameters():
            p.requires_grad = False


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


class Embedding(nn.Module):
    """ Implement input and output embedding """

    def __init__(self, vocab_size, ninp, bert, VOCAB=None, tokenizer=None):
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.ninp = ninp
        self.bert = bert
        self.bert.eval()
        self.VOCAB = VOCAB
        self.PAD = VOCAB.vocab.stoi["[PAD]"]
        self.tokenizer = tokenizer

    def forward(self, x):
        x_converted = []
        attention_mask = []
        x = x.transpose(0, 1)
        for elem in x:
            sequence = []
            attention = []
            for tok_id in elem:
                tok = self.VOCAB.vocab.itos[tok_id.item()]
                if tok_id == self.PAD:
                    attention.append(0)
                else:
                    attention.append(1)
                sequence.append(tok)
            tokenized_sequence = self.tokenizer.convert_tokens_to_ids(sequence)

            x_converted.append(tokenized_sequence)
            attention_mask.append(attention)
        x_converted = torch.Tensor(x_converted).long().to(self.device)
        attention_mask = torch.Tensor(attention_mask).long().to(self.device)
        sequence_output, _, _ = self.bert(x_converted, token_type_ids=None, attention_mask=attention_mask)
        sequence_output = sequence_output.transpose(0, 1).to(self.device)
        return sequence_output
