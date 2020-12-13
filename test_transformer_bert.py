import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class BertEmbedding(nn.Module):
    def __init__(self, vocab, bert_model, bert_tokenizer, blank_word):
        super(BertEmbedding, self).__init__()
        self.bert_model = bert_model
        self.bert_model.eval()

        self.tokenizer = bert_tokenizer
        self.vocab = vocab
        self.blank_word = blank_word

    def forward(self, x):
        x_converted = []
        attention_mask = []

        for elem in x:
            sequence = []
            attention = []
            for tok_id in elem:
                tok = self.vocab.itos[tok_id.item()]
                if tok_id == self.vocab.stoi[self.blank_word]:
                    attention.append(0)
                else:
                    attention.append(1)
                sequence.append(tok)
            tokenized_sequence = self.tokenizer.convert_tokens_to_ids(sequence)

            x_converted.append(tokenized_sequence)
            attention_mask.append(attention)

        x_converted = torch.Tensor(x_converted).long()
        attention_mask = torch.Tensor(attention_mask).long()
        sequence_output, _, _ = self.bert_model(x_converted, token_type_ids=None, attention_mask=attention_mask)
        return sequence_output


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


class BertEncoderTransformer(nn.Module):
    def __init__(self, src_vocab, bert_model, bert_tokenizer, blank_word,
                 d_model=768, nhead=8, num_decoder_layers=6, dim_feedforward=3072,
                 dropout=0.1, activation="relu", target_vocab_length=60000):
        super(BertEncoderTransformer, self).__init__()

        self.source_embedding = BertEmbedding(src_vocab, bert_model, bert_tokenizer, blank_word)

        self.target_embedding = nn.Embedding(target_vocab_length, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.out = nn.Linear(d_model, target_vocab_length)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        src = src.transpose(0, 1)
        memory = self.source_embedding(src)

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

        #self.target_embedding.weight = bert_model.embeddings.word_embeddings.weight
        self.target_embedding.weight.requires_grad = False
        self.out.weight.requires_grad = False
        self.out.bias.requires_grad = False