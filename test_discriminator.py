import math

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, src_vocab_size, pad_id_src, trg_vocab_size, pad_id_trg, max_len_src, max_len_tgt, d_model=512, use_gpu=False):
        super(Discriminator, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.linear_len_src = int(math.floor(max_len_src/4))
        self.linear_len_tgt = int(math.floor((max_len_tgt-1)/4))
        self.use_gpu = use_gpu
        self.pad_id_trg = pad_id_trg

        self.embed_src_tokens = Embedding(self.src_vocab_size, d_model, pad_id_src)
        self.embed_trg_tokens = Embedding(self.trg_vocab_size, d_model, pad_id_trg)

        self.conv1 = nn.Sequential(
            Conv2d(in_channels=1024,
                   out_channels=512,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            Conv2d(in_channels=512,
                   out_channels=256,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear(256 * self.linear_len_src * self.linear_len_tgt, self.linear_len_src * self.linear_len_tgt),
            nn.ReLU(),
            nn.Dropout(),
            Linear(self.linear_len_src * self.linear_len_tgt, self.linear_len_src * self.linear_len_tgt),
            nn.ReLU(),
            Linear(self.linear_len_src * self.linear_len_tgt, 1),
        )

        self._reset_parameters()

    def forward(self, src_sentence, trg_sentence):
        batch_size = src_sentence.size(0)

        src_out = self.embed_src_tokens(src_sentence)
        trg_out = self.embed_trg_tokens(trg_sentence)

        src_out = torch.stack([src_out] * trg_out.size(1), dim=2)
        trg_out = torch.stack([trg_out] * src_out.size(1), dim=1)

        out = torch.cat([src_out, trg_out], dim=3)

        out = out.permute(0, 3, 1, 2)

        out = self.conv1(out)
        out = self.conv2(out)

        out = out.permute(0, 2, 3, 1)

        out = out.contiguous().view(batch_size, -1)

        out = torch.sigmoid(self.classifier(out))

        return out

    def _reset_parameters(self):
        # fix discriminator word embedding (as Wu et al. do)
        for p in self.embed_src_tokens.parameters():
            p.requires_grad = False
        for p in self.embed_trg_tokens.parameters():
            p.requires_grad = False


def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m


def Linear(in_features, out_features, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.kaiming_uniform_(m.weight.data)
    if bias:
        nn.init.constant_(m.bias.data, 0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m
