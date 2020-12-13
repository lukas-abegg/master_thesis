import comet_ml

import tqdm

import copy
from typing import Optional, Any
import numpy as np
import torch
from torch.nn.init import xavier_uniform_
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
# For data loading.
from torchtext import data, datasets
import spacy

# Load the Spacy Models
from test_transformer import Transformer

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


# Add the following code anywhere in your machine learning file
experiment = comet_ml.Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                                 project_name="test_MK30_dataset",
                                 workspace="abeggluk")
experiment.display()

# Special Tokens
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_de, init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)

MAX_LEN = 5
train, val, test = datasets.IWSLT.splits(
    exts=('.en', '.de'), fields=(SRC, TGT),
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)

for i, example in enumerate([(x.src, x.trg) for x in train[0:5]]):
    print("Example_{}:{}".format(i, example))

MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

BATCH_SIZE = 60
# Create iterators to process text in batches of approx. the same length
train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, repeat=False, sort_key=lambda x: len(x.src))
val_iter = data.BucketIterator(val, batch_size=1, repeat=False, sort_key=lambda x: len(x.src))

batch = next(iter(train_iter))
src_matrix = batch.src.T
print(src_matrix, src_matrix.size())

trg_matrix = batch.trg.T
print(trg_matrix, trg_matrix.size())

print(SRC.vocab.itos[1])
print(TGT.vocab.itos[2])
print(TGT.vocab.itos[1])

print(TGT.vocab.stoi['</s>'])


source_vocab_length = len(SRC.vocab)
target_vocab_length = len(TGT.vocab)

model = Transformer(source_vocab_length=source_vocab_length, target_vocab_length=target_vocab_length)
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#model = model.cuda()


def train(train_iter, val_iter, model, optim, num_epochs, use_gpu=True):
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        valid_loss = 0
        # Train model
        model.train()

        print("Training started - ")
        for i, batch in enumerate(train_iter):
            if i > 0:
                break
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg

            # change to shape (bs , max_seq_len)
            src = src.transpose(0, 1)
            # change to shape (bs , max_seq_len+1) , Since right shifted
            trg = trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            src_mask = (src != 0)
            src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))

            trg_mask = (trg_input != 0)
            trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
            trg_mask = trg_mask.cuda() if use_gpu else trg_mask

            size = trg_input.size(1)
            # print(size)
            np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
            np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
            np_mask = np_mask.cuda() if use_gpu else np_mask

            # Forward, backprop, optimizer
            optim.zero_grad()
            preds = model(src.transpose(0, 1), trg_input.transpose(0, 1),
                          tgt_mask=np_mask)  # , src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
            preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))
            loss = F.cross_entropy(preds, targets, ignore_index=0, reduction='sum')
            loss.backward()
            optim.step()

            experiment.log_metric("batch_loss", loss.item())

            train_loss += loss.item() / BATCH_SIZE

        model.eval()
        with torch.no_grad():

            print("Evaluation started - ")
            for i, batch in enumerate(val_iter):
                if i > 0:
                    break
                src = batch.src.cuda() if use_gpu else batch.src
                trg = batch.trg.cuda() if use_gpu else batch.trg

                # change to shape (bs , max_seq_len)
                src = src.transpose(0, 1)
                # change to shape (bs , max_seq_len+1) , Since right shifted
                trg = trg.transpose(0, 1)
                trg_input = trg[:, :-1]
                targets = trg[:, 1:].contiguous().view(-1)

                src_mask = (src != 0)
                src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1,
                                                                                                  float(0.0))
                src_mask = src_mask.cuda() if use_gpu else src_mask

                trg_mask = (trg_input != 0)
                trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1,
                                                                                                  float(0.0))
                trg_mask = trg_mask.cuda() if use_gpu else trg_mask

                size = trg_input.size(1)
                # print(size)
                np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
                np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
                np_mask = np_mask.cuda() if use_gpu else np_mask

                preds = model(src.transpose(0, 1), trg_input.transpose(0, 1),
                              tgt_mask=np_mask)  # , src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
                preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))
                loss = F.cross_entropy(preds, targets, ignore_index=0, reduction='sum')
                valid_loss += loss.item() / 1

                experiment.log_metric("valid_loss", loss.item())

        # Log after each epoch
        print("Epoch [{0}/{1}] complete. Train Loss: {2:.3f}. Val Loss: {3:.3f}".format(epoch + 1, num_epochs,
                                                                                      train_loss / len(train_iter),
                                                                                      valid_loss / len(val_iter)))

        # Save best model till now:
        if valid_loss / len(val_iter) < min(valid_losses, default=1e9):
            print("saving state dict")
            torch.save(model.state_dict(), "checkpoint_best_epoch.pt")

        train_losses.append(train_loss / len(train_iter))
        valid_losses.append(valid_loss / len(val_iter))

        # Check Example after each epoch:
        sentences = ["This is an example to check how our model is performing."]
        for sentence in sentences:
            print("Original Sentence: {}".format(sentence))
            print("Translated Sentence: {}".format(greeedy_decode_sentence(model, sentence, use_gpu)))
    return train_losses, valid_losses


def greeedy_decode_sentence(model, sentence, use_gpu=False):
    model.eval()
    sentence = SRC.preprocess(sentence)
    indexed = []
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(0)
    sentence = Variable(torch.LongTensor([indexed]))
    trg_init_tok = TGT.vocab.stoi[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]])
    translated_sentence = ""

    if use_gpu:
        sentence = sentence.cuda()
        trg = trg.cuda()

    maxlen = 5
    for i in range(maxlen):
        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.cuda() if use_gpu else np_mask

        pred = model(sentence.transpose(0, 1), trg, tgt_mask=np_mask)
        pred = F.softmax(pred, dim=-1)
        add_word = TGT.vocab.itos[pred.argmax(dim=2)[-1]]

        if add_word == EOS_WORD:
            break

        translated_sentence += " " + add_word

        if use_gpu:
            trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
        else:
            trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]])))

    return translated_sentence


train_losses, valid_losses = train(train_iter, val_iter, model, optim, 35, False)


