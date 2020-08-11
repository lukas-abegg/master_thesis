import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

from zzz_playground.zzz_code.optim import CosineWithRestarts
from layers.transformer_with_gru import Transformer

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


# ### BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # Model parameter
# MAX_SEQ_LEN = 128
# PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
# UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
#
# text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
#                    fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)


class Newsela(TranslationDataset):
    name = 'newsela'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='test', validation='test', test='test', **kwargs):
        """Create dataset objects for splits of the Newsela dataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(Newsela, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


train_data, valid_data, test_data = Newsela.splits(exts=('.src', '.dst'),
                                                   fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

ntokens = len(SRC.vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention zzz_bert_models
dropout = 0.2  # the dropout value

model = Transformer(ntokens, emsize, nhid, nhead, nlayers, use_gate=False, dropout=dropout, max_len=5000).to(device)

parser = argparse.ArgumentParser()
opt = parser.parse_args()

SRC_PAD_IDX = SRC.vocab.stoi['<pad>']
TRG_PAD_IDX = TRG.vocab.stoi['<pad>']

opt.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
opt.criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

opt.SGDR = False

if opt.SGDR:
    opt.sched = CosineWithRestarts(opt.optimizer, T_max=10)

opt.checkpoint = 0

if opt.checkpoint > 0:
    print("model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          opt,
          clip: float):

    print("training model...")

    model.train()

    epoch_loss = 0

    if opt.checkpoint > 0:
        torch.save(model.state_dict(), 'weights/model_weights')

    for i, batch in enumerate(iterator):

        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)

        # the Simple sentence we input has all words except
        # the last, as it is using each word to predict the next
        trg_input = trg[:, :-1]

        preds = model(src, trg_input)

        ys = trg[:, 1:].contiguous().view(-1)

        optimizer.zero_grad()

        loss = criterion(preds.view(-1, preds.size(-1)), ys, ignore_index=TRG_PAD_IDX)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if opt.SGDR:
            opt.sched.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, opt.optimizer, opt.criterion, opt, CLIP)
    valid_loss = evaluate(model, valid_iterator, opt.criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iterator, opt.criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
