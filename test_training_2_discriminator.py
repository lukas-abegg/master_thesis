import math
import os
from collections import OrderedDict

import torch
from torch.optim import Adam

from load_datasets import get_iterator, load_dataset_data
from meters import AverageMeter
from test_discriminator import Discriminator
from test_training_2_discriminator_dataloader import prepare_training_data, DatasetProcessing, train_dataloader, \
    eval_dataloader
from test_transformer import Transformer


def train(train_iter, val_iter, generator, discriminator, num_epochs, target_vocab, use_gpu=False):
    if use_gpu:
        generator.cuda()
        discriminator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    # check checkpoints saving path
    if not os.path.exists('checkpoints/discriminator'):
        os.makedirs('checkpoints/discriminator')

    checkpoints_path = 'checkpoints/discriminator/'

    logging_meters = OrderedDict()
    logging_meters['train_loss'] = AverageMeter()
    logging_meters['train_acc'] = AverageMeter()
    logging_meters['valid_loss'] = AverageMeter()
    logging_meters['valid_acc'] = AverageMeter()

    criterion = torch.nn.BCELoss()

    optimizer = Adam(filter(lambda x: x.requires_grad, discriminator.parameters()), lr=1e-3)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5)

    min_lr = 1e-6

    # Train until the accuracy achieve the define value
    max_epoch = num_epochs
    epoch_i = 1
    trg_acc = 0.82
    best_valid_acc = 0
    best_dev_loss = math.inf
    lr = optimizer.param_groups[0]['lr']

    # validation set data loader (only prepare once)
    train = prepare_training_data(train_iter, generator, target_vocab, BOS_WORD, MAX_LEN, EOS_WORD, BLANK_WORD, use_gpu)
    valid = prepare_training_data(val_iter, generator, target_vocab, BOS_WORD, MAX_LEN, EOS_WORD, BLANK_WORD, use_gpu)
    data_train = DatasetProcessing(data=train, max_len=MAX_LEN)
    data_valid = DatasetProcessing(data=valid, max_len=MAX_LEN)

    # main training loop
    while lr > min_lr and epoch_i <= max_epoch:
        print("At {0}-th epoch ---------------".format(epoch_i))

        if epoch_i > 1:
            train = prepare_training_data(train_iter, generator, target_vocab, BOS_WORD, MAX_LEN, EOS_WORD, BLANK_WORD,
                                          use_gpu)
            data_train = DatasetProcessing(data=train, max_len=MAX_LEN)

        # discriminator training dataloader
        train_loader = train_dataloader(data_train, batch_size=BATCH_SIZE, sort_by_source_size=False)

        valid_loader = eval_dataloader(data_valid, batch_size=BATCH_SIZE)

        # set training mode
        discriminator.train()

        # reset meters
        for key, val in logging_meters.items():
            if val is not None:
                val.reset()

        for i, batch in enumerate(train_loader):

            sources = batch['src_tokens']
            targets = batch['trg_tokens'][:, 1:]

            sources = sources.cuda() if use_gpu else sources
            targets = targets.cuda() if use_gpu else targets

            disc_out = discriminator(sources, targets)

            loss = criterion(disc_out, batch['labels'].unsqueeze(1).float())
            prediction = torch.round(disc_out).int()
            acc = torch.sum(prediction == batch['labels'].unsqueeze(1)).float() / len(batch['labels'])

            logging_meters['train_acc'].update(acc.item())
            logging_meters['train_loss'].update(loss.item())
            print("D training loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
                logging_meters['train_loss'].avg,
                acc,
                logging_meters['train_acc'].avg,
                optimizer.param_groups[0]['lr'], i
            ))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer.step()

            # del sources, targets, loss, disc_out, labels, prediction, acc
            del sources, targets, disc_out, loss, prediction, acc

        # set validation mode
        discriminator.eval()

        with torch.no_grad():

            for i, batch in enumerate(valid_loader):
                sources = batch['src_tokens']
                targets = batch['trg_tokens'][:, 1:]

                sources = sources.cuda() if use_gpu else sources
                targets = targets.cuda() if use_gpu else targets

                disc_out = discriminator(sources, targets)

                loss = criterion(disc_out, batch['labels'].unsqueeze(1).float())
                prediction = torch.round(disc_out).int()
                acc = torch.sum(prediction == batch['labels'].unsqueeze(1)).float() / len(batch['labels'])

                logging_meters['valid_acc'].update(acc.item())
                logging_meters['valid_loss'].update(loss.item())
                print("D eval loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
                    logging_meters['valid_loss'].avg,
                    acc,
                    logging_meters['valid_acc'].avg,
                    optimizer.param_groups[0]['lr'], i
                ))

            del disc_out, loss, prediction, acc

        lr_scheduler.step(logging_meters['valid_loss'].avg)

        if best_valid_acc < logging_meters['valid_acc'].avg:
            best_valid_acc = logging_meters['valid_acc'].avg

            model_path = checkpoints_path + "ce_{0:.3f}_acc_{1:.3f}.epoch_{2}.pt".format(
                logging_meters['valid_loss'].avg, logging_meters['valid_acc'].avg, epoch_i)

            print("Save model to", model_path)
            torch.save(discriminator.state_dict(), model_path)

        if logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = logging_meters['valid_loss'].avg

            model_path = checkpoints_path + "best_dmodel.pt"

            print("Save model to", model_path)
            torch.save(discriminator.state_dict(), model_path)

        # pretrain the discriminator to achieve accuracy 82%
        if logging_meters['valid_acc'].avg >= trg_acc:
            return

        epoch_i += 1


### Load Data
###-----------------------------
# Special Tokens
BOS_WORD = '[CLS]'
EOS_WORD = '[SEP]'
BLANK_WORD = '[PAD]'

MAX_LEN = 40

dataset = "newsela"
# dataset = "mws"

train_data, valid_data, test_data, SRC, TGT = load_dataset_data(MAX_LEN, dataset, BOS_WORD, EOS_WORD, BLANK_WORD)

BATCH_SIZE = 30
# Create iterators to process text in batches of approx. the same length
train_iter = get_iterator(train_data, BATCH_SIZE)
val_iter = get_iterator(valid_data, BATCH_SIZE)

### Load Generator
###-----------------------------
source_vocab_length = len(SRC.vocab)
target_vocab_length = len(TGT.vocab)

generator = Transformer(source_vocab_length=source_vocab_length, target_vocab_length=target_vocab_length)
generator_path = "checkpoint_best_epoch.pt"
generator.load_state_dict(torch.load(generator_path))

### Load Discriminator
###-----------------------------
discriminator = Discriminator(src_vocab_size=source_vocab_length, pad_id_src=SRC.vocab.stoi[BLANK_WORD],
                              trg_vocab_size=target_vocab_length, pad_id_trg=TGT.vocab.stoi[BLANK_WORD],
                              max_len=MAX_LEN, use_gpu=False)

### Start Training
###-----------------------------
NUM_EPOCHS = 25
use_cuda = (torch.cuda.device_count() >= 1)
train(train_iter, val_iter, generator, discriminator, NUM_EPOCHS, TGT.vocab, use_cuda)

print("Training finished")
