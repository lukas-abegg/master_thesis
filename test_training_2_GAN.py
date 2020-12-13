import math
import os
from collections import OrderedDict
from random import random

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F

from PGLoss import PGLoss
from load_datasets import get_iterator, load_dataset_data, bert_tokenizer
from meters import AverageMeter
from test_discriminator import Discriminator
from test_training_2_discriminator_dataloader import greedy_decode_sentence
from test_transformer import Transformer

import spacy

spacy_en = spacy.load('en')


def train(train_iter, val_iter, generator, discriminator, max_epochs, target_vocab, use_gpu=False):
    if use_gpu:
        generator.cuda()
        discriminator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    g_logging_meters = OrderedDict()
    g_logging_meters['train_loss'] = AverageMeter()
    g_logging_meters['valid_loss'] = AverageMeter()
    g_logging_meters['train_acc'] = AverageMeter()
    g_logging_meters['valid_acc'] = AverageMeter()
    g_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    d_logging_meters = OrderedDict()
    d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['train_acc'] = AverageMeter()
    d_logging_meters['valid_acc'] = AverageMeter()
    d_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    # adversarial training checkpoints saving path
    if not os.path.exists('checkpoints/joint'):
        os.makedirs('checkpoints/joint')
    checkpoints_save_path = 'checkpoints/joint/'

    # define loss function
    g_criterion = torch.nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi[BLANK_WORD], reduction='sum')
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss(ignore_index=TGT.vocab.stoi[BLANK_WORD], size_average=True, reduce=True)

    # fix discriminator word embedding (as Wu et al. do)
    for p in discriminator.embed_src_tokens.parameters():
        p.requires_grad = False
    for p in discriminator.embed_trg_tokens.parameters():
        p.requires_grad = False

    # define optimizer
    g_lr = 1e-4
    g_optimizer = Adam(filter(lambda x: x.requires_grad, generator.parameters()), lr=g_lr)
    d_lr = 1e-4
    d_optimizer = Adam(filter(lambda x: x.requires_grad, discriminator.parameters()), lr=d_lr)

    # Start Joint Training
    print("-------------- Start Joint Training --------------")

    best_dev_loss = math.inf
    num_update = 0

    # main training loop
    for epoch_i in range(1, max_epochs + 1):
        print("At {0}-th epoch.".format(epoch_i))

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()

        # set training mode
        generator.train()
        discriminator.train()
        update_learning_rate(num_update, 8e4, g_lr, 0.5, g_optimizer)

        for i, batch in enumerate(train_iter):

            ## part I: use gradient policy method to train the generator

            # use policy gradient training when random.random() > 50%
            if random.random() >= 0.5:

                print("Policy Gradient Training")

                # a tensor with max possible translation length
                src = batch.src.cuda() if use_gpu else batch.src
                trg = batch.trg.cuda() if use_gpu else batch.trg

                # change to shape (bs , max_seq_len)
                src = src.transpose(0, 1)
                # change to shape (bs , max_seq_len+1) , Since right shifted
                trg = trg.transpose(0, 1)

                trg_input = trg[:, :-1]
                targets = trg[:, 1:]

                size = trg_input.size(1)

                np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
                np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
                np_mask = np_mask.cuda() if use_gpu else np_mask

                # Forward, backprop, optimizer
                sys_out_batch = generator(src.transpose(0, 1), trg_input.transpose(0, 1), tgt_mask=np_mask)
                sys_out_batch = F.log_softmax(sys_out_batch.transpose(0, 1), dim=-1)
                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))

                _, predictions = out_batch.topk(1)
                predictions = predictions.squeeze(1)
                predictions = torch.reshape(predictions, trg_input.shape)

                with torch.no_grad():
                    reward = discriminator(src, predictions)

                tokenized_origins = [convert_ids_to_tokens(origin, SRC) for origin in src]
                tokenized_predictions = [convert_ids_to_tokens(pred, TGT) for pred in predictions]
                tokenized_targets = [convert_ids_to_tokens(target, TGT) for target in targets]

                pg_loss = pg_criterion(sys_out_batch, targets, tokenized_origins, tokenized_predictions,
                                       tokenized_targets, reward, use_gpu)

                sample_size = targets.size(0)  # batch-size
                logging_loss = pg_loss

                g_logging_meters['train_loss'].update(logging_loss.item(), sample_size)
                print(
                    "G policy gradient loss at batch {i}: {pg_loss.item():.3f}, lr={g_optimizer.param_groups[0]['lr']}")

                g_optimizer.zero_grad()
                pg_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                g_optimizer.step()

            else:
                # MLE training
                print("MLE Training")

                # a tensor with max possible translation length
                src = batch.src.cuda() if use_gpu else batch.src
                trg = batch.trg.cuda() if use_gpu else batch.trg

                # change to shape (bs , max_seq_len)
                src = src.transpose(0, 1)
                # change to shape (bs , max_seq_len+1) , Since right shifted
                trg = trg.transpose(0, 1)

                trg_input = trg[:, :-1]
                targets = trg[:, 1:].contiguous().view(-1)

                size = trg_input.size(1)

                np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
                np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
                np_mask = np_mask.cuda() if use_gpu else np_mask

                # Forward, backprop, optimizer
                preds = generator(src.transpose(0, 1), trg_input.transpose(0, 1), tgt_mask=np_mask)
                preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))
                loss = g_criterion(preds, targets)

                sample_size = targets.size(0)
                logging_loss = loss / sample_size

                g_logging_meters['bsz'].update(sample_size)
                g_logging_meters['train_loss'].update(logging_loss, sample_size)
                print(
                    "G MLE loss at batch {i}: {g_logging_meters['train_loss'].avg:.3f}, lr={g_optimizer.param_groups[0]['lr']}")

                g_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                g_optimizer.step()

            num_update += 1

            # part II: train the discriminator
            print("Discriminator Training")

            # a tensor with max possible translation length
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg

            # change to shape (bs , max_seq_len)
            src = src.transpose(0, 1)

            # change to shape (bs , max_seq_len)
            trg = trg.transpose(0, 1)

            bsz = trg.size(0)

            neg_tokens = []
            for origin_sentence in src:
                neg_tokens.append(
                    greedy_decode_sentence(generator, origin_sentence, target_vocab, MAX_LEN, BOS_WORD, EOS_WORD,
                                           BLANK_WORD, use_gpu))

            fake_sentences = torch.stack(neg_tokens)
            fake_labels = Variable(torch.zeros(bsz).float())

            true_sentences = trg
            true_labels = Variable(torch.ones(bsz).float())

            src_sentences = torch.cat([src, src], dim=0)
            trg_sentences = torch.cat([true_sentences, fake_sentences], dim=0)
            labels = torch.cat([true_labels, fake_labels], dim=0)

            indices = np.random.permutation(2 * bsz)
            trg_sentences = trg_sentences[indices][:bsz]
            labels = labels[indices][:bsz]

            if use_gpu:
                src_sentences = src_sentences.cuda()
                trg_sentences = trg_sentences.cuda()
                labels = labels.cuda()

            disc_out = discriminator(src_sentences, trg_sentences)
            d_loss = d_criterion(disc_out, labels)
            acc = torch.sum(torch.round(disc_out).squeeze(1) == labels).float() / len(labels)

            d_logging_meters['train_acc'].update(acc)
            d_logging_meters['train_loss'].update(d_loss)
            print(
                "D training loss {d_logging_meters['train_loss'].avg:.3f}, acc {d_logging_meters['train_acc'].avg:.3f} at batch {i}")

            d_optimizer.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            d_optimizer.step()

        # Validation
        print("-------------- Start Validation --------------")

        # set validation mode
        generator.eval()
        discriminator.eval()

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()

        with torch.no_grad():
            for i, batch in enumerate(val_iter):

                print("Generator Validation")

                # a tensor with max possible translation length
                src = batch.src.cuda() if use_gpu else batch.src
                trg = batch.trg.cuda() if use_gpu else batch.trg

                # change to shape (bs , max_seq_len)
                src = src.transpose(0, 1)
                # change to shape (bs , max_seq_len+1) , Since right shifted
                trg = trg.transpose(0, 1)

                trg_input = trg[:, :-1]
                targets = trg[:, 1:].contiguous().view(-1)

                size = trg_input.size(1)

                np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
                np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
                np_mask = np_mask.cuda() if use_gpu else np_mask

                # generator validation
                preds = generator(src.transpose(0, 1), trg_input.transpose(0, 1), tgt_mask=np_mask)
                preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))
                loss = g_criterion(preds, targets)

                sample_size = targets.size(0)
                logging_loss = loss / sample_size

                g_logging_meters['valid_loss'].update(logging_loss, sample_size)
                print("G dev loss at batch {0}: {1:.3f}".format(i, g_logging_meters['valid_loss'].avg))

                print("Discriminator Validation")

                # discriminator validation
                # a tensor with max possible translation length
                src = batch.src.cuda() if use_gpu else batch.src
                trg = batch.trg.cuda() if use_gpu else batch.trg

                # change to shape (bs , max_seq_len)
                src = src.transpose(0, 1)

                # change to shape (bs , max_seq_len)
                trg = trg.transpose(0, 1)

                bsz = trg.size(0)

                neg_tokens = []
                for origin_sentence in src:
                    neg_tokens.append(
                        greedy_decode_sentence(generator, origin_sentence, target_vocab, MAX_LEN, BOS_WORD, EOS_WORD,
                                               BLANK_WORD, use_gpu))

                fake_sentences = torch.stack(neg_tokens)
                fake_labels = Variable(torch.zeros(bsz).float())

                true_sentences = trg
                true_labels = Variable(torch.ones(bsz).float())

                src_sentences = torch.cat([src, src], dim=0)
                trg_sentences = torch.cat([true_sentences, fake_sentences], dim=0)
                labels = torch.cat([true_labels, fake_labels], dim=0)

                indices = np.random.permutation(2 * bsz)
                trg_sentences = trg_sentences[indices][:bsz]
                labels = labels[indices][:bsz]

                if use_gpu:
                    src_sentences = src_sentences.cuda()
                    trg_sentences = trg_sentences.cuda()
                    labels = labels.cuda()

                disc_out = discriminator(src_sentences, trg_sentences)
                d_loss = d_criterion(disc_out, labels)
                acc = torch.sum(torch.round(disc_out).squeeze(1) == labels).float() / len(labels)

                d_logging_meters['valid_acc'].update(acc)
                d_logging_meters['valid_loss'].update(d_loss)
                print(
                    "D dev loss {d_logging_meters['valid_loss'].avg:.3f}, acc {d_logging_meters['valid_acc'].avg:.3f} at batch {i}")

            generator_model_path_to_save = checkpoints_save_path + "joint_{0:.3f}.epoch_{1}.pt".format(
                g_logging_meters['valid_loss'].avg, epoch_i)

            print("Save model to", generator_model_path_to_save)
            torch.save(generator.state_dict(), generator_model_path_to_save)

            if g_logging_meters['valid_loss'].avg < best_dev_loss:
                best_dev_loss = g_logging_meters['valid_loss'].avg

                generator_model_path_to_save = checkpoints_save_path + "best_generator_g_model.pt"
                torch.save(generator.state_dict(), generator_model_path_to_save)


def tokenize_en(text):
    text = text.replace("<unk>", "UNK")
    return [tok.text for tok in spacy_en.tokenizer(text)]


def convert_ids_to_tokens(tensor, vocab_field):
    sentence = []

    for elem in tensor:
        token = vocab_field.vocab.itos[elem.item()]

        if token != vocab_field.pad_token and token != vocab_field.eos_token:
            sentence.append(token)

    translated_sentence = tokenize_en(bert_tokenizer.convert_tokens_to_string(sentence))
    return translated_sentence


def update_learning_rate(update_times, target_times, init_lr, lr_shrink, optimizer):
    lr = init_lr * (lr_shrink ** (update_times // target_times))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

checkpoints_path = 'checkpoints/'

### Load Generator
###-----------------------------
source_vocab_length = len(SRC.vocab)
target_vocab_length = len(TGT.vocab)

generator = Transformer(None, source_vocab_length=source_vocab_length, target_vocab_length=target_vocab_length)
generator_path = checkpoints_path + '../' + 'checkpoint_best_epoch.pt'
generator.load_state_dict(torch.load(generator_path))
print("Generator has successfully loaded!")

### Load Discriminator
###-----------------------------
discriminator = Discriminator(src_vocab_size=source_vocab_length, pad_id_src=SRC.vocab.stoi[BLANK_WORD],
                              trg_vocab_size=target_vocab_length, pad_id_trg=TGT.vocab.stoi[BLANK_WORD],
                              max_len=MAX_LEN, use_gpu=False)
discriminator_path = checkpoints_path + 'discriminator/' + 'best_dmodel.pt'
discriminator.load_state_dict(torch.load(discriminator_path))
print("Discriminator has successfully loaded!")

### Start Training
###-----------------------------
NUM_EPOCHS = 25
use_cuda = (torch.cuda.device_count() >= 1)
train(train_iter, val_iter, generator, discriminator, NUM_EPOCHS, TGT.vocab, use_cuda)

print("Training finished")
