import os
import re
import math
import sys
from collections import OrderedDict
from tqdm import tqdm

from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from load_datasets import load_dataset_data, get_iterator, bert_tokenizer
from meters import AverageMeter
from test_transformer_bert import BertEncoderTransformer
from transformers import BertModel, BertConfig


def train(train_iter, val_iter, model, num_epochs, checkpoint_base, use_gpu=True, experiment=None):
    if use_gpu:
        model.cuda()
    else:
        model.cpu()

    # check checkpoints saving path
    checkpoints_path = os.path.join(checkpoint_base, 'checkpoints/mle/')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    logging_meters = OrderedDict()
    logging_meters['train_loss'] = AverageMeter()
    logging_meters['train_acc'] = AverageMeter()
    logging_meters['valid_loss'] = AverageMeter()
    logging_meters['valid_acc'] = AverageMeter()

    criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi[BLANK_WORD], reduction='sum')

    # optimizer = NoamOptimizer(filter(lambda p: p.requires_grad, model.parameters()), d_model=512, warmup_steps=2500)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=hyper_params["learning_rate"], betas=(0.9, 0.98),
                                 eps=1e-9)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5)

    # Train until the accuracy achieve the define value
    best_avg_valid_loss = math.inf
    best_avg_valid_acc = 0

    step_train_i = 0
    step_valid_i = 0
    for epoch in range(1, num_epochs + 1):

        if experiment is not None:
            experiment.set_epoch(epoch)

        # Train model
        model.train()

        desc = '  - (Training)   '
        for batch in tqdm(train_iter, desc=desc, leave=False):
            step_train_i = step_train_i + 1

            if experiment is not None:
                experiment.set_step(step_train_i)

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
            preds = model(src.transpose(0, 1), trg_input.transpose(0, 1),
                          tgt_mask=np_mask)  # , src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
            preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))
            loss = criterion(preds, targets)

            sample_size = targets.size(0)
            logging_loss = loss / sample_size

            # Accuracy
            predicts = preds.argmax(dim=1)
            corrects = predicts == targets
            corrects.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
            acc = corrects.sum()

            logging_meters['train_acc'].update(acc.item())
            logging_meters['train_loss'].update(logging_loss.item())
            print("\nTraining loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
                logging_meters['train_loss'].avg,
                acc,
                logging_meters['train_acc'].avg,
                optimizer.param_groups[0]['lr'], step_train_i
            ))

            if experiment is not None:
                experiment.log_metric("batch_train_loss", logging_meters['train_loss'].avg)
                experiment.log_metric("batch_train_acc", logging_meters['train_acc'].avg)
                experiment.log_metric("batch_train_acc", acc)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # del src, trg, preds, loss, labels, acc
            del src, trg, preds, loss, acc

        # set validation mode
        model.eval()

        with torch.no_grad():

            desc = '  - (Validation)   '

            for batch in tqdm(val_iter, desc=desc, leave=False):
                step_valid_i = step_valid_i + 1

                if experiment is not None:
                    experiment.set_step(step_valid_i)

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
                np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1,
                                                                                               float(0.0))
                np_mask = np_mask.cuda() if use_gpu else np_mask

                preds = model(src.transpose(0, 1), trg_input.transpose(0, 1),
                              tgt_mask=np_mask)  # , src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
                preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))
                loss = criterion(preds, targets)

                sample_size = targets.size(0)
                logging_loss = loss / sample_size

                # Accuracy
                predicts = preds.argmax(dim=1)
                corrects = predicts == targets
                corrects.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                acc = corrects.sum()

                logging_meters['valid_acc'].update(acc.item())
                logging_meters['valid_loss'].update(logging_loss.item())
                print("\nValidation loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
                    logging_meters['valid_loss'].avg,
                    acc,
                    logging_meters['valid_acc'].avg,
                    optimizer.param_groups[0]['lr'], step_valid_i
                ))

                if experiment is not None:
                    experiment.log_metric("batch_valid_loss", logging_meters['valid_loss'].avg)
                    experiment.log_metric("batch_valid_acc", logging_meters['valid_acc'].avg)
                    experiment.log_metric("batch_valid_acc", acc)

        lr_scheduler.step(logging_meters['valid_loss'].avg)

        # Log after each epoch
        print("\nEpoch [{0}/{1}] complete. Train Loss: {2:.3f}. Val Loss: {3:.3f}"
              .format(epoch, num_epochs, logging_meters['train_loss'].avg, logging_meters['valid_loss'].avg))

        if experiment is not None:
            experiment.log_metric("epoch_train_loss", logging_meters['train_loss'].avg)
            experiment.log_metric("epoch_train_acc", logging_meters['train_acc'].avg)

            experiment.log_metric("epoch_valid_loss", logging_meters['valid_loss'].avg)
            experiment.log_metric("epoch_valid_acc", logging_meters['valid_acc'].avg)

            experiment.log_metric("current_lr", optimizer.param_groups[0]['lr'])

            experiment.log_epoch_end(epoch_cnt=epoch)

        # Save best model till now:
        if best_avg_valid_acc < logging_meters['valid_acc'].avg:
            best_avg_valid_acc = logging_meters['valid_acc'].avg

            model_path = checkpoints_path + "ce_{0:.3f}_acc_{1:.3f}.epoch_{2}.pt".format(
                logging_meters['valid_loss'].avg, logging_meters['valid_acc'].avg, epoch)

            print("Save model to", model_path)
            torch.save(model.state_dict(), model_path)

            if experiment is not None:
                experiment.log_other("best_model_valid_acc", model_path)

        if logging_meters['valid_loss'].avg < best_avg_valid_loss:
            best_avg_valid_loss = logging_meters['valid_loss'].avg

            model_path = checkpoints_path + "best_dmodel.pt"

            print("Save model to", model_path)
            torch.save(model.state_dict(), model_path)

            if experiment is not None:
                experiment.log_other("best_model_valid_loss", model_path)

        # Check Example after each epoch:
        if dataset == "newsela":
            sentences = ["Mundell was on a recent flight to Orlando .",
                         "Normally , they would see just one or two sea lions in the entire month ."]
            exp_sentences = ["A Fake Flight Vest",
                             "Usually , they see just one or two ."]
        else:
            sentences = [
                "Diverticulitis is a common digestive disease which involves the formation of pouches diverticula within the bowel wall .",
                '"In 1998 , swine flu was found in pigs in four U .S . states ."']
            exp_sentences = ["Diverticulitis is a disease of the digestive system .",
                             "Swine flu is common in pigs ."]

        print("\nTranslate Samples:")
        for step_i in range(0, len(sentences)):
            if step_i == 0:
                print("Train Sample:")
            else:
                print("Validation Sample:")
            print("Original Sentence: {}".format(sentences[step_i]))
            greedy_sent = greedy_decode_sentence(model, sentences[step_i], use_gpu)
            sent = re.split(r'\s+', greedy_sent)
            sent = bert_tokenizer.convert_tokens_to_string(sent)
            print("Translated Sentence: {}".format(sent))
            print("Expected Sentence: {}".format(exp_sentences[step_i]))
            print("---------------------------------------")


def greedy_decode_sentence(model, sentence, use_gpu=False):
    model.eval()
    sentence = SRC.preprocess(BOS_WORD + " " + sentence + " " + EOS_WORD)
    indexed = []
    for tok in sentence:
        if SRC.vocab.stoi[tok] != SRC.vocab.stoi[BLANK_WORD]:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(SRC.vocab.stoi[BLANK_WORD])

    sentence = Variable(torch.LongTensor([indexed]))
    trg_init_tok = TGT.vocab.stoi[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]])
    translated_sentence = ""

    if use_gpu:
        sentence = sentence.cuda()
        trg = trg.cuda()

    for i in range(max_len_tgt):

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

    translated_sentence = re.split(r"\s+", translated_sentence)
    translated_sentence = bert_tokenizer.convert_tokens_to_string(translated_sentence)

    return translated_sentence


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Found ", torch.cuda.device_count(), " GPU devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device ", device, " for task")

    hyper_params = {
        "dataset": "newsela",  # mws
        "sequence_length_src": 72,
        "sequence_length_tgt": 43,
        "batch_size": 25,
        "num_epochs": 15,
        "learning_rate": 1e-4,
        "d_model": 768,
        "n_head": 8,
        "dim_feedforward": 3072,
        "n_layers": 6,
        "dropout": 0.1,
    }

    bert_path = "/glusterfs/dfs-gfs-dist/abeggluk/zzz_bert_models_1/bert_base_cased_12"
    checkpoint_base = "./"
    project_name = "newsela_test_bert"
    tracking_active = True
    base_path = "/glusterfs/dfs-gfs-dist/abeggluk/data_1"

    max_len_src = hyper_params["sequence_length_src"]
    max_len_tgt = hyper_params["sequence_length_tgt"]

    dataset = hyper_params["dataset"]

    experiment = None
    if tracking_active:
        # Add the following code anywhere in your machine learning file
        experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                                project_name=project_name,
                                workspace="abeggluk")
        experiment.log_parameters(hyper_params)

    ### Load Data
    # Special Tokens
    BOS_WORD = '[CLS]'
    EOS_WORD = '[SEP]'
    BLANK_WORD = '[PAD]'

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset, BOS_WORD, EOS_WORD, BLANK_WORD)

    BATCH_SIZE = hyper_params["batch_size"]
    # Create iterators to process text in batches of approx. the same length
    train_iter = get_iterator(train_data, BATCH_SIZE)
    val_iter = get_iterator(valid_data, BATCH_SIZE)

    ### Load Generator
    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)

    bert_config = BertConfig.from_pretrained(bert_path, output_hidden_states=True)
    bert_model = BertModel.from_pretrained(bert_path, config=bert_config)

    model = BertEncoderTransformer(SRC.vocab, bert_model, bert_tokenizer, BLANK_WORD,
                                   d_model=hyper_params["d_model"], nhead=hyper_params["n_head"],
                                   num_decoder_layers=hyper_params["n_layers"],
                                   dim_feedforward=hyper_params["dim_feedforward"], dropout=hyper_params["dropout"],
                                   target_vocab_length=target_vocab_length, max_len_tgt=max_len_tgt)

    ### Start Training
    NUM_EPOCHS = hyper_params["num_epochs"]

    if experiment is not None:
        with experiment.train():
            train(train_iter, val_iter, model, NUM_EPOCHS, checkpoint_base, use_cuda, experiment)
    else:
        train(train_iter, val_iter, model, NUM_EPOCHS, checkpoint_base, use_cuda, experiment)

    print("Training finished")
    sys.exit()
