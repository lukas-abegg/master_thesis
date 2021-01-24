import math
import os
import re
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from comet_ml import Experiment
from tqdm import tqdm

from comparing_models.seq2seq import test_model
from comparing_models.seq2seq.lstm_bi_model import LSTMModel
from load_datasets import load_dataset_data, get_iterator, bert_tokenizer
from meters import AverageMeter


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

    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi[BLANK_WORD], reduction='sum')

    # define optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=hyper_params["learning_rate"], betas=(0.9, 0.999),
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
            # change to shape (bs , max_seq_len)
            trg = trg.transpose(0, 1)

            targets = trg[1:].contiguous().view(-1)

            # Forward, backprop, optimizer
            preds = model(src, trg)
            preds = preds[1:].contiguous().view(-1, preds.size(-1))
            loss = criterion(preds, targets)

            sample_size = targets.size(1)
            logging_loss = loss / sample_size

            # Accuracy
            preds = preds.transpose(0, 1)
            targets = targets.transpose(0, 1)

            predicts = preds.argmax(dim=1)
            corrects = predicts == targets
            all = targets == targets
            corrects.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
            all.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
            acc = corrects.sum().float() / all.sum()

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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # del src, trg, preds, loss, labels, acc
            del src, trg, targets, preds, loss, acc

        # set validation mode
        model.eval()

        with torch.no_grad():

            desc = '  - (Validation)   '

            for batch in tqdm(val_iter, desc=desc, leave=False):
                step_valid_i = step_valid_i + 1

                if experiment is not None:
                    experiment.set_step(step_valid_i)

                src, src_len = batch.src.cuda() if use_gpu else batch.src
                trg = batch.trg.cuda() if use_gpu else batch.trg

                # change to shape (bs , max_seq_len)
                src = src.transpose(0, 1)
                # change to shape (bs , max_seq_len)
                trg = trg.transpose(0, 1)

                targets = trg[:, 1:].contiguous().view(-1)

                # Forward, backprop, optimizer
                preds = model(src, src_len, trg)
                preds = preds[:, 1:].contiguous().view(-1, preds.size(-1))
                loss = criterion(preds, targets)

                sample_size = targets.size(1)
                logging_loss = loss / sample_size

                # Accuracy
                preds = preds.transpose(0, 1)
                targets = targets.transpose(0, 1)

                predicts = preds.argmax(dim=1)
                corrects = predicts == targets
                all = targets == targets
                corrects.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                all.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                acc = corrects.sum().float() / all.sum()

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

        # Log after each epoch
        print(
            "\nEpoch [{0}/{1}] complete. Train loss: {2:.3f}, avgAcc {3:.3f}. Val Loss: {4:.3f}, avgAcc {5:.3f}. lr={6}."
                .format(epoch, num_epochs, logging_meters['train_loss'].avg, logging_meters['train_acc'].avg,
                        logging_meters['valid_loss'].avg, logging_meters['valid_acc'].avg,
                        optimizer.param_groups[0]['lr']))

        lr_scheduler.step(logging_meters['valid_loss'].avg)

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

            model_path = checkpoints_path + "best_model.pt"

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
            if experiment is not None:
                if step_i == 0:
                    experiment.log_text(str("Train Sample: " + sent))
                else:
                    experiment.log_text(str("Validation Sample: " + sent))


def greedy_decode_sentence(model, sentence, use_gpu=False):
    model.eval()
    sentence = SRC.preprocess(BOS_WORD + " " + sentence + " " + EOS_WORD)

    indexed = []
    for tok in sentence:
        if SRC.vocab.stoi[tok] != SRC.vocab.stoi[BLANK_WORD]:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(SRC.vocab.stoi[BLANK_WORD])

    src_tensor = torch.LongTensor(indexed).unsqueeze(1)

    src_tensor = src_tensor.cuda() if use_gpu else src_tensor

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    trg_indexes = [TGT.vocab.stoi[BOS_WORD]]

    for i in range(max_len_tgt):

        trg_tensor = torch.LongTensor([trg_indexes[-1]])
        trg_tensor = trg_tensor.cuda() if use_gpu else trg_tensor

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == TGT.vocab.stoi[TGT.eos_token]:
            break

    trg_tokens = [TGT.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Found ", torch.cuda.device_count(), " GPU devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device ", device, " for task")

    hyper_params = {
        "dataset": "newsela",  # mws # iwslt #pwkp #wikilarge
        "tokenizer": "wordpiece",  # wordpiece
        "sequence_length_src": 72,
        "sequence_length_tgt": 43,
        "batch_size": 64,
        "num_epochs": 30,
        "learning_rate": 1e-3,
        "num_layer": 2,
        "d_layer": 256,
        "d_embedding": 300,
        "dropout": 0.2,
    }

    checkpoint_base = "./"  # "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer_bert_encoder"
    project_name = "transforner-newsela"
    tracking_active = False
    base_path = "./"  # "/glusterfs/dfs-gfs-dist/abeggluk/data_3"

    max_len_src = hyper_params["sequence_length_src"]
    max_len_tgt = hyper_params["sequence_length_tgt"]

    dataset = hyper_params["dataset"]
    tokenizer = hyper_params["tokenizer"]

    experiment = None
    if tracking_active:
        # Add the following code anywhere in your machine learning file
        experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                                project_name=project_name,
                                workspace="abeggluk")
        experiment.log_parameters(hyper_params)

    ### Load Data
    # Special Tokens
    if tokenizer == "wordpiece":
        BOS_WORD = '[CLS]'
        EOS_WORD = '[SEP]'
        BLANK_WORD = '[PAD]'
    else:
        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset,
                                                                    tokenizer,
                                                                    BOS_WORD, EOS_WORD, BLANK_WORD)

    BATCH_SIZE = hyper_params["batch_size"]
    # Create iterators to process text in batches of approx. the same length
    train_iter = get_iterator(train_data, BATCH_SIZE)
    val_iter = get_iterator(valid_data, BATCH_SIZE)

    ### Load Generator
    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)

    EMB_DIM = hyper_params["d_embedding"]
    HID_DIM = hyper_params["d_layer"]
    NUM_LAYERS = hyper_params["num_layer"]
    DROPOUT = hyper_params["dropout"]
    src_pad_idx = SRC.vocab.stoi[BLANK_WORD]
    tgt_pad_idx = TGT.vocab.stoi[BLANK_WORD]

    model = LSTMModel(source_vocab_length, target_vocab_length, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT, src_pad_idx,
                      tgt_pad_idx, use_cuda=use_cuda)

    # Set model parameters
    args = {
        "encoder_embed_dim": 1000,
        "encoder_layers": 2,  # 4,
        "encoder_dropout_in": 0,
        "encoder_dropout_out": 0,
        "decoder_embed_dim": 1000,
        "decoder_layers": 2,  # 4
        "decoder_out_embed_dim": 1000,
        "decoder_dropout_in": 0,
        "decoder_dropout_out": 0,
    }

    #model = test_model.LSTMModel(args, source_vocab_length, target_vocab_length, use_cuda=use_cuda)

    ### Start Training
    NUM_EPOCHS = hyper_params["num_epochs"]

    if experiment is not None:
        with experiment.train():
            train(train_iter, val_iter, model, NUM_EPOCHS, checkpoint_base, use_cuda, experiment)
    else:
        train(train_iter, val_iter, model, NUM_EPOCHS, checkpoint_base, use_cuda, experiment)

    print("Training finished")
    sys.exit()