import math
import os
import sys
from collections import OrderedDict
from random import random
import numpy as np
from tqdm import tqdm

from comet_ml import Experiment

import torch
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
from transformers import BertModel, BertConfig

from PGLoss import PGLoss
from load_datasets import get_iterator, load_dataset_data, bert_tokenizer
from meters import AverageMeter
from test_discriminator import Discriminator
from test_training_2_discriminator_dataloader import greedy_decode_sentence
from test_transformer import Transformer

import spacy

spacy_en = spacy.load('en')


def train(train_iter, val_iter, generator, discriminator, max_epochs, target_vocab, checkpoint_base, use_gpu=False,
          experiment=None):
    if use_gpu:
        generator.cuda()
        discriminator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    # adversarial training checkpoints saving path
    checkpoints_path = os.path.join(checkpoint_base, 'checkpoints/joint/')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    g_logging_meters = OrderedDict()
    g_logging_meters['train_loss_joint'] = AverageMeter()
    g_logging_meters['train_acc_joint'] = AverageMeter()
    g_logging_meters['train_loss_mle'] = AverageMeter()
    g_logging_meters['train_acc_mle'] = AverageMeter()
    g_logging_meters['valid_loss'] = AverageMeter()
    g_logging_meters['valid_acc'] = AverageMeter()

    d_logging_meters = OrderedDict()
    d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['train_acc'] = AverageMeter()
    d_logging_meters['valid_acc'] = AverageMeter()

    # define loss function
    g_criterion = torch.nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi[BLANK_WORD], reduction='sum')
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss(ignore_index=TGT.vocab.stoi[BLANK_WORD], size_average=True, reduce=True)

    # define optimizer
    g_optimizer = Adam(filter(lambda x: x.requires_grad, generator.parameters()), lr=hyper_params["learning_rate_g"],
                       betas=(0.9, 0.98), eps=1e-9)
    d_optimizer = Adam(filter(lambda x: x.requires_grad, discriminator.parameters()),
                       lr=hyper_params["learning_rate_d"],
                       betas=(0.9, 0.98), eps=1e-9)

    # Start Joint Training
    print("-------------- Start Joint Training --------------")

    best_dev_loss = math.inf
    best_valid_acc = 0
    num_update = 0
    step_train_i = 0

    mle_training = 0
    joint_training = 0

    # main training loop
    for epoch_i in range(1, max_epochs + 1):

        if experiment is not None:
            experiment.set_epoch(epoch_i)

        # set training mode
        generator.train()
        discriminator.train()
        update_learning_rate(num_update, 8e4, hyper_params["learning_rate_g"], 0.5, g_optimizer)
        update_learning_rate(num_update, 8e4, hyper_params["learning_rate_d"], 0.5, d_optimizer)

        i = 0
        desc = '  - (Training)   '
        for batch in tqdm(train_iter, desc=desc, leave=False):
            step_train_i = step_train_i + 1
            i = i + 1

            if experiment is not None:
                experiment.set_step(step_train_i)

            ## part I: use gradient policy method to train the generator

            # use policy gradient training when random.random() > 50%
            if random() >= 0.5:

                print("Policy Gradient Training")
                joint_training += 1

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

                # Accuracy
                corrects = predictions == targets
                all = targets == targets
                corrects.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                all.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                acc = corrects.sum().float() / all.sum()

                g_logging_meters['train_acc_joint'].update(acc.item())
                g_logging_meters['train_loss_joint'].update(pg_loss.item(), sample_size)
                # print("G policy gradient loss at batch {i}: {pg_loss.item():.3f}, lr={g_optimizer.param_groups[0]['lr']}")

                if experiment is not None:
                    experiment.log_metric("g_batch_train_acc_joint", g_logging_meters['train_acc_joint'].avg)
                    experiment.log_metric("g_batch_train_loss_joint", g_logging_meters['train_loss_joint'].avg)

                g_optimizer.zero_grad()
                pg_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                g_optimizer.step()

                # del src, trg, preds, loss, ...
                del src, trg, trg_input, targets, np_mask, sys_out_batch, out_batch, predictions, tokenized_origins, \
                    tokenized_predictions, tokenized_targets, pg_loss, acc, corrects, all

            else:
                # MLE training
                print("MLE Training")
                mle_training += 1

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

                # Accuracy
                preds = preds.argmax(dim=1)
                corrects = preds == targets
                all = targets == targets
                corrects.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                all.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                acc = corrects.sum().float() / all.sum()

                g_logging_meters['train_acc_mle'].update(acc.item())
                g_logging_meters['train_loss_mle'].update(logging_loss.item())
                # print("G policy gradient loss at batch {i}: {pg_loss.item():.3f}, lr={g_optimizer.param_groups[0]['lr']}")

                if experiment is not None:
                    experiment.log_metric("g_batch_train_acc_mle", g_logging_meters['train_acc_mle'].avg)
                    experiment.log_metric("g_batch_train_loss_mle", g_logging_meters['train_loss_mle'].avg)

                g_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                g_optimizer.step()

                # del src, trg, preds, loss, ...
                del src, trg, trg_input, targets, np_mask, preds, loss, logging_loss, corrects, all, acc

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
                    greedy_decode_sentence(generator, origin_sentence, target_vocab, max_len_tgt, BOS_WORD, EOS_WORD,
                                           BLANK_WORD, use_gpu))

            fake_sentences = torch.stack(neg_tokens)[:, 1:]
            fake_labels = Variable(torch.zeros(bsz).float())

            true_sentences = trg[:, 1:]
            true_labels = Variable(torch.ones(bsz).float())

            src_sentences = torch.cat([src, src], dim=0)
            trg_sentences = torch.cat([true_sentences, fake_sentences], dim=0)
            labels = torch.cat([true_labels, fake_labels], dim=0)

            indices = np.random.permutation(2 * bsz)
            src_sentences = src_sentences[indices][:bsz]
            trg_sentences = trg_sentences[indices][:bsz]
            labels = labels[indices][:bsz]

            if use_gpu:
                src_sentences = src_sentences.cuda()
                trg_sentences = trg_sentences.cuda()
                labels = labels.cuda()

            disc_out = discriminator(src_sentences, trg_sentences)
            d_loss = d_criterion(disc_out, labels.unsqueeze(1).float())
            prediction = torch.round(disc_out).int()
            acc = torch.sum(prediction == labels.unsqueeze(1)).float() / len(labels)

            d_logging_meters['train_acc'].update(acc.item())
            d_logging_meters['train_loss'].update(d_loss.item())
            # print("D training loss {d_logging_meters['train_loss'].avg:.3f}, acc {d_logging_meters['train_acc'].avg:.3f} at batch {i}")

            if experiment is not None:
                experiment.log_metric("d_batch_train_acc", d_logging_meters['train_acc'].avg)
                experiment.log_metric("d_batch_train_loss", d_logging_meters['train_loss'].avg)

            d_optimizer.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            d_optimizer.step()

            # del src, trg, preds, loss, ...
            del src, trg, neg_tokens, fake_sentences, fake_labels, true_sentences, true_labels, src_sentences, trg_sentences, labels, \
                disc_out, d_loss, acc, prediction

        # set validation mode
        generator.eval()
        discriminator.eval()

        with torch.no_grad():
            desc = '  - (Validation)   '
            for batch in tqdm(val_iter, desc=desc, leave=False):

                # generator validation
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

                # Accuracy
                preds = preds.argmax(dim=1)
                corrects = preds == targets
                all = targets == targets
                corrects.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                all.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                acc = corrects.sum().float() / all.sum()

                g_logging_meters['valid_acc'].update(acc.item())
                g_logging_meters['valid_loss'].update(logging_loss.item(), sample_size)
                # print("G dev loss at batch {0}: {1:.3f}".format(i, g_logging_meters['valid_loss'].avg))

                if experiment is not None:
                    experiment.log_metric("g_batch_valid_acc", g_logging_meters['valid_acc'].avg)
                    experiment.log_metric("g_batch_valid_loss", g_logging_meters['valid_loss'].avg)

                # del src, trg, preds, loss, ...
                del src, trg, trg_input, targets, np_mask, preds, loss, logging_loss, corrects, acc, all

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
                        greedy_decode_sentence(generator, origin_sentence, target_vocab, max_len_tgt, BOS_WORD,
                                               EOS_WORD,
                                               BLANK_WORD, use_gpu))

                fake_sentences = torch.stack(neg_tokens)[:, 1:]
                fake_labels = Variable(torch.zeros(bsz).float())

                true_sentences = trg[:, 1:]
                true_labels = Variable(torch.ones(bsz).float())

                src_sentences = torch.cat([src, src], dim=0)
                trg_sentences = torch.cat([true_sentences, fake_sentences], dim=0)
                labels = torch.cat([true_labels, fake_labels], dim=0)

                indices = np.random.permutation(2 * bsz)
                src_sentences = src_sentences[indices][:bsz]
                trg_sentences = trg_sentences[indices][:bsz]
                labels = labels[indices][:bsz]

                if use_gpu:
                    src_sentences = src_sentences.cuda()
                    trg_sentences = trg_sentences.cuda()
                    labels = labels.cuda()

                disc_out = discriminator(src_sentences, trg_sentences)
                d_loss = d_criterion(disc_out, labels.unsqueeze(1).float())
                prediction = torch.round(disc_out).int()
                acc = torch.sum(prediction == labels.unsqueeze(1)).float() / len(labels)

                d_logging_meters['valid_acc'].update(acc.item())
                d_logging_meters['valid_loss'].update(d_loss.item())
                # print("D dev loss {d_logging_meters['valid_loss'].avg:.3f}, acc {d_logging_meters['valid_acc'].avg:.3f} at batch {i}")

                if experiment is not None:
                    experiment.log_metric("d_batch_valid_acc", d_logging_meters['valid_acc'].avg)
                    experiment.log_metric("d_batch_valid_loss", d_logging_meters['valid_loss'].avg)

                # del src, trg, preds, loss, ...
                del src, trg, neg_tokens, fake_sentences, fake_labels, true_sentences, true_labels, src_sentences, \
                    trg_sentences, labels, disc_out, d_loss, acc, prediction

        # Log after each epoch
        print(
            "\nEpoch [{0}/{1}] complete. Generator: Train loss joint: {2:.3f}, avgAcc {3:.3f}. Train loss mle: {4:.3f}, avgAcc {5:.3f}. Val Loss: {6:.3f}, avgAcc {7:.3f}. lr={8}. "
            "\nDiscriminator: Train loss joint: {9:.3f}, avgAcc {10:.3f}. Val Loss: {11:.3f}, avgAcc {12:.3f}. lr={13}."
            "\nJoint training={14}, MLE training={15} "
            .format(epoch_i, max_epochs,
                    g_logging_meters['train_loss_joint'].avg, g_logging_meters['train_acc_joint'].avg,
                    g_logging_meters['train_loss_mle'].avg, g_logging_meters['train_acc_mle'].avg,
                    g_logging_meters['valid_loss'].avg, g_logging_meters['valid_acc'].avg,
                    g_optimizer.param_groups[0]['lr'],
                    d_logging_meters['train_loss'].avg, d_logging_meters['train_acc'].avg,
                    d_logging_meters['valid_loss'].avg, d_logging_meters['valid_acc'].avg,
                    d_optimizer.param_groups[0]['lr'],
                    joint_training, mle_training
                    ))

        if experiment is not None:
            experiment.log_metric("epoch_train_loss_joint_g", g_logging_meters['train_loss_joint'].avg)
            experiment.log_metric("epoch_train_acc_joint_g", g_logging_meters['train_acc_joint'].avg)

            experiment.log_metric("epoch_train_loss_mle_g", g_logging_meters['train_loss_mle'].avg)
            experiment.log_metric("epoch_train_acc_mle_g", g_logging_meters['train_acc_mle'].avg)

            experiment.log_metric("current_lr_g", g_optimizer.param_groups[0]['lr'])

            experiment.log_metric("epoch_valid_loss_g", g_logging_meters['valid_loss'].avg)
            experiment.log_metric("epoch_valid_acc_g", g_logging_meters['valid_acc'].avg)

            experiment.log_metric("epoch_train_loss_d", d_logging_meters['train_loss'].avg)
            experiment.log_metric("epoch_train_acc_d", d_logging_meters['train_acc'].avg)

            experiment.log_metric("current_lr_d", d_optimizer.param_groups[0]['lr'])

            experiment.log_metric("epoch_valid_loss_d", d_logging_meters['valid_loss'].avg)
            experiment.log_metric("epoch_valid_acc_d", d_logging_meters['valid_acc'].avg)

            experiment.log_epoch_end(epoch_cnt=epoch_i)

        if best_valid_acc < g_logging_meters['valid_acc'].avg:
            best_valid_acc = g_logging_meters['valid_acc'].avg

            generator_model_path_to_save = checkpoints_path + "joint_{0:.3f}.epoch_{1}.pt".format(
                g_logging_meters['valid_loss'].avg, epoch_i)

            print("Save model to", generator_model_path_to_save)
            torch.save(generator.state_dict(), generator_model_path_to_save)

        if g_logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = g_logging_meters['valid_loss'].avg

            generator_model_path_to_save = checkpoints_path + "best_generator_g_model.pt"

            print("Save model to", generator_model_path_to_save)
            torch.save(generator.state_dict(), generator_model_path_to_save)

    if experiment is not None:
        experiment.log_metric("joint_training", joint_training)
        experiment.log_metric("mle_training", mle_training)


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


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Found ", torch.cuda.device_count(), " GPU devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device ", device, " for task")

    hyper_params = {
        "dataset": "mws",  # mws # iwslt
        "tokenizer": "word",  # wordpiece
        "sequence_length_src": 40,
        "sequence_length_tgt": 40,
        "batch_size": 50,
        "num_epochs": 15,
        "learning_rate_g": 1e-4,
        "learning_rate_d": 1e-4,
        "d_model": 512,
        "n_head": 8,
        "dim_feedforward": 2048,
        "n_layers": 6,
        "dropout": 0.1,
        "load_embedding_weights": False
    }

    bert_path = "/glusterfs/dfs-gfs-dist/abeggluk/zzz_bert_models_1/bert_base_cased_12"
    checkpoint_base = "/glusterfs/dfs-gfs-dist/abeggluk/mws_transformer/_1"
    #checkpoint_base = "./"  # "/glusterfs/dfs-gfs-dist/abeggluk/test_MK30_dataset/_1"
    # project_name = "newsela-transformer"  # newsela-transformer-bert-weights
    project_name = "gan-mws"  # newsela-transformer-bert-weights
    tracking_active = True
    base_path = "/glusterfs/dfs-gfs-dist/abeggluk/data_1"

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

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset, tokenizer,
                                                                    BOS_WORD, EOS_WORD, BLANK_WORD)

    BATCH_SIZE = hyper_params["batch_size"]
    # Create iterators to process text in batches of approx. the same length
    train_iter = get_iterator(train_data, BATCH_SIZE)
    val_iter = get_iterator(valid_data, BATCH_SIZE)

    ### Load Generator
    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)

    if experiment is not None:
        experiment.log_other("source_vocab_length", source_vocab_length)
        experiment.log_other("target_vocab_length", target_vocab_length)
        experiment.log_other("len_train_data", str(len(train_data)))

    bert_model = None
    if hyper_params["load_embedding_weights"]:
        bert_config = BertConfig.from_pretrained(bert_path, output_hidden_states=True)
        bert_model = BertModel.from_pretrained(bert_path, config=bert_config)

    ### Load Generator
    generator = Transformer(bert_model, d_model=hyper_params["d_model"], nhead=hyper_params["n_head"],
                            num_encoder_layers=hyper_params["n_layers"], num_decoder_layers=hyper_params["n_layers"],
                            dim_feedforward=hyper_params["dim_feedforward"], dropout=hyper_params["dropout"],
                            source_vocab_length=source_vocab_length, target_vocab_length=target_vocab_length,
                            load_embedding_weights=hyper_params["load_embedding_weights"])
    generator_path = "best_model.pt"
    generator_path = os.path.join(checkpoint_base, 'checkpoints/mle', generator_path)
    generator.load_state_dict(torch.load(generator_path))
    print("Generator is successfully loaded!")

    ### Load Discriminator
    discriminator = Discriminator(src_vocab_size=source_vocab_length, pad_id_src=SRC.vocab.stoi[BLANK_WORD],
                                  trg_vocab_size=target_vocab_length, pad_id_trg=TGT.vocab.stoi[BLANK_WORD],
                                  max_len_src=max_len_src, max_len_tgt=max_len_tgt, use_gpu=False)
    discriminator_path = "best_dmodel.pt"
    discriminator_path = os.path.join(checkpoint_base, 'checkpoints/discriminator', discriminator_path)
    discriminator.load_state_dict(torch.load(discriminator_path))
    print("Discriminator is successfully loaded!")

    ### Start Training
    NUM_EPOCHS = hyper_params["num_epochs"]

    if experiment is not None:
        with experiment.train():
            train(train_iter, val_iter, generator, discriminator, NUM_EPOCHS, TGT.vocab, checkpoint_base, use_cuda,
                  experiment)
    else:
        train(train_iter, val_iter, generator, discriminator, NUM_EPOCHS, TGT.vocab, checkpoint_base, use_cuda,
              experiment)

    print("Training finished")
    sys.exit()
