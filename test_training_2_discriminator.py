import math
import os
import sys
from collections import OrderedDict
from tqdm import tqdm

from comet_ml import Experiment

import torch

from transformers import BertConfig, BertModel

from load_datasets import get_iterator, load_dataset_data
from meters import AverageMeter
from test_discriminator import Discriminator
from test_training_2_discriminator_dataloader import prepare_training_data, DatasetProcessing, train_dataloader, \
    eval_dataloader
from test_transformer import Transformer


def train(train_iter, val_iter, generator, discriminator, num_epochs, target_vocab, checkpoint_base, use_gpu=False,
          experiment=None):
    if use_gpu:
        generator.cuda()
        discriminator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    # check checkpoints saving path
    checkpoints_path = os.path.join(checkpoint_base, 'checkpoints/discriminator/')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    logging_meters = OrderedDict()
    logging_meters['train_loss'] = AverageMeter()
    logging_meters['train_acc'] = AverageMeter()
    logging_meters['valid_loss'] = AverageMeter()
    logging_meters['valid_acc'] = AverageMeter()

    # define loss function
    criterion = torch.nn.BCELoss()

    # define optimizer
    # optimizer = NoamOptimizer(filter(lambda p: p.requires_grad, model.parameters()), d_model=512, warmup_steps=2500)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, discriminator.parameters()),
                                 lr=hyper_params["learning_rate"], betas=(0.9, 0.98),
                                 eps=1e-9)

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
    print("Prepare Train Data")
    train = prepare_training_data(train_iter, generator, target_vocab, BOS_WORD, max_len_tgt, EOS_WORD, BLANK_WORD,
                                  use_gpu)
    print("Prepare Validation Data")
    valid = prepare_training_data(val_iter, generator, target_vocab, BOS_WORD, max_len_tgt, EOS_WORD, BLANK_WORD,
                                  use_gpu)

    data_train = DatasetProcessing(data=train, max_len_src=max_len_src, max_len_tgt=max_len_tgt)
    data_valid = DatasetProcessing(data=valid, max_len_src=max_len_src, max_len_tgt=max_len_tgt)

    step_train_i = 0

    # main training loop
    while lr > min_lr and epoch_i <= max_epoch:

        if experiment is not None:
            experiment.set_epoch(epoch_i)

        if (epoch_i % 5) == 0:
            print("Prepare Train Data")
            train = prepare_training_data(train_iter, generator, target_vocab, BOS_WORD, max_len_tgt, EOS_WORD,
                                          BLANK_WORD, use_gpu)
            data_train = DatasetProcessing(data=train, max_len_src=max_len_src, max_len_tgt=max_len_tgt)

        # discriminator training dataloader
        train_loader = train_dataloader(data_train, batch_size=BATCH_SIZE, sort_by_source_size=False)
        valid_loader = eval_dataloader(data_valid, batch_size=BATCH_SIZE)

        # set training mode
        discriminator.train()

        i = 0
        desc = '  - (Training)   '
        for batch in tqdm(train_loader, desc=desc, leave=False):
            step_train_i = step_train_i + 1
            i = i + 1

            if experiment is not None:
                experiment.set_step(step_train_i)

            sources = batch['src_tokens']
            targets = batch['trg_tokens'][:, 1:]
            labels = batch['labels'].unsqueeze(1)

            len_labels = batch['labels'].size(0)

            sources = sources.cuda() if use_gpu else sources
            targets = targets.cuda() if use_gpu else targets
            labels = labels.cuda() if use_gpu else labels

            disc_out = discriminator(sources, targets)

            loss = criterion(disc_out, labels.float())
            prediction = torch.round(disc_out).int()
            acc = torch.sum(prediction == labels).float() / len_labels

            logging_meters['train_acc'].update(acc.item())
            logging_meters['train_loss'].update(loss.item())
            # print("D training loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
            #     logging_meters['train_loss'].avg,
                                                                                                                                                                                                                     #     acc,
            #     logging_meters['train_acc'].avg,
            #     optimizer.param_groups[0]['lr'], i
            # ))

            if experiment is not None:
                experiment.log_metric("batch_train_loss", logging_meters['train_loss'].avg)
                experiment.log_metric("batch_train_acc", logging_meters['train_acc'].avg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer.step()

            # del src, trg, preds, loss, acc
            del sources, targets, disc_out, loss, prediction, acc

        # set validation mode
        discriminator.eval()

        with torch.no_grad():

            desc = '  - (Validation)   '
            for batch in tqdm(valid_loader, desc=desc, leave=False):

                sources = batch['src_tokens']
                targets = batch['trg_tokens'][:, 1:]
                labels = batch['labels'].unsqueeze(1)

                len_labels = batch['labels'].size(0)

                sources = sources.cuda() if use_gpu else sources
                targets = targets.cuda() if use_gpu else targets
                labels = labels.cuda() if use_gpu else labels

                disc_out = discriminator(sources, targets)

                loss = criterion(disc_out, labels.float())
                prediction = torch.round(disc_out).int()
                acc = torch.sum(prediction == labels).float() / len_labels

                logging_meters['valid_acc'].update(acc.item())
                logging_meters['valid_loss'].update(loss.item())
                # print("D eval loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
                #     logging_meters['valid_loss'].avg,
                #     acc,
                #     logging_meters['valid_acc'].avg,
                #     optimizer.param_groups[0]['lr'], i
                # ))

                if experiment is not None:
                    experiment.log_metric("batch_valid_loss", logging_meters['valid_loss'].avg)
                    experiment.log_metric("batch_valid_acc", logging_meters['valid_acc'].avg)

                # del src, trg, preds, loss, acc
                del sources, targets, disc_out, loss, prediction, acc

        # Log after each epoch
        print(
            "\nEpoch [{0}/{1}] complete. Train loss: {2:.3f}, avgAcc {3:.3f}. Val Loss: {4:.3f}, avgAcc {5:.3f}. lr={6}."
            .format(epoch_i, num_epochs, logging_meters['train_loss'].avg, logging_meters['train_acc'].avg,
                    logging_meters['valid_loss'].avg, logging_meters['valid_acc'].avg,
                    optimizer.param_groups[0]['lr']))

        lr_scheduler.step(logging_meters['valid_loss'].avg)

        if experiment is not None:
            experiment.log_metric("epoch_train_loss", logging_meters['train_loss'].avg)
            experiment.log_metric("epoch_train_acc", logging_meters['train_acc'].avg)

            experiment.log_metric("current_lr", optimizer.param_groups[0]['lr'])

            experiment.log_metric("epoch_valid_loss", logging_meters['valid_loss'].avg)
            experiment.log_metric("epoch_valid_acc", logging_meters['valid_acc'].avg)

            experiment.log_epoch_end(epoch_cnt=epoch_i)

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


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Found ", torch.cuda.device_count(), " GPU devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device ", device, " for task")

    hyper_params = {
        "dataset": "mws",  # mws # iwslt
        "tokenizer": "wordpiece",  # wordpiece
        "sequence_length_src": 76,
        "sequence_length_tgt": 65,
        "batch_size": 50,
        "num_epochs": 25,
        "learning_rate": 1e-4,
        "d_model": 512,
        "n_head": 8,
        "dim_feedforward": 2048,
        "n_layers": 4,
        "dropout": 0.1,
        "load_embedding_weights": False
    }

    bert_path = "/glusterfs/dfs-gfs-dist/abeggluk/zzz_bert_models_1/bert_base_cased_12"
    checkpoint_base = "/glusterfs/dfs-gfs-dist/abeggluk/mws_transformer/_2_3"
    project_name = "discriminator-mws"
    tracking_active = True
    base_path = "/glusterfs/dfs-gfs-dist/abeggluk/data_5"

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
