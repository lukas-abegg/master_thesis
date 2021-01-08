import math
import os
import sys
from collections import OrderedDict

from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset
from tqdm import tqdm

from comet_ml import Experiment

import torch

from transformers import BartTokenizer, BartForConditionalGeneration

from meters import AverageMeter
from test_discriminator import Discriminator
from test_training_2_discriminator_dataloader_BART import prepare_training_data, DatasetProcessing, train_dataloader, \
    eval_dataloader


def get_fields(max_len_src, max_len_tgt, tokenizer, blank_word):

    src = Field(tokenize=tokenizer.encode,
                fix_length=max_len_src,
                pad_token=blank_word,
                use_vocab=False)

    trg = Field(tokenize=tokenizer.encode,
                fix_length=max_len_tgt,
                pad_token=blank_word,
                use_vocab=False)

    return src, trg


class Newsela(TranslationDataset):
    name = 'newsela'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='train', validation='valid', test='test', **kwargs):
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(Newsela, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


class MWS(TranslationDataset):
    name = 'mws'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='train', validation='valid', test='test', **kwargs):
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(MWS, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


def load_dataset_data(base_path, max_len_src, max_len_tgt, dataset, tokenizer, blank_word):
    SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenizer, blank_word)

    if dataset == "newsela":
        path = os.path.join(base_path, "newsela/splits/bert_base")
        #path = os.path.join(base_path, "data/test/newsela")

        train_data, valid_data, test_data = Newsela.splits(exts=('.src', '.dst'),
                                                           fields=(SRC, TGT),
                                                           train='train',
                                                           validation='valid',
                                                           test='test',
                                                           path=path,
                                                           filter_pred=lambda x: len(
                                                               vars(x)['src']) <= max_len_src and len(
                                                               vars(x)['trg']) <= max_len_tgt)
    else:
        path = os.path.join(base_path, "wiki_simple/splits/bert_base")

        train_data, valid_data, test_data = MWS.splits(exts=('.src', '.dst'),
                                                       fields=(SRC, TGT),
                                                       train='train',
                                                       validation='valid',
                                                       test='test',
                                                       path=path,
                                                       filter_pred=lambda x: len(
                                                           vars(x)['src']) <= max_len_src and len(
                                                           vars(x)['trg']) <= max_len_tgt)

    return train_data, valid_data, test_data, SRC, TGT


def get_iterator(data, batch_size):
    return BucketIterator(data, batch_size=batch_size, repeat=False, sort_key=lambda x: len(x.src))


def train(train_iter, val_iter, generator, discriminator, num_epochs, checkpoint_base, tokenizer, beam_size, use_gpu=False,
          experiment=None, device="cpu"):
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
    train = prepare_training_data(train_iter, generator, tokenizer, beam_size, max_len_tgt, use_gpu, device)
    print("Prepare Validation Data")
    valid = prepare_training_data(val_iter, generator, tokenizer, beam_size, max_len_tgt, use_gpu, device)

    data_train = DatasetProcessing(data=train, max_len_src=max_len_src, max_len_tgt=max_len_tgt)
    data_valid = DatasetProcessing(data=valid, max_len_src=max_len_src, max_len_tgt=max_len_tgt)

    step_train_i = 0

    # main training loop
    while lr > min_lr and epoch_i <= max_epoch:

        if experiment is not None:
            experiment.set_epoch(epoch_i)

        if (epoch_i % 5) == 0:
            print("Prepare Train Data")
            train = prepare_training_data(train_iter, generator, tokenizer, beam_size, max_len_tgt, use_gpu, device)
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
            print("D training loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
                logging_meters['train_loss'].avg,
                                                                                                                                                                                                                         acc,
                logging_meters['train_acc'].avg,
                optimizer.param_groups[0]['lr'], i
            ))

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
                print("D eval loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
                    logging_meters['valid_loss'].avg,
                    acc,
                    logging_meters['valid_acc'].avg,
                    optimizer.param_groups[0]['lr'], i
                ))

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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hyper_params = {
        "dataset": "newsela",  # mws
        "sequence_length_src": 70,
        "sequence_length_tgt": 45,
        "batch_size": 10,
        "num_epochs": 25,
        "learning_rate": 1e-4,
        "bart_model": "facebook/bart-large",  # facebook/bart-large-cnn,
        "beam_size": 1
    }

    tokenizer = BartTokenizer.from_pretrained(hyper_params["bart_model"])
    generator = BartForConditionalGeneration.from_pretrained(hyper_params["bart_model"])

    checkpoint_base = "./"  #"/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_0"
    project_name = "discriminator-bart-newsela"
    tracking_active = False
    base_path = "./"  #"/glusterfs/dfs-gfs-dist/abeggluk/data_6"

    max_len_src = hyper_params["sequence_length_src"]
    max_len_tgt = hyper_params["sequence_length_tgt"]

    dataset = hyper_params["dataset"]
    beam_size = hyper_params["beam_size"]

    experiment = None
    if tracking_active:
        # Add the following code anywhere in your machine learning file
        experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                                project_name=project_name,
                                workspace="abeggluk")
        experiment.log_parameters(hyper_params)

    ### Load Data
    # Special Tokens
    BOS_WORD = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    EOS_WORD = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    BLANK_WORD = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset,
                                                                    tokenizer, BLANK_WORD)

    BATCH_SIZE = hyper_params["batch_size"]
    # Create iterators to process text in batches of approx. the same length
    train_iter = get_iterator(train_data, BATCH_SIZE)
    val_iter = get_iterator(valid_data, BATCH_SIZE)

    ### Load Generator
    source_vocab_length = tokenizer.vocab_size
    target_vocab_length = tokenizer.vocab_size

    if experiment is not None:
        experiment.log_other("source_vocab_length", source_vocab_length)
        experiment.log_other("target_vocab_length", target_vocab_length)
        experiment.log_other("len_train_data", str(len(train_data)))

    ### Load Generator
    generator_path = "best_model.pt"
    generator_path = os.path.join(checkpoint_base, 'checkpoints/mle', generator_path)
    generator.load_state_dict(torch.load(generator_path))
    print("Generator is successfully loaded!")

    ### Load Discriminator
    discriminator = Discriminator(src_vocab_size=source_vocab_length, pad_id_src=tokenizer.pad_token_id,
                                  trg_vocab_size=target_vocab_length, pad_id_trg=tokenizer.pad_token_id,
                                  max_len_src=max_len_src, max_len_tgt=max_len_tgt, use_gpu=use_cuda)
    print("Discriminator is successfully loaded!")

    ### Start Training
    NUM_EPOCHS = hyper_params["num_epochs"]

    if experiment is not None:
        with experiment.train():
            train(train_iter, val_iter, generator, discriminator, NUM_EPOCHS, checkpoint_base, tokenizer, beam_size, use_cuda,
                  experiment, device)
    else:
        train(train_iter, val_iter, generator, discriminator, NUM_EPOCHS, checkpoint_base, tokenizer, beam_size, use_cuda,
              experiment, device)

    print("Training finished")
    sys.exit()
