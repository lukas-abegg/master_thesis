import math
import os
import sys
from collections import OrderedDict
from comet_ml import Experiment

import torch
import torch.nn as nn

from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from torchtext.vocab import GloVe
from tqdm import tqdm
from transformers import BertTokenizer

from comparing_models.seq2seq.lstm_bi_model import LSTMModel, load_embeddings
from comparing_models.seq2seq.sequence_generator import SequenceGenerator
from meters import AverageMeter

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def get_fields(max_len_src, max_len_tgt, bos_word, eos_word, blank_word):
    src = Field(sequential=True, lower=True,
                tokenize="spacy",
                fix_length=max_len_src,
                # init_token=bos_word,
                # eos_token=eos_word,
                pad_token=blank_word)

    trg = Field(sequential=True, lower=True,
                tokenize="spacy",
                fix_length=max_len_tgt,
                init_token=bos_word,
                eos_token=eos_word,
                pad_token=blank_word)

    return src, trg


class Newsela(TranslationDataset):
    name = 'newsela'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='train', validation='valid', test='test', **kwargs):
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


class MWS(TranslationDataset):
    name = 'mws'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='train', validation='valid', test='test', **kwargs):
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

        return super(MWS, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


class PWKP(TranslationDataset):
    name = 'pwkp'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='train', validation='valid', test='test', **kwargs):
        """Create dataset objects for splits of the PWKP dataset.

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

        return super(PWKP, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


def load_dataset_data(base_path, max_len_src, max_len_tgt, dataset, bos_word, eos_word, blank_word):
    SRC, TGT = get_fields(max_len_src, max_len_tgt, bos_word, eos_word, blank_word)

    if dataset == "newsela":
        path = os.path.join(base_path, "newsela_w")
        #   path = os.path.join(base_path, "data/test/newsela_w")

        train_data, valid_data, test_data = Newsela.splits(exts=('.src', '.dst'),
                                                           fields=(SRC, TGT),
                                                           train='train',
                                                           validation='valid',
                                                           test='test',
                                                           path=path,
                                                           filter_pred=lambda x: len(
                                                               vars(x)['src']) <= max_len_src and len(
                                                               vars(x)['trg']) <= max_len_tgt)
    elif dataset == "mws":
        path = os.path.join(base_path, "wiki_simple_w")

        train_data, valid_data, test_data = MWS.splits(exts=('.src', '.dst'),
                                                       fields=(SRC, TGT),
                                                       train='train',
                                                       validation='test',
                                                       test='valid',
                                                       path=path,
                                                       filter_pred=lambda x: len(
                                                           vars(x)['src']) <= max_len_src and len(
                                                           vars(x)['trg']) <= max_len_tgt)
    else:
        path = os.path.join(base_path, "pwkp_w")

        train_data, valid_data, test_data = PWKP.splits(exts=('.src', '.dst'),
                                                        fields=(SRC, TGT),
                                                        train='train',
                                                        validation='valid',
                                                        test='test',
                                                        path=path,
                                                        filter_pred=lambda x: len(
                                                            vars(x)['src']) <= max_len_src and len(
                                                            vars(x)['trg']) <= max_len_tgt)

    SRC.build_vocab([train_data.src, valid_data.src, test_data.src], vectors=GloVe(name='6B', dim=300))
    TGT.build_vocab([train_data.trg, valid_data.trg, test_data.trg], vectors=GloVe(name='6B', dim=300))

    return train_data, valid_data, test_data, SRC, TGT


def get_iterator(data, batch_size):
    return BucketIterator(data, batch_size=batch_size, repeat=False, sort_key=lambda x: len(x.src))


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

    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=hyper_params["learning_rate"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.7)

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

            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)

            # Forward, backprop, optimizer
            preds = model(src, trg_input)
            preds = preds.contiguous().view(-1, preds.size(-1))
            loss = criterion(preds, targets)

            sample_size = targets.size(0)
            logging_loss = loss / sample_size

            # Accuracy
            predicts = preds.argmax(dim=1)
            corrects = predicts == targets
            all = targets == targets
            corrects.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
            all.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
            acc = corrects.sum().float() / all.sum()

            logging_meters['train_acc'].update(acc.item())
            logging_meters['train_loss'].update(logging_loss.item())
            # print("\nTraining loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
            #     logging_meters['train_loss'].avg,
            #     acc,
            #     logging_meters['train_acc'].avg,
            #     optimizer.param_groups[0]['lr'], step_train_i
            # ))

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

                src = batch.src.cuda() if use_gpu else batch.src
                trg = batch.trg.cuda() if use_gpu else batch.trg

                # change to shape (bs , max_seq_len)
                src = src.transpose(0, 1)
                # change to shape (bs , max_seq_len)
                trg = trg.transpose(0, 1)

                trg_input = trg[:, :-1]
                targets = trg[:, 1:].contiguous().view(-1)

                # Forward, backprop, optimizer
                preds = model(src, trg_input)
                preds = preds.contiguous().view(-1, preds.size(-1))
                loss = criterion(preds, targets)

                sample_size = targets.size(0)
                logging_loss = loss / sample_size

                # Accuracy
                predicts = preds.argmax(dim=1)
                corrects = predicts == targets
                all = targets == targets
                corrects.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                all.masked_fill_((targets == TGT.vocab.stoi[BLANK_WORD]), 0)
                acc = corrects.sum().float() / all.sum()

                logging_meters['valid_acc'].update(acc.item())
                logging_meters['valid_loss'].update(logging_loss.item())
                # print("\nValidation loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(
                #     logging_meters['valid_loss'].avg,
                #     acc,
                #     logging_meters['valid_acc'].avg,
                #     optimizer.param_groups[0]['lr'], step_valid_i
                # ))

                if experiment is not None:
                    experiment.log_metric("batch_valid_loss", logging_meters['valid_loss'].avg)
                    experiment.log_metric("batch_valid_acc", logging_meters['valid_acc'].avg)

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        #scheduler.step()

        # Log after each epoch
        print(
            "\nEpoch [{0}/{1}] complete. Train loss: {2:.3f}, avgAcc {3:.3f}. Val Loss: {4:.3f}, avgAcc {5:.3f}. lr={6}."
                .format(epoch, num_epochs, logging_meters['train_loss'].avg, logging_meters['train_acc'].avg,
                        logging_meters['valid_loss'].avg, logging_meters['valid_acc'].avg,
                        optimizer.param_groups[0]['lr']))

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
            sent = greedy_decode_sentence(model, sentences[step_i], use_gpu)
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
    # sentence = SRC.preprocess(BOS_WORD + " " + sentence + " " + EOS_WORD)
    sentence = SRC.preprocess(sentence)

    indexed = []
    for tok in sentence:
        if SRC.vocab.stoi[tok] != SRC.vocab.stoi[BLANK_WORD]:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(SRC.vocab.stoi[BLANK_WORD])

    src_tensor = torch.LongTensor(indexed).unsqueeze(0)

    src_tensor = src_tensor.cuda() if use_gpu else src_tensor

    generator = SequenceGenerator(model, beam_size=1, minlen=1, maxlen=max_len_tgt,
                                  stop_early=True, normalize_scores=True, len_penalty=1,
                                  unk_penalty=0, pad_idx=TGT.vocab.stoi[BLANK_WORD],
                                  unk_idx=TGT.vocab.unk_index, eos=TGT.vocab.stoi[EOS_WORD],
                                  len_tgt_vocab=target_vocab_length)

    trg_tokens = generator.generate(src_tensor, beam_size=1, maxlen=max_len_tgt)

    translated_sentence = [TGT.vocab.itos[i] for i in trg_tokens[0][0]["tokens"]]
    translated_sentence = bert_tokenizer.convert_tokens_to_string(translated_sentence)

    return translated_sentence


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Found ", torch.cuda.device_count(), " GPU devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device ", device, " for task")

    hyper_params = {
        "dataset": "newsela",  # mws #pwkp #newsela
        "sequence_length_src": 55,
        "sequence_length_tgt": 35,
        "batch_size": 64,
        "num_epochs": 30,
        "learning_rate": 1e-3,
        "num_layer": 2,
        "d_layer": 256,
        "d_embedding": 300,
        "dropout": 0.2,
        "pretrained_embeddings": True
    }

    checkpoint_base = "/glusterfs/dfs-gfs-dist/abeggluk/newsela_lstm/_2"
    project_name = "lstm-newsela"
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
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset,
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

    pretrained_embeddings = hyper_params["pretrained_embeddings"]

    if pretrained_embeddings:
        embeddings_in = load_embeddings(SRC.vocab, EMB_DIM, src_pad_idx)
        embeddings_out = load_embeddings(TGT.vocab, EMB_DIM, tgt_pad_idx)
    else:
        embeddings_in = None
        embeddings_out = None

    model = LSTMModel(source_vocab_length, target_vocab_length, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT, src_pad_idx,
                      tgt_pad_idx, embeddings_in=embeddings_in, embeddings_out=embeddings_out, use_cuda=use_cuda)

    ### Start Training
    NUM_EPOCHS = hyper_params["num_epochs"]

    if experiment is not None:
        with experiment.train():
            train(train_iter, val_iter, model, NUM_EPOCHS, checkpoint_base, use_cuda, experiment)
    else:
        train(train_iter, val_iter, model, NUM_EPOCHS, checkpoint_base, use_cuda, experiment)

    print("Training finished")
    sys.exit()
