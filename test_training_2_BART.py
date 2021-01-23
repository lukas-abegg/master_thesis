import math
import os
import sys
from collections import OrderedDict

from comet_ml import Experiment

import torch
import torch.nn as nn
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup

from meters import AverageMeter


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
        # path = os.path.join(base_path, "data/test/newsela")

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


def train(train_iter, val_iter, model, num_epochs, num_steps_epoch, checkpoint_base, tokenizer, use_gpu=True, experiment=None,
          device="cpu"):
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
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')

    # define optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=hyper_params["learning_rate"], betas=(0.9, 0.98),
                                 eps=1e-9)

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5)
    num_train_steps = num_steps_epoch * num_epochs
    warmup_steps = 4000
    print(num_train_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)

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

            trg_input = trg[:, :-1].contiguous()
            targets = trg[:, 1:].contiguous().view(-1)

            pad_token_id = tokenizer.pad_token_id
            src_mask = (src != pad_token_id)
            src_mask = src_mask.int()

            inputs = {
                "input_ids": src.to(device),
                "attention_mask": src_mask.to(device),
                "decoder_input_ids": trg_input.to(device)
            }

            # Forward, backprop, optimizer
            preds = model(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                          decoder_input_ids=inputs["decoder_input_ids"])
            # model outputs are always tuple in pytorch-transformers (see doc)
            preds = preds.logits
            preds = preds.contiguous().view(-1, preds.size(-1))
            loss = criterion(preds, targets)

            sample_size = targets.size(0)
            logging_loss = loss / sample_size

            # Accuracy
            predicts = preds.argmax(dim=1)
            corrects = predicts == targets
            all = targets == targets
            corrects.masked_fill_((targets == BLANK_WORD), 0)
            all.masked_fill_((targets == BLANK_WORD), 0)
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
            lr_scheduler.step()

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

                pad_token_id = tokenizer.pad_token_id
                src_mask = (src != pad_token_id)
                src_mask = src_mask.int()

                inputs = {
                    "input_ids": src.to(device),
                    "attention_mask": src_mask.to(device),
                    "decoder_input_ids": trg_input.to(device)
                }

                # Forward, backprop, optimizer
                preds = model(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                              decoder_input_ids=inputs["decoder_input_ids"])
                # model outputs are always tuple in pytorch-transformers (see doc)
                preds = preds.logits
                preds = preds.contiguous().view(-1, preds.size(-1))
                loss = criterion(preds, targets)

                sample_size = targets.size(0)
                logging_loss = loss / sample_size

                # Accuracy
                predicts = preds.argmax(dim=1)
                corrects = predicts == targets
                all = targets == targets
                corrects.masked_fill_((targets == BLANK_WORD), 0)
                all.masked_fill_((targets == BLANK_WORD), 0)
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
            sent = greedy_decode_sentence(model, sentences[step_i], tokenizer, device)
            print("Translated Sentence: {}".format(sent))
            print("Expected Sentence: {}".format(exp_sentences[step_i]))
            print("---------------------------------------")
            if experiment is not None:
                if step_i == 0:
                    experiment.log_text(str("Train Sample: " + sent))
                else:
                    experiment.log_text(str("Validation Sample: " + sent))


def greedy_decode_sentence(model, sentence, tokenizer, device):
    model.eval()
    tokenized_sentence = tokenizer([sentence], max_length=max_len_src, return_tensors='pt')
    tokenized_sentence = tokenized_sentence.to(device)

    pred_sent = model.generate(tokenized_sentence["input_ids"], num_beams=1, max_length=max_len_tgt)
    translated_sentence = tokenizer.decode(pred_sent[0], clean_up_tokenization_spaces=False, skip_special_tokens=True)

    return translated_sentence


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
        "batch_size": 15,
        "num_epochs": 10,
        "learning_rate": 5e-7,
        "bart_model": "facebook/bart-large"  # facebook/bart-large-cnn
    }

    tokenizer = BartTokenizer.from_pretrained(hyper_params["bart_model"])
    model = BartForConditionalGeneration.from_pretrained(hyper_params["bart_model"])

    checkpoint_base = "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_4"
    project_name = "bart-newsela"
    tracking_active = True
    base_path = "/glusterfs/dfs-gfs-dist/abeggluk/data_4"

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
    model_path = "best_model.pt"
    model_path = os.path.join("/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_2", 'checkpoints/mle', model_path)
    model.load_state_dict(torch.load(model_path))
    print("Generator is successfully loaded!")

    source_vocab_length = tokenizer.vocab_size
    target_vocab_length = tokenizer.vocab_size

    if len(train_data) % BATCH_SIZE > 0:
        num_steps = math.floor(len(train_data) / BATCH_SIZE) + 1
    else:
        num_steps = math.floor(len(train_data) / BATCH_SIZE)

    if experiment is not None:
        experiment.log_other("source_vocab_length", source_vocab_length)
        experiment.log_other("target_vocab_length", target_vocab_length)
        experiment.log_other("len_train_data", str(len(train_data)))
        experiment.log_other("num_steps", str(num_steps))

    ### Start Training
    NUM_EPOCHS = hyper_params["num_epochs"]

    if experiment is not None:
        with experiment.train():
            train(train_iter, val_iter, model, NUM_EPOCHS, num_steps, checkpoint_base, tokenizer, use_cuda, experiment, device)
    else:
        train(train_iter, val_iter, model, NUM_EPOCHS, num_steps, checkpoint_base, tokenizer, use_cuda, experiment, device)

    print("Training finished")
    sys.exit()
