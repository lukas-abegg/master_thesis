import os
import sys
import traceback
from os.path import dirname, abspath

import torch
from torch.autograd import Variable

from build_models.choose_model import build_model, load_model
from configs.load_config import load_dataset_config, \
    load_prediction_config, load_hyperparameter_config_val
from layers.utils.bert_loader import BertModelLoader

from evaluation.evaluation_utils import load_data


def _greeedy_decode_sentence(sentence, model, SRC, TRG, device):
    sentence = SRC.preprocess(sentence)
    indexed = []

    for tok in sentence:
        if SRC.vocab.stoi[tok] != 1:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(1)
    print(sentence)
    print(indexed)
    sentence = Variable(torch.LongTensor([indexed])).to(device)
    trg_init_tok = TRG.vocab.stoi["[CLS]"]
    trg = torch.LongTensor([[trg_init_tok]]).to(device)
    translated_sentence = ""
    maxlen = 25
    for i in range(maxlen):

        # src_mask = (sentence != self.SRC.vocab.stoi['[PAD]'])
        # src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
        # src_mask = src_mask.to(self.device)
        #
        # memory_mask = src_mask.clone()
        # memory_mask = memory_mask.to(self.device)

        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.to(device)

        pred = model(sentence.transpose(0, 1), trg, tgt_mask=np_mask)  # , src_key_padding_mask=src_mask, memory_key_padding_mask=memory_mask)

        add_word = TRG.vocab.itos[pred.argmax(dim=2)[-1]]
        translated_sentence += " " + add_word
        if add_word == "[SEP]":
            break
        trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]]).to(device)))

    print(translated_sentence)


def load_origins(filepath):
    # Open a file: file
    origins = []
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            origins.append(line.strip())
            line = fp.readline()
            cnt += 1
    print(cnt, ' lines read from', filepath)
    return origins


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load Config
    BASE_DIR = dirname(abspath(__file__))
    prediction_config = load_prediction_config(BASE_DIR)
    hyperparameter_config = load_hyperparameter_config_val(BASE_DIR, prediction_config["load_config_file"])
    bert_model_name = hyperparameter_config["bert_model"]
    dataset_name = hyperparameter_config["dataset"]
    dataset_config = load_dataset_config(BASE_DIR, dataset_name)

    # Base dir for Loading
    base_dir_load = os.path.join(BASE_DIR, prediction_config['base_dir_load'])

    # Load Bert
    print('Loading bert...')
    bert_model_loader = BertModelLoader(bert_model_name, BASE_DIR, base_dir_load)
    tokenizer = bert_model_loader.tokenizer

    # Load Data
    origins = load_origins(os.path.join(BASE_DIR, prediction_config['origins_path']))

    # Load Dataset
    train_iterator, valid_iterator, lengths_sets, SRC, TRG = load_data(BASE_DIR, dataset_name, dataset_config,
                                                                       hyperparameter_config, tokenizer)

    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TRG.vocab)

    # Load Transformer
    print('Loading transformer...')
    model = build_model(hyperparameter_config, dataset_config, bert_model_loader.model, source_vocab_length, target_vocab_length)
    model = load_model(model, None, hyperparameter_config)

    for origin in origins:
        _greeedy_decode_sentence(origin, model, SRC, TRG, device)

    sys.exit()
