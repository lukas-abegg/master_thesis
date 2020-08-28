import os
from datetime import datetime

from transformers import BertTokenizer

from torchtext.data import BucketIterator
from preprocessings.load_newsela import get_fields as newsela_fields, Newsela
from preprocessings.load_wiki_simple import get_fields as wikisimple_fields, WikiSimple
from utils.log import get_logger

""" Tracking """


def init_logger(hyperparameter_config):
    run_name_format = (
        "ninp={transformer_ninp}-"
        "n_layers={transformer_n_layers}-"
        "nheads={transformer_nheads}-"
        "nhidden={transformer_nhidden}-"
        "dropout={transformer_dropout}-"
        "epochs={epochs}-"
        "batch_size={batch_size}-"
        "optimizer={optimizer}-"
        "loss={loss}-"
        "{timestamp}"
    )

    run_name = run_name_format.format(**hyperparameter_config, timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    logger = get_logger(run_name, save_log=hyperparameter_config['save_log'])
    logger.info(f'Run name : {run_name}')
    logger.info(hyperparameter_config)

    return logger, run_name


""" Datasets """


def load_data(base_dir, dataset_name, dataset_config, hyperparameter_config, tokenizer: BertTokenizer, device):
    PATH = os.path.join(base_dir, dataset_config[hyperparameter_config["bert_model"]]["path"])
    print("Load {} dataset from {}".format(dataset_name, PATH))

    if dataset_name == 'newsela':
        SRC, TRG = newsela_fields(dataset_config[hyperparameter_config["bert_model"]]["max_seq_length"], tokenizer)

        train_data, valid_data, _ = Newsela.splits(exts=('.src', '.dst'),
                                                   fields=(SRC, TRG),
                                                   train='train',
                                                   validation='valid',
                                                   test='test',
                                                   path=PATH)
    else:  # WikiSimple
        SRC, TRG = wikisimple_fields(dataset_config['max_seq_length'], tokenizer)

        train_data, valid_data, _ = WikiSimple.splits(exts=('.src', '.dst'),
                                                      fields=(SRC, TRG),
                                                      train='train',
                                                      validation='valid',
                                                      test='test',
                                                      path=PATH)

    BATCH_SIZE = hyperparameter_config['batch_size']

    train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data),
                                                           batch_size=BATCH_SIZE,
                                                           device=device)

    return train_iterator, valid_iterator, [len(train_iterator), len(valid_iterator)]
