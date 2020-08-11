from datetime import datetime

from torchtext.data import BucketIterator
from transformers import BertTokenizer

from preprocessings.load_newsela import get_fields as newsela_fields, Newsela
from preprocessings.load_wiki_simple import get_fields as wikisimple_fields, WikiSimple
from utils.log import get_logger

""" Tracking """


def init_logger(evaluation_config):
    run_name_format = (
        "model={model}-"
        "bert_mode={bert_model}"
        "dataset={dataset}"
        "beam_size={beam_size}"
        "{timestamp}"
    )

    run_name = run_name_format.format(**evaluation_config, timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    logger = get_logger(run_name, save_log=evaluation_config['save_log'])
    logger.info(f'Run name : {run_name}')
    logger.info(evaluation_config)

    return logger, run_name


""" Datasets """


def load_data(dataset_name, dataset_config, hyperparameter_config, tokenizer: BertTokenizer, device):
    if dataset_name == 'newsela':
        SRC, TRG = newsela_fields(dataset_config['max_seq_length'], tokenizer)

        _, _, test_data = Newsela.splits(exts=('.src', '.dst'),
                                         fields=(SRC, TRG),
                                         root='data/test',
                                         # root='data/newsela/splits',
                                         train='test',
                                         validation='test',
                                         test='test')
    else:  # WikiSimple
        SRC, TRG = wikisimple_fields(dataset_config['max_seq_length'], tokenizer)

        _, _, test_data = WikiSimple.splits(exts=('.src', '.dst'),
                                            fields=(SRC, TRG),
                                            root='data/test',
                                            # root='data/wiki_simple/splits',
                                            train='test',
                                            validation='test',
                                            test='test')

    BATCH_SIZE = hyperparameter_config['batch_size']

    test_iterator = BucketIterator.splits(test_data,
                                          batch_size=BATCH_SIZE,
                                          device=device)

    return test_iterator, len(test_iterator)
