import os
from datetime import datetime

from torchtext.data import BucketIterator
from transformers import BertTokenizer

from preprocessings.load_newsela import get_fields as newsela_fields, Newsela
from preprocessings.load_wiki_simple import WikiSimple
from utils.log import get_logger

""" Tracking """


def init_logger(evaluation_config, hyperparameter_config):
    run_name_format = (
        "model_file={model_dir}/{model_file}-"
        "bert_model={bert_model}-"
        "dataset={dataset}-"
        "beam_size={beam_size}-"
        "num_candidates={num_candidates}-"
        "num_evals={num_evals}-"
        "{timestamp}"
    )

    run_name = run_name_format.format(**evaluation_config,
                                      dataset=hyperparameter_config["dataset"],
                                      bert_model=hyperparameter_config["bert_model"],
                                      timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    logger = get_logger(run_name, save_log=evaluation_config['save_log'])
    logger.info(f'Run name : {run_name}')
    logger.info(evaluation_config)

    return logger, run_name


""" Datasets """


def load_data(base_dir, dataset_name, dataset_config, hyperparameter_config, tokenizer: BertTokenizer):
    PATH = os.path.join(base_dir, dataset_config[hyperparameter_config["bert_model"]]["path"])
    print("Load {} dataset from {}".format(dataset_name, PATH))

    BATCH_SIZE = hyperparameter_config["batch_size"]
    MAX_LEN = os.path.join(base_dir, dataset_config[hyperparameter_config["bert_model"]]["max_seq_length"])

    if dataset_name == 'newsela':
        SRC, TRG = newsela_fields(dataset_config[hyperparameter_config["bert_model"]]["max_seq_length"], tokenizer)

        train_data, valid_data, test_data = Newsela.splits(exts=('.src', '.dst'),
                                                           fields=(SRC, TRG),
                                                           train='train',
                                                           validation='valid',
                                                           test='test',
                                                           path=PATH,
                                                           filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                                                                 len(vars(x)['trg']) <= MAX_LEN)

    else:  # WikiSimple
        SRC, TRG = newsela_fields(dataset_config[hyperparameter_config["bert_model"]]["max_seq_length"], tokenizer)

        train_data, valid_data, test_data = WikiSimple.splits(exts=('.src', '.dst'),
                                                              fields=(SRC, TRG),
                                                              train='train',
                                                              validation='valid',
                                                              test='test',
                                                              path=PATH,
                                                              filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                                                                    len(vars(x)['trg']) <= MAX_LEN)

    SRC.build_vocab(train_data.src, valid_data.src, train_data.src)
    TRG.build_vocab(train_data.trg, valid_data.trg, train_data.trg)

    # Create iterators to process text in batches of approx. the same length
    test_iterator = BucketIterator(test_data, batch_size=BATCH_SIZE, repeat=False, sort_key=lambda x: len(x.src))

    for i, example in enumerate([(x.src, x.trg) for x in test_data[0:5]]):
        print("Example_{}:{}".format(i, example))

    return test_iterator, len(test_iterator), SRC, TRG
