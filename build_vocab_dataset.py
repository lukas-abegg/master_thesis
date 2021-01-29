import os
import sys

import spacy
from torchtext.data import Field
from torchtext.datasets import TranslationDataset

spacy_en = spacy.load('en')


def get_fields(max_len_src, max_len_tgt, blank_word):
    src = Field(sequential=True, lower=True,
                tokenize="spacy",
                fix_length=max_len_src,
                # init_token=bos_word,
                # eos_token=eos_word,
                pad_token=blank_word)

    trg = Field(sequential=True, lower=True,
                tokenize="spacy",
                fix_length=max_len_tgt,
                # init_token=bos_word,
                # eos_token=eos_word,
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


def load_dataset_data(max_seq_len_src, max_seq_len_dst, base_path, dataset, blank_word):
    SRC, TGT = get_fields(max_seq_len_src, max_seq_len_dst, blank_word)

    if dataset == "newsela":
        path = os.path.join(base_path, "newsela_w")
        # path = os.path.join(base_path, "data/test/newsela_w")

        train_data, valid_data, test_data = Newsela.splits(exts=('.src', '.dst'),
                                                           fields=(SRC, TGT),
                                                           train='train',
                                                           validation='valid',
                                                           test='test',
                                                           path=path,
                                                           filter_pred=lambda x: len(
                                                               vars(x)['src']) <= max_seq_len_src and len(
                                                               vars(x)['trg']) <= max_seq_len_dst)
    elif dataset == "mws":
        path = os.path.join(base_path, "wiki_simple/splits/bert_base")

        train_data, valid_data, test_data = MWS.splits(exts=('.src', '.dst'),
                                                       fields=(SRC, TGT),
                                                       train='train',
                                                       validation='valid',
                                                       test='test',
                                                       path=path,
                                                       filter_pred=lambda x: len(
                                                           vars(x)['src']) <= max_seq_len_src and len(
                                                           vars(x)['trg']) <= max_seq_len_dst)
    else:
        path = os.path.join(base_path, "pwkp")

        train_data, valid_data, test_data = PWKP.splits(exts=('.src', '.dst'),
                                                        fields=(SRC, TGT),
                                                        train='train',
                                                        validation='valid',
                                                        test='test',
                                                        path=path,
                                                        filter_pred=lambda x: len(
                                                            vars(x)['src']) <= max_seq_len_src and len(
                                                            vars(x)['trg']) <= max_seq_len_dst)

    SRC.build_vocab(train_data.src, train_data.trg)

    return train_data, valid_data, test_data, SRC


def write_vocab(vocab, path):
    f = open(path, "w")
    for key, val in vocab.freqs.items():
        f.write(key + " " + str(val) + "\n")
    f.close()


def build_vocab(dataset, hyper_params):
    max_seq_len_src = hyper_params["sequence_length_src"]
    max_seq_len_dst = hyper_params["sequence_length_tgt"]
    vocab_base_path = hyper_params["vocab_base_path"]
    base_path = "./data"

    train_data, valid_data, test_data, SRC = load_dataset_data(max_seq_len_src, max_seq_len_dst, base_path, dataset, "PAD")

    vocab_path = os.path.join(vocab_base_path, "vocab.txt")
    write_vocab(SRC.vocab, vocab_path)


hyper_params_newsela_word = {
    "sequence_length_src": 55,
    "sequence_length_tgt": 35,
    "vocab_base_path": "./data/newsela_w"
}

hyper_params_mws_word = {
    "sequence_length_src": 56,
    "sequence_length_tgt": 49,
    "vocab_base_path": "./data/wiki_simple_w"
}

hyper_params_pwkp_word = {
    "sequence_length_src": 80,
    "sequence_length_tgt": 70,
    "vocab_base_path": "./data/pwkp_w"
}

build_vocab("newsela", hyper_params_newsela_word)
build_vocab("mws", hyper_params_mws_word)
build_vocab("pwkp", hyper_params_pwkp_word)

sys.exit()
