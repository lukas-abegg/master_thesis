import os

from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize_bert(text):
    return [tok for tok in bert_tokenizer.tokenize(text)]


def get_fields(max_len, tokenizer, bos_word, eos_word, blank_word):

    src = Field(tokenize=tokenizer,
                fix_length=max_len,
                init_token=bos_word,
                eos_token=eos_word,
                pad_token=blank_word)

    trg = Field(tokenize=tokenizer,
                fix_length=max_len,
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


def load_dataset_data(max_len, dataset, bos_word, eos_word, blank_word):

    SRC, TGT = get_fields(max_len, tokenize_bert, bos_word, eos_word, blank_word)

    if dataset == "newsela":
        PATH = "data/test/newsela"

        train_data, valid_data, test_data = Newsela.splits(exts=('.src', '.dst'),
                                                           fields=(SRC, TGT),
                                                           train='train',
                                                           validation='valid',
                                                           test='test',
                                                           path=PATH,
                                                           filter_pred=lambda x: len(vars(x)['src']) <= max_len and len(
                                                               vars(x)['trg']) <= max_len)
    else:
        PATH = "data/test/wiki_simple"

        train_data, valid_data, test_data = MWS.splits(exts=('.src', '.dst'),
                                                       fields=(SRC, TGT),
                                                       train='train',
                                                       validation='valid',
                                                       test='test',
                                                       path=PATH,
                                                       filter_pred=lambda x: len(vars(x)['src']) <= max_len and len(
                                                           vars(x)['trg']) <= max_len)

    SRC.build_vocab([train_data.src, valid_data.src, test_data.src], min_freq=1)
    TGT.build_vocab([train_data.trg, valid_data.trg, test_data.trg], min_freq=1)

    return train_data, valid_data, test_data, SRC, TGT


def get_iterator(data, batch_size):
    return BucketIterator(data, batch_size=batch_size, repeat=False, sort_key=lambda x: len(x.src))