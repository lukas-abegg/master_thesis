import os

from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from transformers import BertTokenizer

import spacy

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize_bert(text):
    return [tok for tok in bert_tokenizer.tokenize(text)]


spacy_en = spacy.load('en')
spacy_de = spacy.load('de')


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def get_fields(max_len_src, max_len_tgt, tokenizer_src, tokenizer_dst, bos_word, eos_word, blank_word):
    src = Field(tokenize=tokenizer_src,
                fix_length=max_len_src,
                #init_token=bos_word,
                #eos_token=eos_word,
                pad_token=blank_word)

    trg = Field(tokenize=tokenizer_dst,
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


class WIKILARGE(TranslationDataset):
    name = 'wikilarge'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='train', validation='valid', test='test', **kwargs):
        """Create dataset objects for splits of the Wiki-Large dataset.

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


class IWSLT(TranslationDataset):
    name = 'iwslt'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='train', validation='valid', test='test', **kwargs):
        """Create dataset objects for splits of the IWSLT dataset.

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

        return super(IWSLT, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


def load_dataset_data(base_path, max_len_src, max_len_tgt, dataset, bos_word, eos_word, blank_word):
    if dataset == "newsela":
        SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenize_bert, tokenize_bert, bos_word, eos_word, blank_word)
        #SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenize_en, tokenize_en, bos_word, eos_word, blank_word)

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
    elif dataset == "mws":
        SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenize_bert, tokenize_bert, bos_word, eos_word, blank_word)
        #SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenize_en, tokenize_en, bos_word, eos_word, blank_word)

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
    elif dataset == "pwkp":
        #SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenize_bert, tokenize_bert, bos_word, eos_word, blank_word)
        SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenize_en, tokenize_en, bos_word, eos_word, blank_word)

        path = os.path.join(base_path, "pwkp")

        train_data, valid_data, test_data = PWKP.splits(exts=('.src', '.dst'),
                                                        fields=(SRC, TGT),
                                                        train='train',
                                                        validation='valid',
                                                        test='test',
                                                        path=path,
                                                        filter_pred=lambda x: len(
                                                            vars(x)['src']) <= max_len_src and len(
                                                            vars(x)['trg']) <= max_len_tgt)

    elif dataset == "wikilarge":
        #SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenize_bert, tokenize_bert, bos_word, eos_word, blank_word)
        SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenize_en, tokenize_en, bos_word, eos_word, blank_word)

        path = os.path.join(base_path, "wikilarge")

        train_data, valid_data, test_data = WIKILARGE.splits(exts=('.src', '.dst'),
                                                        fields=(SRC, TGT),
                                                        train='train',
                                                        validation='valid',
                                                        test='test',
                                                        path=path,
                                                        filter_pred=lambda x: len(
                                                            vars(x)['src']) <= max_len_src and len(
                                                            vars(x)['trg']) <= max_len_tgt)

    else:
        SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenize_en, tokenize_de, bos_word, eos_word, blank_word)

        path = os.path.join(base_path, "iwslt")

        train_data, valid_data, test_data = IWSLT.splits(exts=('.en', '.de'),
                                                         fields=(SRC, TGT),
                                                         path=path,
                                                         filter_pred=lambda x: len(
                                                             vars(x)['src']) <= max_len_src and len(
                                                             vars(x)['trg']) <= max_len_tgt)

    SRC.build_vocab([train_data.src, valid_data.src, test_data.src], min_freq=2)
    TGT.build_vocab([train_data.trg, valid_data.trg, test_data.trg], min_freq=2)

    return train_data, valid_data, test_data, SRC, TGT


def get_iterator(data, batch_size):
    return BucketIterator(data, batch_size=batch_size, repeat=False, sort_key=lambda x: len(x.src))
