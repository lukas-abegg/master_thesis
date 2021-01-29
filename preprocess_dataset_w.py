import re
from tqdm import tqdm
import os
import sys

from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize_bert(text):
    return [tok for tok in bert_tokenizer.tokenize(text)]


def get_fields(max_len_src, max_len_tgt, blank_word):
    src = Field(tokenize=tokenize_bert,
                fix_length=max_len_src,
                pad_token=blank_word)

    trg = Field(tokenize=tokenize_bert,
                fix_length=max_len_tgt,
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


def load_dataset_data(base_path, max_len_src, max_len_tgt, dataset, blank_word):
    SRC, TGT = get_fields(max_len_src, max_len_tgt, blank_word)

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
    elif dataset == "mws":
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
    else:
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

    SRC.build_vocab([train_data.src, valid_data.src, test_data.src], min_freq=2)
    TGT.build_vocab([train_data.trg, valid_data.trg, test_data.trg], min_freq=2)

    return train_data, valid_data, test_data, SRC, TGT


def get_iterator(data, batch_size):
    return BucketIterator(data, batch_size=batch_size, repeat=False, shuffle=False)


def preprocess_dataset(max_seq_len_src, max_seq_len_dst, batch_size, target_base_path, base_path, dataset):

    def __iter_batch(batch, vocab, pad_id, unk_token):
        sentences = []

        for sent in batch:
            translated_sentence = []
            for tok in sent:
                if pad_id != tok:
                    translated_sentence.append(vocab.itos[tok])
            translated_sentence = bert_tokenizer.convert_tokens_to_string(translated_sentence).replace(unk_token, "UNK")
            translated_sentence = re.split(r"\s+", translated_sentence)
            translated_sentence = bert_tokenizer.convert_tokens_to_string(
                ["UNK" if "UNK" in token else token for token in translated_sentence]
            )
            sentences.append(translated_sentence)

        return sentences

    def __write_to_file(sentences, filename):
        file = open(filename, "w")
        len_sent = len(sentences)
        for i, sentence in enumerate(sentences):
            if i < len_sent-1:
                file.write(sentence + "\n")
            else:
                file.write(sentence)
        file.close()
        print("Sentences saved to file", filename)

    def __iter_set(data_set, target_base_path, file_name, batch_size):
        data_iter = get_iterator(data_set, batch_size)

        pad_id_src = SRC.vocab.stoi[SRC.pad_token]
        pad_id_tgt = TGT.vocab.stoi[TGT.pad_token]

        text_src = []
        text_dst = []

        for batch in tqdm(data_iter, desc="Transform Batch - ", leave=False):
            src = batch.src
            tgt = batch.trg

            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)

            text_src += __iter_batch(src, SRC.vocab, pad_id_src, SRC.unk_token)
            text_dst += __iter_batch(tgt, TGT.vocab, pad_id_tgt, TGT.unk_token)

            # check target_base_path saving path
            if not os.path.exists(target_base_path):
                os.makedirs(target_base_path)

        src_path = os.path.join(target_base_path, file_name + ".src")
        dst_path = os.path.join(target_base_path, file_name + ".dst")

        __write_to_file(text_src, src_path)
        __write_to_file(text_dst, dst_path)

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_seq_len_src, max_seq_len_dst, dataset, "PAD")

    __iter_set(train_data, target_base_path, "train", batch_size)
    __iter_set(valid_data, target_base_path, "valid", batch_size)
    __iter_set(test_data, target_base_path, "test", batch_size)


def process_dataset(dataset, hyper_params):
    max_seq_len_src = hyper_params["sequence_length_src"]
    max_seq_len_dst = hyper_params["sequence_length_tgt"]
    batch_size = hyper_params["batch_size"]
    target_base_path = hyper_params["target_base_path"]
    base_path = "./data"

    preprocess_dataset(max_seq_len_src, max_seq_len_dst, batch_size, target_base_path, base_path, dataset)


hyper_params_newsela_wordpiece = {
    "sequence_length_src": 70,
    "sequence_length_tgt": 45,
    "batch_size": 64,
    "target_base_path": "./data/newsela_w"
}

hyper_params_mws_wordpiece = {
    "sequence_length_src": 76,
    "sequence_length_tgt": 65,
    "batch_size": 64,
    "target_base_path": "./data/wiki_simple_w"
}

hyper_params_pwkp_wordpiece = {
    "sequence_length_src": 80,
    "sequence_length_tgt": 70,
    "batch_size": 64,
    "target_base_path": "./data/pwkp_w"
}

process_dataset("newsela", hyper_params_newsela_wordpiece)
process_dataset("mws", hyper_params_mws_wordpiece)
process_dataset("pwkp", hyper_params_pwkp_wordpiece)

sys.exit()
