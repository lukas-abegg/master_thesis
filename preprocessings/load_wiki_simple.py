import os

from torchtext.data import Field
from torchtext.datasets import TranslationDataset
from transformers import BertTokenizer


def get_fields(max_seq_length, tokenizer: BertTokenizer):
    # Model parameter
    MAX_SEQ_LEN = max_seq_length
    PAD_INDEX = tokenizer.pad_token_id
    UNK_INDEX = tokenizer.unk_token_id
    EOS_INDEX = tokenizer.sep_token_id
    INIT_INDEX = tokenizer.cls_token_id

    src = Field(use_vocab=False,
                tokenize=tokenizer.encode,
                lower=False,
                include_lengths=False,
                batch_first=True,
                fix_length=MAX_SEQ_LEN,
                pad_token=PAD_INDEX,
                unk_token=UNK_INDEX,
                init_token=INIT_INDEX,
                eos_token=EOS_INDEX)

    trg = Field(use_vocab=False,
                tokenize=tokenizer.encode,
                lower=False,
                include_lengths=False,
                batch_first=True,
                fix_length=MAX_SEQ_LEN,
                pad_token=PAD_INDEX,
                unk_token=UNK_INDEX,
                init_token=INIT_INDEX,
                eos_token=EOS_INDEX)

    return src, trg


class WikiSimple(TranslationDataset):
    name = 'WikiSimple'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='train', validation='valid', test='test', **kwargs):
        """Create dataset objects for splits of the Wiki Simple dataset.

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

        return super(WikiSimple, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)
