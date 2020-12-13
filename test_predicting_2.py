import os
import re

import torch
import torch.nn.functional as F
# For data loading.
from torchtext import data
from torchtext.data import Field
from torchtext.datasets import TranslationDataset
from transformers import BertTokenizer, BertModel, BertConfig

from test_transformer import Transformer

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

bert_path = "zzz_bert_models/bert_base_cased_12"
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
bert_config = BertConfig.from_pretrained(bert_path, output_hidden_states=True)
bert_model = BertModel.from_pretrained(bert_path, config=bert_config)


def tokenize_bert(text):
    return [tok for tok in bert_tokenizer.tokenize(text)]


# Special Tokens
BOS_WORD = '[CLS]'
EOS_WORD = '[SEP]'
BLANK_WORD = '[PAD]'


def get_fields(max_seq_length, tokenizer):
    # Model parameter
    MAX_SEQ_LEN = max_seq_length

    src = Field(tokenize=tokenizer,
                fix_length=MAX_SEQ_LEN,
                pad_token=BLANK_WORD)

    trg = Field(tokenize=tokenizer,
                fix_length=MAX_SEQ_LEN,
                init_token=BOS_WORD,
                eos_token=EOS_WORD,
                pad_token=BLANK_WORD)

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


MAX_LEN = 40
SRC, TGT = get_fields(MAX_LEN, tokenize_bert)

dataset = "newsela"

if dataset == "newsela":
    PATH = "data/test/newsela"

    train_data, valid_data, test_data = Newsela.splits(exts=('.src', '.dst'),
                                                       fields=(SRC, TGT),
                                                       train='train',
                                                       validation='valid',
                                                       test='test',
                                                       path=PATH,
                                                       filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(
                                                           vars(x)['trg']) <= MAX_LEN)
else:
    PATH = "data/test/wiki_simple"

    train_data, valid_data, test_data = MWS.splits(exts=('.src', '.dst'),
                                                   fields=(SRC, TGT),
                                                   train='train',
                                                   validation='valid',
                                                   test='test',
                                                   path=PATH,
                                                   filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(
                                                       vars(x)['trg']) <= MAX_LEN)


MIN_FREQ = 2
SRC.build_vocab([train_data.src, valid_data.src, test_data.src], min_freq=MIN_FREQ)
TGT.build_vocab([train_data.trg, valid_data.trg, test_data.trg], min_freq=MIN_FREQ)

BATCH_SIZE = 60
# Create iterators to process text in batches of approx. the same length
test_iter = data.BucketIterator(test_data, batch_size=BATCH_SIZE, repeat=False, sort_key=lambda x: len(x.src))

source_vocab_length = len(SRC.vocab)
target_vocab_length = len(TGT.vocab)
model = Transformer(source_vocab_length=source_vocab_length, target_vocab_length=target_vocab_length)
model_path = "checkpoint_best_epoch.pt"
model.load_state_dict(torch.load(model_path))


def predict(test_iter, model, use_gpu=True):

    origin_sentences = []
    reference_sentences = []
    predicted_sentences = []

    model.eval()
    with torch.no_grad():

        print("Predicting started - ")
        for i, batch in enumerate(test_iter):

            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.src.cuda() if use_gpu else batch.trg

            # change to shape (bs , max_seq_len)
            src = src.transpose(0, 1)
            # change to shape (bs , max_seq_len+1) , Since right shifted
            trg = trg.transpose(0, 1)

            for origin_sentence, reference_sentence in zip(src, trg):
                predicted_sentence = greedy_decode_sentence(model, origin_sentence, use_gpu)

                origin_sentences.append(convert_ids_to_tokens(origin_sentence, SRC.vocab))

                reference_sentence = reference_sentence[1:]
                reference_sentences.append(convert_ids_to_tokens(reference_sentence, TGT.vocab))

                predicted_sentences.append(predicted_sentence)

        print("Predicting finished - ")

    return origin_sentences, reference_sentences, predicted_sentences


def convert_ids_to_tokens(tensor, vocab):
    sentence = []

    for elem in tensor:
        token = vocab.itos[elem.item()]

        if token != BLANK_WORD and token != EOS_WORD:
            sentence.append(token)

    translated_sentence = bert_tokenizer.convert_tokens_to_string(sentence)
    return translated_sentence


def greedy_decode_sentence(model, origin_sentence, use_gpu=False):
    sentence_tensor = torch.unsqueeze(origin_sentence, 0)

    trg_init_tok = TGT.vocab.stoi[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]])

    translated_sentence = ""

    if use_gpu:
        sentence_tensor = sentence_tensor.cuda()
        trg = trg.cuda()

    for i in range(MAX_LEN):
        size = trg.size(0)

        np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.cuda() if use_gpu else np_mask

        pred = model(sentence_tensor.transpose(0, 1), trg, tgt_mask=np_mask)
        pred = F.softmax(pred, dim=-1)
        add_word = TGT.vocab.itos[pred.argmax(dim=2)[-1]]

        if add_word == EOS_WORD:
            break

        translated_sentence += " " + add_word

        if use_gpu:
            trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
        else:
            trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]])))

    translated_sentence = re.split(r"\s+", translated_sentence)
    translated_sentence = bert_tokenizer.convert_tokens_to_string(translated_sentence)

    return translated_sentence


def write_to_file(sentences, filename):
    print("Write File - ", )
    file = open(filename, "w")
    for sentence in sentences:
        file.write(sentence + "\n")
    file.close()
    print("Sentences saved to file", filename)


for i in range(5):
    origin_sentences, reference_sentences, predicted_sentences = predict(test_iter, model, False)

    filename = str(i+1) + "_origin_sentences.txt"
    write_to_file(origin_sentences, filename)

    filename = str(i + 1) + "_reference_sentences.txt"
    write_to_file(reference_sentences, filename)

    filename = str(i + 1) + "_predicted_sentences.txt"
    write_to_file(predicted_sentences, filename)

