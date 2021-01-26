import os
import sys

import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from torchtext.vocab import GloVe
from tqdm import tqdm

from comparing_models.seq2seq.lstm_bi_model import load_embeddings, LSTMModel
from comparing_models.seq2seq.sequence_generator import SequenceGenerator
from load_datasets import load_dataset_data, get_iterator, bert_tokenizer


def tokenize_bert(text):
    return [tok for tok in bert_tokenizer.tokenize(text)]


def get_fields(max_len_src, max_len_tgt, bos_word, eos_word, blank_word):
    src = Field(sequential=True, lower=True,
                tokenize="spacy",
                fix_length=max_len_src,
                # init_token=bos_word,
                # eos_token=eos_word,
                pad_token=blank_word)

    trg = Field(sequential=True, lower=True,
                tokenize="spacy",
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


def load_dataset_data(base_path, max_len_src, max_len_tgt, dataset, bos_word, eos_word, blank_word):
    SRC, TGT = get_fields(max_len_src, max_len_tgt, bos_word, eos_word, blank_word)

    if dataset == "newsela":
        path = os.path.join(base_path, "newsela/splits/bert_base")
        #   path = os.path.join(base_path, "data/test/newsela")

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
                                                       validation='test',
                                                       test='valid',
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

    SRC.build_vocab([train_data.src, valid_data.src, test_data.src], min_freq=2, vectors=GloVe(name='6B', dim=300))
    TGT.build_vocab([train_data.trg, valid_data.trg, test_data.trg], min_freq=2, vectors=GloVe(name='6B', dim=300))

    return train_data, valid_data, test_data, SRC, TGT


def get_iterator(data, batch_size):
    return BucketIterator(data, batch_size=batch_size, repeat=False, sort_key=lambda x: len(x.src))


def predict_sentence(model, origin_sentence, beam_size, tgt_vocab, use_gpu=False):
    num_candidates = 2

    translated_sentences = beam_decode_sentence(model, origin_sentence, beam_size, num_candidates, tgt_vocab, use_gpu)
    translated_sentence_1, translated_sentence_2 = translated_sentences[0], translated_sentences[1]

    return translated_sentence_1, translated_sentence_2


def predict(test_iter, model, beam_size, src_vocab, tgt_vocab, use_gpu):
    if use_gpu:
        model.cuda()
    else:
        model.cpu()

    origin_sentences_1 = []
    origin_sentences_2 = []

    reference_sentences_1 = []
    reference_sentences_2 = []

    predicted_sentences_1 = []
    predicted_sentences_2 = []

    model.eval()
    with torch.no_grad():

        print("Predicting started - ")
        desc = '  - (Predicting)   '

        for batch in tqdm(test_iter, desc=desc, leave=False):

            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg

            # change to shape (bs , max_seq_len)
            src = src.transpose(0, 1)
            # change to shape (bs , max_seq_len+1) , Since right shifted
            trg = trg.transpose(0, 1)

            for origin_sentence, reference_sentence in zip(src, trg):
                predicted_sentence_1, predicted_sentence_2 = predict_sentence(model, origin_sentence, beam_size,
                                                                              tgt_vocab, use_gpu)

                origin_sentences_1.append(convert_ids_to_tokens(origin_sentence, src_vocab))
                origin_sentences_2.append(convert_ids_to_tokens(origin_sentence, src_vocab))

                reference_sentence = reference_sentence[1:]
                reference_sentences_1.append(convert_ids_to_tokens(reference_sentence, tgt_vocab))
                reference_sentences_2.append(convert_ids_to_tokens(reference_sentence, tgt_vocab))

                predicted_sentences_1.append(predicted_sentence_1)
                predicted_sentences_2.append(predicted_sentence_2)

        print("Predicting finished - ")

        return origin_sentences_1, origin_sentences_2, reference_sentences_1, reference_sentences_2, \
               predicted_sentences_1, predicted_sentences_2


def convert_ids_to_tokens(tensor, vocab):
    sentence = []

    for elem in tensor:
        token = vocab.itos[elem.item()]

        if token != BLANK_WORD and token != EOS_WORD:
            sentence.append(token)

    translated_sentence = bert_tokenizer.convert_tokens_to_string(sentence)
    return translated_sentence


def beam_decode_sentence(model, origin_sentence, beam_size, num_candidates, tgt_vocab, use_gpu=False):
    src_tensor = origin_sentence.unsqueeze(0)

    src_tensor = src_tensor.cuda() if use_gpu else src_tensor

    generator = SequenceGenerator(model, beam_size=beam_size, minlen=1, maxlen=max_len_tgt,
                                  stop_early=True, normalize_scores=True, len_penalty=1,
                                  unk_penalty=0, pad_idx=tgt_vocab.stoi[BLANK_WORD],
                                  unk_idx=tgt_vocab.unk_index, eos=tgt_vocab.stoi[EOS_WORD],
                                  len_tgt_vocab=target_vocab_length)

    def _translate_to_string(tokens):
        translated_sentence = [tgt_vocab.itos[j] for j in tokens]
        return bert_tokenizer.convert_tokens_to_string(translated_sentence)

    if beam_size > 1:
        trg_tokens = generator.generate(src_tensor, beam_size=beam_size, maxlen=max_len_tgt)

        translated_sentences = []
        for i in range(num_candidates):
            translated_sentences.append(_translate_to_string(trg_tokens[0][i]["tokens"][:-1]))
    else:
        trg_tokens_1 = generator.generate(src_tensor, beam_size=beam_size, maxlen=max_len_tgt)
        trg_tokens_2 = generator.generate(src_tensor, beam_size=beam_size, maxlen=max_len_tgt)

        translated_sentences = [_translate_to_string(trg_tokens_1[0][0]["tokens"][:-1]),
                                _translate_to_string(trg_tokens_2[0][0]["tokens"][:-1])]

    return translated_sentences


def write_to_file(sentences, filename):
    print("Write File - ", )
    file = open(filename, "w")
    for sentence in sentences:
        file.write(sentence + "\n")
    file.close()
    print("Sentences saved to file", filename)


def run(test_iter, model, base_path, src_vocab, tgt_vocab, use_cuda):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in [2, 1, 4, 6, 12]:
        beam_size = i

        origin_sentences_1, origin_sentences_2, reference_sentences_1, reference_sentences_2, \
        predicted_sentences_1, predicted_sentences_2 = predict(test_iter, model, beam_size, src_vocab,
                                                               tgt_vocab, use_cuda)

        filename = os.path.join(base_path, str(i) + "_origin_sentences_1.txt")
        write_to_file(origin_sentences_1, filename)
        filename = os.path.join(base_path, str(i) + "_origin_sentences_2.txt")
        write_to_file(origin_sentences_2, filename)

        filename = os.path.join(base_path, str(i) + "_reference_sentences_1.txt")
        write_to_file(reference_sentences_1, filename)
        filename = os.path.join(base_path, str(i) + "_reference_sentences_2.txt")
        write_to_file(reference_sentences_2, filename)

        filename = os.path.join(base_path, str(i) + "_predicted_sentences_1.txt")
        write_to_file(predicted_sentences_1, filename)
        filename = os.path.join(base_path, str(i) + "_predicted_sentences_2.txt")
        write_to_file(predicted_sentences_2, filename)


def init_data(hyper_params, set):
    base_path = "/glusterfs/dfs-gfs-dist/abeggluk/data_1"
    dataset = set["dataset"]

    max_len_src = hyper_params["sequence_length_src"]
    max_len_tgt = hyper_params["sequence_length_tgt"]

    ### Load Data
    # Special Tokens
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset,
                                                                    BOS_WORD, EOS_WORD, BLANK_WORD)

    ### Load Generator
    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)

    BATCH_SIZE = hyper_params["batch_size"]
    # Create iterators to process text in batches of approx. the same length
    test_iter = get_iterator(test_data, BATCH_SIZE)

    return test_iter, SRC.vocab, TGT.vocab, BOS_WORD, EOS_WORD, BLANK_WORD, source_vocab_length, target_vocab_length


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Found ", torch.cuda.device_count(), " GPU devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device ", device, " for task")

    hyper_params_newsela_word = {
        "sequence_length_src": 55,
        "sequence_length_tgt": 35,
        "batch_size": 64,
        "num_layer": 2,
        "d_layer": 256,
        "d_embedding": 300,
        "dropout": 0.2,
        "pretrained_embeddings": True
    }

    hyper_params_mws_word = {
        "sequence_length_src": 56,
        "sequence_length_tgt": 49,
        "batch_size": 64,
        "num_layer": 2,
        "d_layer": 256,
        "d_embedding": 300,
        "dropout": 0.2,
        "pretrained_embeddings": True
    }

    hyper_params_mws_word_sgd = {
        "sequence_length_src": 56,
        "sequence_length_tgt": 49,
        "batch_size": 64,
        "num_layer": 2,
        "d_layer": 500,
        "d_embedding": 300,
        "dropout": 0.3,
        "pretrained_embeddings": True
    }

    hyper_params_pwkp_word = {
        "sequence_length_src": 80,
        "sequence_length_tgt": 70,
        "batch_size": 64,
        "num_layer": 2,
        "d_layer": 256,
        "d_embedding": 300,
        "dropout": 0.2,
        "pretrained_embeddings": True
    }

    experiments_newsela_word = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_lstm", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_lstm/_1", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt"}
    ]

    experiments_mws_word = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_lstm/", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_lstm/_1", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt"}
    ]

    experiments_mws_word_sgd = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_lstm/_2", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt"}
    ]

    experiments_pwkp_word = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/pwkp_lstm/", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/pwkp_lstm/_1", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt"}
    ]

    experiments = [
        {"lstm_model": "newsela_word", "dataset": "newsela", "experiments": experiments_newsela_word},
        {"lstm_model": "mws_word", "dataset": "mws", "experiments": experiments_mws_word},
        {"lstm_model": "mws_word_sgd", "dataset": "mws", "experiments": experiments_mws_word_sgd},
        {"lstm_model": "pwkp_word", "dataset": "pwkp", "experiments": experiments_pwkp_word}
    ]

    for set in experiments:

        if set["dataset"] == "newsela":
            hyper_params = hyper_params_newsela_word
        elif set["dataset"] == "pwkp":
            hyper_params = hyper_params_pwkp_word
        else:
            if set["lstm_model"] == "mws_word_sgd":
                hyper_params = hyper_params_mws_word_sgd
            else:
                hyper_params = hyper_params_mws_word

        print("Run experiments for lstm model: ", set["lstm_model"])
        print(hyper_params)

        test_iter, src_vocab, tgt_vocab, BOS_WORD, EOS_WORD, BLANK_WORD, source_vocab_length, \
        target_vocab_length = init_data(hyper_params, set)

        for experiment in set["experiments"]:
            ### Load Generator

            EMB_DIM = hyper_params["d_embedding"]
            HID_DIM = hyper_params["d_layer"]
            NUM_LAYERS = hyper_params["num_layer"]
            DROPOUT = hyper_params["dropout"]
            src_pad_idx = src_vocab.stoi[BLANK_WORD]
            tgt_pad_idx = tgt_vocab.stoi[BLANK_WORD]

            pretrained_embeddings = hyper_params["pretrained_embeddings"]

            if pretrained_embeddings:
                embeddings_in = load_embeddings(src_vocab, EMB_DIM, src_pad_idx)
                embeddings_out = load_embeddings(tgt_vocab, EMB_DIM, tgt_pad_idx)
            else:
                embeddings_in = None
                embeddings_out = None

            model = LSTMModel(source_vocab_length, target_vocab_length, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT,
                              src_pad_idx, tgt_pad_idx, embeddings_in=embeddings_in, embeddings_out=embeddings_out,
                              use_cuda=use_cuda)

            model_path = os.path.join(experiment["base_path"], experiment["model"])
            model.load_state_dict(torch.load(model_path))

            save_run_files_base = os.path.join(experiment["base_path"], experiment["eval"])
            max_len_tgt = hyper_params["sequence_length_tgt"]

            print("Run experiment: ", model_path)
            print("Save in: ", save_run_files_base)

            run(test_iter, model, save_run_files_base, src_vocab, tgt_vocab, use_cuda)

            del model

    sys.exit()
