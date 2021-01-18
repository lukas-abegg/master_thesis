import os
import sys

import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration


def get_fields(max_len_src, max_len_tgt, tokenizer, blank_word):
    src = Field(tokenize=tokenizer.encode,
                fix_length=max_len_src,
                pad_token=blank_word,
                use_vocab=False)

    trg = Field(tokenize=tokenizer.encode,
                fix_length=max_len_tgt,
                pad_token=blank_word,
                use_vocab=False)

    return src, trg


class Newsela(TranslationDataset):
    name = 'newsela'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='data/test',
               train='train', validation='valid', test='test', **kwargs):
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
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(MWS, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


def load_dataset_data(base_path, max_len_src, max_len_tgt, dataset, tokenizer, blank_word):
    SRC, TGT = get_fields(max_len_src, max_len_tgt, tokenizer, blank_word)

    if dataset == "newsela":
        path = os.path.join(base_path, "newsela/splits/bert_base")
        #path = os.path.join(base_path, "data/test/newsela")

        train_data, valid_data, test_data = Newsela.splits(exts=('.src', '.dst'),
                                                           fields=(SRC, TGT),
                                                           train='train',
                                                           validation='valid',
                                                           test='test',
                                                           path=path,
                                                           filter_pred=lambda x: len(
                                                               vars(x)['src']) <= max_len_src and len(
                                                               vars(x)['trg']) <= max_len_tgt)
    else:
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

    return train_data, valid_data, test_data, SRC, TGT


def get_iterator(data, batch_size):
    return BucketIterator(data, batch_size=batch_size, repeat=False, sort_key=lambda x: len(x.src))


def predict(test_iter, model, tokenizer, beam_size, max_len_tgt, use_gpu=True, device="cpu"):
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
                predicted_sentence_1, predicted_sentence_2 = greedy_decode_sentence(model, origin_sentence, tokenizer,
                                                                                    beam_size, max_len_tgt, device)

                origin_sentences_1.append(convert_ids_to_tokens(origin_sentence, tokenizer))
                origin_sentences_2.append(convert_ids_to_tokens(origin_sentence, tokenizer))

                reference_sentence = reference_sentence[1:]
                reference_sentences_1.append(convert_ids_to_tokens(reference_sentence, tokenizer))
                reference_sentences_2.append(convert_ids_to_tokens(reference_sentence, tokenizer))

                predicted_sentences_1.append(predicted_sentence_1)
                predicted_sentences_2.append(predicted_sentence_2)

        print("Predicting finished - ")

    return origin_sentences_1, origin_sentences_2, reference_sentences_1, reference_sentences_2, \
           predicted_sentences_1, predicted_sentences_2


def convert_ids_to_tokens(tensor, tokenizer):
    translated_sentence = tokenizer.decode(tensor, clean_up_tokenization_spaces=False, skip_special_tokens=True)
    return translated_sentence


def greedy_decode_sentence(model, origin_sentence, tokenizer, beam_size, max_len_tgt, device):
    origin_sentence = torch.unsqueeze(origin_sentence, 0)
    origin_sentence = origin_sentence.to(device)

    if beam_size > 1:
        pred_sent = model.generate(
            origin_sentence,
            num_beams=beam_size,
            num_return_sequences=2,
            max_length=max_len_tgt,
            early_stopping=True)

        translated_sentence_1 = tokenizer.decode(pred_sent[0], clean_up_tokenization_spaces=False,
                                                 skip_special_tokens=True)
        translated_sentence_2 = tokenizer.decode(pred_sent[1], clean_up_tokenization_spaces=False,
                                                 skip_special_tokens=True)
    else:
        pred_sent_1 = model.generate(
            origin_sentence,
            num_beams=beam_size,
            max_length=max_len_tgt,
            early_stopping=True)

        translated_sentence_1 = tokenizer.decode(pred_sent_1[0], clean_up_tokenization_spaces=False, skip_special_tokens=True)

        pred_sent_2 = model.generate(
            origin_sentence,
            num_beams=beam_size,
            max_length=max_len_tgt,
            early_stopping=True)

        translated_sentence_2 = tokenizer.decode(pred_sent_2[0], clean_up_tokenization_spaces=False, skip_special_tokens=True)

    return translated_sentence_1, translated_sentence_2


def write_to_file(sentences, filename):
    print("Write File - ", )
    file = open(filename, "w")
    for sentence in sentences:
        file.write(sentence + "\n")
    file.close()
    print("Sentences saved to file", filename)


def run(test_iter, model, base_path, tokenizer, max_len_tgt, use_cuda, device):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in [1, 2, 4, 6, 12]:
        beam_size = i

        origin_sentences_1, origin_sentences_2, reference_sentences_1, reference_sentences_2, \
        predicted_sentences_1, predicted_sentences_2 = predict(test_iter, model, tokenizer, beam_size, max_len_tgt,
                                                               use_cuda, device)

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
    tokenizer = BartTokenizer.from_pretrained(set["bart_model"])
    base_model = BartForConditionalGeneration.from_pretrained(set["bart_model"])

    base_path = "/glusterfs/dfs-gfs-dist/abeggluk/data_2"
    dataset = set["dataset"]

    max_len_src = hyper_params["sequence_length_src"]
    max_len_tgt = hyper_params["sequence_length_tgt"]

    ### Load Data
    # Special Tokens
    BLANK_WORD = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset,
                                                                    tokenizer, BLANK_WORD)

    BATCH_SIZE = hyper_params["batch_size"]
    # Create iterators to process text in batches of approx. the same length
    test_iter = get_iterator(test_data, BATCH_SIZE)

    return test_iter, base_model, tokenizer


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Found ", torch.cuda.device_count(), " GPU devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device ", device, " for task")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hyper_params_newsela = {
        "sequence_length_src": 70,
        "sequence_length_tgt": 45,
        "batch_size": 15
    }

    hyper_params_mws = {
        "sequence_length_src": 76,
        "sequence_length_tgt": 65,
        "batch_size": 15
    }

    bart_large_experiments_newsela = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_0", "eval": "evaluation/mle",
         "model": "checkpoints/mle/best_model.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_2", "eval": "evaluation/mle",
         "model": "checkpoints/mle/best_model.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_4", "eval": "evaluation/mle",
         "model": "checkpoints/mle/best_model.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_0", "eval": "evaluation/joint/pg_loss_sari/_2",
         "model": "checkpoints/joint/pg_loss_sari/joint_4.053.epoch_2.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_0", "eval": "evaluation/joint/pg_loss_sari/_5",
         "model": "checkpoints/joint/pg_loss_sari/joint_4.834.epoch_5.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_0", "eval": "evaluation/joint/pg_loss_sari/_best",
         "model": "checkpoints/joint/pg_loss_sari/best_generator_g_model.pt"}
    ]

    bart_large_cnn_experiments_newsela = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_1", "eval": "evaluation/mle",
         "model": "checkpoints/mle/best_model.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_3", "eval": "evaluation/mle",
         "model": "checkpoints/mle/best_model.pt"}
    ]

    bart_large_experiments_mws = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_bart/_0", "eval": "evaluation/mle",
         "model": "checkpoints/mle/best_model.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_bart/_2", "eval": "evaluation/mle",
         "model": "checkpoints/mle/best_model.pt"},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_bart/_0", "eval": "evaluation/joint/pg_loss/_5",
         "model": "checkpoints/joint/pg_loss/best_generator_g_model.pt"}
    ]

    bart_large_cnn_experiments_mws = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_bart/_1", "eval": "evaluation/mle",
         "model": "checkpoints/mle/best_model.pt"}
    ]

    experiments = [
        {"bart_model": "facebook/bart-large", "dataset": "mws", "experiments": bart_large_experiments_mws},
        {"bart_model": "facebook/bart-large-cnn", "dataset": "mws", "experiments": bart_large_cnn_experiments_mws},
        # {"bart_model": "facebook/bart-large", "dataset": "newsela", "experiments": bart_large_experiments_newsela},
        # {"bart_model": "facebook/bart-large-cnn", "dataset": "newsela", "experiments": bart_large_cnn_experiments_newsela}
    ]

    for set in experiments:

        if set["dataset"] == "newsela":
            hyper_params = hyper_params_newsela
        else:
            hyper_params = hyper_params_mws

        print("Run experiments for bart model: ", set["bart_model"])
        print(hyper_params)

        test_iter, base_model, tokenizer = init_data(hyper_params, set)

        for experiment in set["experiments"]:

            model = base_model
            model_path = os.path.join(experiment["base_path"], experiment["model"])
            model.load_state_dict(torch.load(model_path))

            save_run_files_base = os.path.join(experiment["base_path"], experiment["eval"])
            max_len_tgt = hyper_params["sequence_length_tgt"]

            print("Run experiment: ", model_path)
            print("Save in: ", save_run_files_base)

            run(test_iter, model, save_run_files_base, tokenizer, max_len_tgt, use_cuda, device)

            del model

    sys.exit()
