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
                                                       validation='valid',
                                                       test='test',
                                                       path=path,
                                                       filter_pred=lambda x: len(
                                                           vars(x)['src']) <= max_len_src and len(
                                                           vars(x)['trg']) <= max_len_tgt)

    return train_data, valid_data, test_data, SRC, TGT


def get_iterator(data, batch_size):
    return BucketIterator(data, batch_size=batch_size, repeat=False, sort_key=lambda x: len(x.src))


def predict(test_iter, model, tokenizer, beam_size, use_gpu=True, device="cpu"):
    if use_gpu:
        model.cuda()
    else:
        model.cpu()

    origin_sentences = []
    reference_sentences = []
    predicted_sentences = []

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
                predicted_sentence = greedy_decode_sentence(model, origin_sentence, tokenizer, beam_size, device)

                origin_sentences.append(convert_ids_to_tokens(origin_sentence, tokenizer))

                reference_sentence = reference_sentence[1:]
                reference_sentences.append(convert_ids_to_tokens(reference_sentence, tokenizer))

                predicted_sentences.append(predicted_sentence)

        print("Predicting finished - ")

    return origin_sentences, reference_sentences, predicted_sentences


def convert_ids_to_tokens(tensor, tokenizer):
    translated_sentence = tokenizer.decode(tensor, clean_up_tokenization_spaces=False, skip_special_tokens=True)
    return translated_sentence


def greedy_decode_sentence(model, origin_sentence, tokenizer, beam_size, device):
    origin_sentence = torch.unsqueeze(origin_sentence, 0)
    origin_sentence = origin_sentence.to(device)

    pred_sent = model.generate(origin_sentence, num_beams=beam_size, max_length=max_len_tgt, early_stopping=True)
    translated_sentence = tokenizer.decode(pred_sent[0], clean_up_tokenization_spaces=False, skip_special_tokens=True)

    return translated_sentence


def write_to_file(sentences, filename):
    print("Write File - ", )
    file = open(filename, "w")
    for sentence in sentences:
        file.write(sentence + "\n")
    file.close()
    print("Sentences saved to file", filename)


def run(test_iter, model, base_path, tokenizer, beam_size, use_cuda, device):

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in range(5):
        origin_sentences, reference_sentences, predicted_sentences = predict(test_iter, model, tokenizer, beam_size, use_cuda, device)

        filename = os.path.join(base_path, str(i + 1) + "_origin_sentences.txt")
        write_to_file(origin_sentences, filename)

        filename = os.path.join(base_path, str(i + 1) + "_reference_sentences.txt")
        write_to_file(reference_sentences, filename)

        filename = os.path.join(base_path, str(i + 1) + "_predicted_sentences.txt")
        write_to_file(predicted_sentences, filename)


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Found ", torch.cuda.device_count(), " GPU devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device ", device, " for task")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hyper_params = {
        "dataset": "newsela",  # mws
        "sequence_length_src": 72,
        "sequence_length_tgt": 43,
        "batch_size": 50,
        "num_epochs": 15,
        "beam_size": 4,
        "bart_model": "facebook/bart-large"  # facebook/bart-large-cnn
    }

    tokenizer = BartTokenizer.from_pretrained(hyper_params["bart_model"])
    model = BartForConditionalGeneration.from_pretrained(hyper_params["bart_model"])

    checkpoint_base = "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_0/checkpoints/mle"
    save_run_files_base = "/glusterfs/dfs-gfs-dist/abeggluk/newsela_bart/_0/evaluation/mle"
    base_path = "/glusterfs/dfs-gfs-dist/abeggluk/data_1"

    max_len_src = hyper_params["sequence_length_src"]
    max_len_tgt = hyper_params["sequence_length_tgt"]

    dataset = hyper_params["dataset"]
    beam_size = hyper_params["beam_size"]

    ### Load Data
    # Special Tokens
    BOS_WORD = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    EOS_WORD = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    BLANK_WORD = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset,
                                                                    tokenizer, BLANK_WORD)

    BATCH_SIZE = hyper_params["batch_size"]
    # Create iterators to process text in batches of approx. the same length
    test_iter = get_iterator(test_data, BATCH_SIZE)

    ### Load Generator
    source_vocab_length = tokenizer.vocab_size
    target_vocab_length = tokenizer.vocab_size

    model_path = os.path.join(checkpoint_base, "best_model.pt")
    model.load_state_dict(torch.load(model_path))

    run(test_iter, model, save_run_files_base, tokenizer, beam_size, use_cuda, device)

    sys.exit()
