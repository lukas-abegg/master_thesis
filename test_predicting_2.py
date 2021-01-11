import os
import re
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from load_datasets import load_dataset_data, get_iterator, bert_tokenizer
from test_transformer import Transformer


def predict(test_iter, model, use_gpu=True):
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

    sentence_tensor = sentence_tensor.cuda() if use_gpu else sentence_tensor
    trg = trg.cuda() if use_gpu else trg

    for i in range(max_len_tgt):
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

    if tokenizer == "wordpiece":
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


def run(test_iter, model, base_path, use_cuda):

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in range(5):
        origin_sentences, reference_sentences, predicted_sentences = predict(test_iter, model, use_cuda)

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

    hyper_params = {
        "dataset": "newsela",  # mws
        "tokenizer": "wordpiece",  # wordpiece
        "sequence_length_src": 70,
        "sequence_length_tgt": 45,
        "batch_size": 50,
        "d_model": 512,
        "n_head": 8,
        "dim_feedforward": 2048,
        "n_layers": 4,
        "dropout": 0.1,
        "load_embedding_weights": False
    }

    bert_path = "/glusterfs/dfs-gfs-dist/abeggluk/zzz_bert_models_1/bert_base_cased_12"

    checkpoint_base = "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer/_6/checkpoints/joint/pgloss"
    save_run_files_base = "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer/_6/evaluation/joint/pgloss"

    base_path = "/glusterfs/dfs-gfs-dist/abeggluk/data_1"

    max_len_src = hyper_params["sequence_length_src"]
    max_len_tgt = hyper_params["sequence_length_tgt"]

    dataset = hyper_params["dataset"]
    tokenizer = hyper_params["tokenizer"]

    ### Load Data
    # Special Tokens
    if tokenizer == "wordpiece":
        BOS_WORD = '[CLS]'
        EOS_WORD = '[SEP]'
        BLANK_WORD = '[PAD]'
    else:
        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset, tokenizer,
                                                                    BOS_WORD, EOS_WORD, BLANK_WORD)

    BATCH_SIZE = hyper_params["batch_size"]
    # Create iterators to process text in batches of approx. the same length
    test_iter = get_iterator(test_data, BATCH_SIZE)

    ### Load Generator
    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)

    bert_model = None
    if hyper_params["load_embedding_weights"]:
        bert_config = BertConfig.from_pretrained(bert_path, output_hidden_states=True)
        bert_model = BertModel.from_pretrained(bert_path, config=bert_config)

    model = Transformer(bert_model, d_model=hyper_params["d_model"], nhead=hyper_params["n_head"],
                        num_encoder_layers=hyper_params["n_layers"], num_decoder_layers=hyper_params["n_layers"],
                        dim_feedforward=hyper_params["dim_feedforward"], dropout=hyper_params["dropout"],
                        source_vocab_length=source_vocab_length, target_vocab_length=target_vocab_length,
                        load_embedding_weights=hyper_params["load_embedding_weights"])

    model_path = os.path.join(checkpoint_base, "best_generator_g_model.pt")
    model.load_state_dict(torch.load(model_path))

    run(test_iter, model, save_run_files_base, use_cuda)

    sys.exit()
