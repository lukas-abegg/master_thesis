import os
import re
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F

from load_datasets import load_dataset_data, get_iterator, bert_tokenizer
from test_transformer import Transformer


def predict_sentence(model, origin_sentence, beam_size, tokenizer, tgt_vocab, use_gpu=False):
    num_candidates = 2

    if beam_size > 1:
        translated_sentences = beam_decode_sentence(model, origin_sentence, beam_size, num_candidates, tgt_vocab, use_gpu)
        translated_sentence_1, translated_sentence_2 = translated_sentences[0], translated_sentences[1]
    else:
        translated_sentence_1 = greedy_decode_sentence(model, origin_sentence, tokenizer, tgt_vocab, use_gpu)
        translated_sentence_2 = greedy_decode_sentence(model, origin_sentence, tokenizer, tgt_vocab, use_gpu)

    return translated_sentence_1, translated_sentence_2


def predict(test_iter, model, beam_size, tokenizer, src_vocab, tgt_vocab, use_gpu):
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
                                                                              tokenizer, tgt_vocab, use_gpu)

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
    len_map = torch.arange(1, max_len_tgt + 1, dtype=torch.long).unsqueeze(0)
    len_map = len_map.cuda() if use_gpu else len_map

    sentence_tensor = torch.unsqueeze(origin_sentence, 0)

    trg_init_tok = tgt_vocab.stoi[BOS_WORD]
    trg_pad_tok = tgt_vocab.stoi[BLANK_WORD]
    trg_eos_tok = tgt_vocab.stoi[EOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]])

    sentence_tensor = sentence_tensor.cuda() if use_gpu else sentence_tensor
    trg = trg.cuda() if use_gpu else trg

    size = trg.size(0)

    np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
    np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
    np_mask = np_mask.cuda() if use_gpu else np_mask

    pred = model(sentence_tensor.transpose(0, 1), trg, tgt_mask=np_mask)
    pred = F.softmax(pred, dim=-1)

    best_k_probs, best_k_idx = pred[:, -1, :].topk(beam_size)

    scores = torch.log(best_k_probs).view(beam_size)

    blank_seqs = torch.full((beam_size, max_len_tgt), trg_pad_tok, dtype=torch.long)

    gen_seq = blank_seqs.clone().detach()
    gen_seq[:, 0] = trg_init_tok
    gen_seq[:, 1] = best_k_idx[0]
    input_tensors = sentence_tensor.repeat(beam_size, 1)

    ans_idx_seq = [0, 1]
    for step in range(2, max_len_tgt):  # decode up to max length
        size = gen_seq[0, :step].size(0)

        input_tensors = input_tensors.cuda() if use_gpu else input_tensors
        gen_seq = gen_seq.cuda() if use_gpu else gen_seq

        np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.cuda() if use_gpu else np_mask

        pred = model(input_tensors.transpose(0, 1), gen_seq[:, :step].transpose(0, 1), tgt_mask=np_mask)
        pred = F.softmax(pred, dim=-1)

        gen_seq, scores = _get_the_best_score_and_idx(gen_seq, pred, scores, step, beam_size)

        # Check if all path finished
        # -- locate the eos in the generated sequences
        eos_locs = gen_seq == trg_eos_tok
        # -- replace the eos with its position for the length penalty use
        seq_lens, _ = len_map.masked_fill(~eos_locs, max_len_tgt).min(1)

        scores = scores.to(device)
        seq_lens = seq_lens.to(device)

        # -- check if all beams contain eos
        if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
            _, ans_idx_seq = scores.div(seq_lens.float() ** 0.7).topk(num_candidates)
            break

    candidates = [gen_seq[ans_idx][:seq_lens[ans_idx]].tolist() for ans_idx in ans_idx_seq]
    hs = [_decode_candidate(h, tgt_vocab) for h in candidates]

    del gen_seq, input_tensors, np_mask, sentence_tensor, trg

    return hs


def _decode_candidate(hypothesis, vocab):
    sentence = []

    for elem in hypothesis:
        token = vocab.itos[elem]

        if token != BLANK_WORD and token != EOS_WORD:
            sentence.append(token)

    translated_sentence = bert_tokenizer.convert_tokens_to_string(sentence)
    return translated_sentence


def _get_the_best_score_and_idx(gen_seq, dec_output, scores, step, beam_size):
    assert len(scores.size()) == 1

    # Get k candidates for each beam, k^2 candidates in total.
    best_k2_probs, best_k2_idx = dec_output[-1, :, :].topk(beam_size)

    # Include the previous scores.
    scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

    # Get the best k candidates from k^2 candidates.
    scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

    # Get the corresponding positions of the best k candidiates.
    best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
    best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

    # Copy the corresponding previous tokens.
    gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
    # Set the best tokens in this beam search step
    gen_seq[:, step] = best_k_idx

    return gen_seq, scores


def greedy_decode_sentence(model, origin_sentence, tokenizer, tgt_vocab, use_gpu=False):
    sentence_tensor = torch.unsqueeze(origin_sentence, 0)

    trg_init_tok = tgt_vocab.stoi[BOS_WORD]
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
        add_word = tgt_vocab.itos[pred.argmax(dim=2)[-1]]

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


def run(test_iter, model, base_path, tokenizer, src_vocab, tgt_vocab, use_cuda):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in [2, 1, 4, 6, 12]:
        beam_size = i

        origin_sentences_1, origin_sentences_2, reference_sentences_1, reference_sentences_2, \
        predicted_sentences_1, predicted_sentences_2 = predict(test_iter, model, beam_size, tokenizer, src_vocab,
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
    tokenizer = set["tokenizer"]

    max_len_src = hyper_params["sequence_length_src"]
    max_len_tgt = hyper_params["sequence_length_tgt"]

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

    train_data, valid_data, test_data, SRC, TGT = load_dataset_data(base_path, max_len_src, max_len_tgt, dataset,
                                                                    tokenizer, BOS_WORD, EOS_WORD, BLANK_WORD)

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
        "batch_size": 50,
        "d_model": 512,
        "n_head": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1
    }

    hyper_params_newsela_wordpiece = {
        "sequence_length_src": 70,
        "sequence_length_tgt": 45,
        "batch_size": 50,
        "d_model": 512,
        "n_head": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1
    }

    hyper_params_mws_word = {
        "sequence_length_src": 56,
        "sequence_length_tgt": 49,
        "batch_size": 50,
        "d_model": 512,
        "n_head": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1
    }

    hyper_params_mws_wordpiece = {
        "sequence_length_src": 76,
        "sequence_length_tgt": 65,
        "batch_size": 50,
        "d_model": 512,
        "n_head": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1
    }

    hyper_params_pwkp_wordpiece = {
        "sequence_length_src": 80,
        "sequence_length_tgt": 70,
        "batch_size": 50,
        "d_model": 512,
        "n_head": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1
    }

    experiments_newsela_word = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer/_4", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt", "n_layers": 4}
    ]

    experiments_newsela_wordpiece = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer/_6", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt", "n_layers": 4},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer/_6", "eval": "evaluation/joint/reward/beam_1",
         "model": "checkpoints/joint/reward/best_generator_g_model.pt", "n_layers": 4},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer/_6_1", "eval": "evaluation/joint/reward/beam_2",
         "model": "checkpoints/joint/reward/joint_2.769.epoch_4.pt", "n_layers": 4},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer/_6_1", "eval": "evaluation/joint/pgloss/beam",
         "model": "checkpoints/joint/pgloss/best_generator_g_model.pt", "n_layers": 4},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer/_6", "eval": "evaluation/joint/pgloss_sari/beam_2",
         "model": "checkpoints/joint/pgloss_sari/joint_2.803.epoch_5.pt", "n_layers": 4},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/newsela_transformer/_6", "eval": "evaluation/joint/pgloss_sari/beam_1",
         "model": "checkpoints/joint/pgloss_sari/best_generator_g_model.pt", "n_layers": 4}
    ]

    experiments_mws_word = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_transformer/_0_2", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt", "n_layers": 6},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_transformer/_0_3", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt", "n_layers": 4}
    ]

    experiments_mws_wordpiece = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_transformer/_2_2", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt", "n_layers": 6},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_transformer/_2_3", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt", "n_layers": 4},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_transformer/_2_3", "eval": "evaluation/joint/pgloss/beam_1",
         "model": "checkpoints/joint/pgloss/joint_1.144.epoch_1.pt", "n_layers": 4},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_transformer/_2_3_1", "eval": "evaluation/joint/pgloss/beam_2",
         "model": "checkpoints/joint/pgloss/best_generator_g_model.pt", "n_layers": 4},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/mws_transformer/_2_3", "eval": "evaluation/joint/reward/beam",
         "model": "checkpoints/joint/reward/best_generator_g_model.pt", "n_layers": 4}
    ]

    experiments_pwkp_wordpiece = [
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/pwkp_transformer/_3", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt", "n_layers": 6},
        {"base_path": "/glusterfs/dfs-gfs-dist/abeggluk/pwkp_transformer/_4", "eval": "evaluation/mle/beam",
         "model": "checkpoints/mle/best_model.pt", "n_layers": 4}
    ]

    experiments = [
        # {"transformer_model": "newsela_word", "dataset": "newsela", "tokenizer": "word", "experiments": experiments_newsela_word},
        # {"transformer_model": "newsela_wordpiece", "dataset": "newsela", "tokenizer": "wordpiece", "experiments": experiments_newsela_wordpiece},
        # {"transformer_model": "mws_word", "dataset": "mws", "tokenizer": "word", "experiments": experiments_mws_word},
        # {"transformer_model": "mws_wordpiece", "dataset": "mws", "tokenizer": "wordpiece", "experiments": experiments_mws_wordpiece}
        {"transformer_model": "pwkp_wordpiece", "dataset": "pwkp", "tokenizer": "wordpiece", "experiments": experiments_pwkp_wordpiece}
    ]

    for set in experiments:

        if set["dataset"] == "newsela":
            if set["tokenizer"] == "word":
                hyper_params = hyper_params_newsela_word
            else:
                hyper_params = hyper_params_newsela_wordpiece
        elif set["dataset"] == "pwkp":
                hyper_params = hyper_params_pwkp_wordpiece
        else:
            if set["tokenizer"] == "word":
                hyper_params = hyper_params_mws_word
            else:
                hyper_params = hyper_params_mws_wordpiece

        print("Run experiments for transformer model: ", set["transformer_model"])
        print(hyper_params)

        test_iter, src_vocab, tgt_vocab, BOS_WORD, EOS_WORD, BLANK_WORD, source_vocab_length, \
        target_vocab_length = init_data(hyper_params, set)

        for experiment in set["experiments"]:
            model = Transformer(None, d_model=hyper_params["d_model"], nhead=hyper_params["n_head"],
                                num_encoder_layers=experiment["n_layers"], num_decoder_layers=experiment["n_layers"],
                                dim_feedforward=hyper_params["dim_feedforward"], dropout=hyper_params["dropout"],
                                source_vocab_length=source_vocab_length, target_vocab_length=target_vocab_length,
                                load_embedding_weights=False)

            model_path = os.path.join(experiment["base_path"], experiment["model"])
            model.load_state_dict(torch.load(model_path))

            save_run_files_base = os.path.join(experiment["base_path"], experiment["eval"])
            max_len_tgt = hyper_params["sequence_length_tgt"]

            print("Run experiment: ", model_path)
            print("Save in: ", save_run_files_base)

            run(test_iter, model, save_run_files_base, set["tokenizer"], src_vocab, tgt_vocab, use_cuda)

            del model

    sys.exit()
