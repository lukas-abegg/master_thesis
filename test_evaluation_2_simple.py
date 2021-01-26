import os

from comet_ml import Experiment

import numpy as np

from metrics.bleu import BleuMetric
from metrics.f1_precision import F1Metrics
from metrics.meteor import MeteorMetric
from metrics.sari import SARISentenceMetric

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

import spacy

spacy_en = spacy.load('en')


def tokenize_en(text):
    text = text.replace("<unk>", "UNK")
    return [tok.text for tok in spacy_en.tokenizer(text)]


def read_from_file(filename):
    file = open(filename, "r")
    lines = [line.replace("\n", "") for line in file.readlines()]
    file.close()
    return lines


def get_parallel_sentences(base_file_path):
    print("Read files -")

    beam_sizes = []

    origin_groups_1 = []
    origin_groups_2 = []

    reference_groups_1 = []
    reference_groups_2 = []

    prediction_groups_1 = []
    prediction_groups_2 = []

    for i in [1, 2, 4, 6, 12]:

        beam_sizes.append(i)

        filename = os.path.join(base_file_path, str(i) + "_origin_sentences_1.txt")
        origins = read_from_file(filename)
        origin_groups_1.append(origins)

        filename = os.path.join(base_file_path, str(i) + "_origin_sentences_2.txt")
        origins = read_from_file(filename)
        origin_groups_2.append(origins)

        filename = os.path.join(base_file_path, str(i) + "_reference_sentences_1.txt")
        references = read_from_file(filename)
        reference_groups_1.append(references)

        filename = os.path.join(base_file_path, str(i) + "_reference_sentences_2.txt")
        references = read_from_file(filename)
        reference_groups_2.append(references)

        filename = os.path.join(base_file_path, str(i) + "_predicted_sentences_1.txt")
        predictions = read_from_file(filename)
        prediction_groups_1.append(predictions)

        filename = os.path.join(base_file_path, str(i) + "_predicted_sentences_2.txt")
        predictions = read_from_file(filename)
        prediction_groups_2.append(predictions)

    print("Files read -")
    return origin_groups_1, origin_groups_2, reference_groups_1, reference_groups_2,\
           prediction_groups_1, prediction_groups_2, beam_sizes


def evaluate_sentence_bleu_and_meteor(origins, tokenized_origins, references, tokenized_references, hypotheses,
                                      tokenized_hypotheses):
    meteor = MeteorMetric()
    bleu_smoothing_function = SmoothingFunction()

    sentence_bleu_scores = []
    meteor_scores = []

    zipped = zip(origins, tokenized_origins, references, tokenized_references, hypotheses, tokenized_hypotheses)

    for origin, tokenized_origin, reference, tokenized_reference, hypothesis, tokenized_hypothesis in zipped:
        sentence_bleu_score = sentence_bleu([tokenized_reference], tokenized_hypothesis,
                                            smoothing_function=bleu_smoothing_function.method3)
        sentence_bleu_scores.append(sentence_bleu_score)

        meteor_score = meteor.evaluate_results(hypothesis, [reference])
        meteor_scores.append(meteor_score)

    sentence_bleu_scores = np.asarray(sentence_bleu_scores)
    avg_sentence_bleu_scores = np.mean(sentence_bleu_scores)

    meteor_scores = np.asarray(meteor_scores)
    avg_meteor_scores = np.mean(meteor_scores)

    return avg_sentence_bleu_scores, avg_meteor_scores


def tokenize(sentences):
    tokenized_sentences = []

    for sent in sentences:
        tokenized_sentences.append(tokenize_en(sent))
        # tokenized_sentences.append(bert_tokenizer.tokenize(sent))
    return tokenized_sentences


def validate_single_round(origins, references, predictions):
    bleu_smoothing_function = SmoothingFunction()
    #bleu = BleuMetric()
    sari = SARISentenceMetric()

    tokenized_origins = tokenize(origins)
    tokenized_references = tokenize(references)
    tokenized_predictions = tokenize(predictions)

    list_of_tokenized_references = []

    for ref in tokenized_references:
        list_of_tokenized_references.append([ref])

    bleu_score_nltk = corpus_bleu(list_of_tokenized_references, tokenized_predictions,
                                  smoothing_function=bleu_smoothing_function.method3)
    #bleu_score_local = bleu.evaluate_results(tokenized_predictions, list_of_tokenized_references)
    sari_score = sari.evaluate_results(tokenized_origins, tokenized_predictions, tokenized_references)

    avg_sentence_bleu_scores, avg_meteor_scores = evaluate_sentence_bleu_and_meteor(origins, tokenized_origins,
                                                                                    references, tokenized_references,
                                                                                    predictions, tokenized_predictions)

    return bleu_score_nltk, avg_sentence_bleu_scores, avg_meteor_scores, sari_score


def validate_f1_p(origins, references, predictions):
    f1_metrics = F1Metrics()

    tokenized_origins = tokenize(origins)
    tokenized_references = tokenize(references)
    tokenized_predictions = tokenize(predictions)

    list_of_tokenized_references = []

    for ref in tokenized_references:
        list_of_tokenized_references.append([ref])

    f1_keep, f1_add, p_del = f1_metrics.evaluate_results(tokenized_origins, tokenized_predictions, tokenized_references)

    return f1_keep, f1_add, p_del


def validate(origin_groups, reference_groups, prediction_groups, beam_sizes, experiment=None):
    bleu_score_nltk = 0
    #bleu_score_local = 0
    avg_sentence_bleu_scores = 0
    avg_meteor_scores = 0
    sari_score = 0
    f1_keep = 0
    f1_add = 0
    p_del = 0

    best_beam_size = {
        "bleu_score_nltk": 0,
        "avg_sentence_bleu_scores": 0,
        "avg_meteor_scores": 0,
        "sari_score": 0,
        "f1_keep": 0,
        "f1_add": 0,
        "p_del": 0
    }

    for i in range(len(origin_groups)):
        beam_size = beam_sizes[i]

        origins = origin_groups[i]
        references = reference_groups[i]
        predictions = prediction_groups[i]

        bleu_score_nltk_s, avg_sentence_bleu_scores_s, avg_meteor_scores_s, sari_score_s = validate_single_round(origins, references, predictions)
        f1_keep_s, f1_add_s, p_del_s = validate_f1_p(origins, references, predictions)

        print("bleu_score_nltk_" + str(beam_size), float(bleu_score_nltk_s))
        # print("bleu_score_local_"+beam_size), float(bleu_score_local_s))
        print("avg_sentence_bleu_scores_" + str(beam_size), float(avg_sentence_bleu_scores_s))
        print("avg_meteor_scores_" + str(beam_size), float(avg_meteor_scores_s))
        print("sari_score_" + str(beam_size), float(sari_score_s))
        print("f1_keep_" + str(beam_size), float(f1_keep_s))
        print("f1_add_" + str(beam_size), float(f1_add_s))
        print("p_del_" + str(beam_size), float(p_del_s))

        if experiment is not None:
            experiment.log_metric("bleu_score_nltk_"+str(beam_size), float(bleu_score_nltk_s))
            #experiment.log_metric("bleu_score_local_"+beam_size), float(bleu_score_local_s))
            experiment.log_metric("avg_sentence_bleu_scores_"+str(beam_size), float(avg_sentence_bleu_scores_s))
            experiment.log_metric("avg_meteor_scores_"+str(beam_size), float(avg_meteor_scores_s))
            experiment.log_metric("sari_score_"+str(beam_size), float(sari_score_s))
            experiment.log_metric("f1_keep_" + str(beam_size), float(f1_keep_s))
            experiment.log_metric("f1_add_" + str(beam_size), float(f1_add_s))
            experiment.log_metric("p_del_" + str(beam_size), float(p_del_s))

        if bleu_score_nltk < bleu_score_nltk_s:
            bleu_score_nltk = bleu_score_nltk_s
            best_beam_size["bleu_score_nltk"] = beam_size

        #if bleu_score_local < bleu_score_local_s:
        #    bleu_score_local = bleu_score_local_s

        if avg_sentence_bleu_scores < avg_sentence_bleu_scores_s:
            avg_sentence_bleu_scores = avg_sentence_bleu_scores_s
            best_beam_size["avg_sentence_bleu_scores"] = beam_size

        if avg_meteor_scores < avg_meteor_scores_s:
            avg_meteor_scores = avg_meteor_scores_s
            best_beam_size["avg_meteor_scores"] = beam_size

        if sari_score < sari_score_s:
            sari_score = sari_score_s
            best_beam_size["sari_score"] = beam_size

        if f1_keep < f1_keep_s:
            f1_keep = f1_keep_s
            best_beam_size["f1_keep"] = beam_size

        if f1_add < f1_add_s:
            f1_add = f1_add_s
            best_beam_size["f1_add"] = beam_size

        if p_del < p_del_s:
            p_del = p_del_s
            best_beam_size["p_del"] = beam_size

    return bleu_score_nltk, avg_sentence_bleu_scores, avg_meteor_scores, sari_score, best_beam_size, f1_keep, f1_add, p_del


if __name__ == "__main__":
    project_name = "transformer-newsela-eval-beam"  # newsela-transformer-bert-weights
    tracking_active = True
    base_file_path = "/glusterfs/dfs-gfs-dist/abeggluk/newsela_lstm/_1/evaluation/mle/beam"

    experiment = None
    if tracking_active:
        # Add the following code anywhere in your machine learning file
        experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                                project_name=project_name,
                                workspace="abeggluk")

        experiment.log_other("evaluation_files_path", base_file_path)

    origin_groups_1, origin_groups_2, reference_groups_1, reference_groups_2, \
    prediction_groups_1, prediction_groups_2, beam_sizes = get_parallel_sentences(base_file_path)

    if experiment is not None:
        experiment.set_step(1)

    print("\nHypothesis 1: \n")

    best_1_bleu_score_nltk, best_1_avg_sentence_bleu_scores, best_1_avg_meteor_scores, best_1_sari_score, \
    best_1_best_beam_size, best_1_f1_keep, best_1_f1_add, best_1_p_del = \
        validate(origin_groups_1, reference_groups_1, prediction_groups_1, beam_sizes, experiment)

    if experiment is not None:
        experiment.set_step(2)

    print("\nHypothesis 2: \n")

    best_2_bleu_score_nltk, best_2_avg_sentence_bleu_scores, best_2_avg_meteor_scores, best_2_sari_score, \
    best_2_best_beam_size, best_2_f1_keep, best_2_f1_add, best_2_p_del = \
        validate(origin_groups_2, reference_groups_2, prediction_groups_2, beam_sizes, experiment)

    if experiment is not None:
        experiment.log_metric("best_1_bleu_score_nltk", float(best_1_bleu_score_nltk))
        #experiment.log_metric("bleu_score_local_1", float(bleu_score_local_1))
        experiment.log_metric("best_1_avg_sentence_bleu_scores", float(best_1_avg_sentence_bleu_scores))
        experiment.log_metric("best_1_avg_meteor_scores", float(best_1_avg_meteor_scores))
        experiment.log_metric("best_1_sari_score", float(best_1_sari_score))
        experiment.log_metric("best_1_f1_keep", float(best_1_f1_keep))
        experiment.log_metric("best_1_f1_add", float(best_1_f1_add))
        experiment.log_metric("best_1_p_del", float(best_1_p_del))
        experiment.log_other("best_1_best_beam_size", best_1_best_beam_size)

        experiment.log_metric("best_2_bleu_score_nltk", float(best_2_bleu_score_nltk))
        # experiment.log_metric("bleu_score_local_2", float(bleu_score_local_2))
        experiment.log_metric("best_2_avg_sentence_bleu_scores", float(best_2_avg_sentence_bleu_scores))
        experiment.log_metric("best_2_avg_meteor_scores", float(best_2_avg_meteor_scores))
        experiment.log_metric("best_2_sari_score", float(best_2_sari_score))
        experiment.log_metric("best_2_f1_keep", float(best_2_f1_keep))
        experiment.log_metric("best_2_f1_add", float(best_2_f1_add))
        experiment.log_metric("best_2_p_del", float(best_2_p_del))
        experiment.log_other("best_2_best_beam_size", best_2_best_beam_size)

    print("\nBest values: \n")

    print("best_1_bleu_score_nltk = ", best_1_bleu_score_nltk)
    #print("bleu_score_local 1 = ", bleu_score_local)
    print("best_1_avg_sentence_bleu_scores = ", best_1_avg_sentence_bleu_scores)
    print("best_1_avg_meteor_scores = ", best_1_avg_meteor_scores)
    print("best_1_sari_score = ", best_1_sari_score)
    print("best_1_f1_keep", best_1_f1_keep)
    print("best_1_f1_add", best_1_f1_add)
    print("best_1_p_del", best_1_p_del)
    print("best_1_best_beam_size = ", best_1_best_beam_size)

    print("\n")

    print("best_2_bleu_score_nltk = ", best_2_bleu_score_nltk)
    # print("bleu_score_local 2 = ", bleu_score_local_2)
    print("best_2_avg_sentence_bleu_scores = ", best_2_avg_sentence_bleu_scores)
    print("best_2_avg_meteor_scores = ", best_2_avg_meteor_scores)
    print("best_2_sari_score = ", best_2_sari_score)
    print("best_2_f1_keep", best_2_f1_keep)
    print("best_2_f1_add", best_2_f1_add)
    print("best_2_p_del", best_2_p_del)
    print("best_2_best_beam_size = ", best_2_best_beam_size)

    print("\n")
