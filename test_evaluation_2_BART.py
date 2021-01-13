import os

from comet_ml import Experiment

import numpy as np

from metrics.bleu import BleuMetric
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

    origin_groups_1 = []
    origin_groups_2 = []

    reference_groups_1 = []
    reference_groups_2 = []

    prediction_groups_1 = []
    prediction_groups_2 = []

    for i in [1, 2, 4, 6, 12]:
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
           prediction_groups_1, prediction_groups_2


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


def validate(origin_groups, reference_groups, prediction_groups, experiment=None):
    bleu_score_nltk = 0
    #bleu_score_local = 0
    avg_sentence_bleu_scores = 0
    avg_meteor_scores = 0
    sari_score = 0

    best_beam_size = {
        "bleu_score_nltk": 0,
        "avg_sentence_bleu_scores": 0,
        "avg_meteor_scores": 0,
        "sari_score": 0
    }

    for i in range(len(origin_groups)):
        origins = origin_groups[i]
        references = reference_groups[i]
        predictions = prediction_groups[i]

        bleu_score_nltk_s, avg_sentence_bleu_scores_s, avg_meteor_scores_s, sari_score_s = validate_single_round(origins, references, predictions)

        if experiment is not None:
            experiment.log_metric("bleu_score_nltk_s", float(bleu_score_nltk_s))
            #experiment.log_metric("bleu_score_local_s", float(bleu_score_local_s))
            experiment.log_metric("avg_sentence_bleu_scores_s", float(avg_sentence_bleu_scores_s))
            experiment.log_metric("avg_meteor_scores_s", float(avg_meteor_scores_s))
            experiment.log_metric("sari_score_s", float(sari_score_s))

        if bleu_score_nltk < bleu_score_nltk_s:
            bleu_score_nltk = bleu_score_nltk_s
            best_beam_size["bleu_score_nltk"] = i+1

        #if bleu_score_local < bleu_score_local_s:
        #    bleu_score_local = bleu_score_local_s

        if avg_sentence_bleu_scores < avg_sentence_bleu_scores_s:
            avg_sentence_bleu_scores = avg_sentence_bleu_scores_s
            best_beam_size["avg_sentence_bleu_scores"] = i+1

        if avg_meteor_scores < avg_meteor_scores_s:
            avg_meteor_scores = avg_meteor_scores_s
            best_beam_size["avg_meteor_scores"] = i+1

        if sari_score < sari_score_s:
            sari_score = sari_score_s
            best_beam_size["sari_score"] = i+1

    return bleu_score_nltk, avg_sentence_bleu_scores, avg_meteor_scores, sari_score, best_beam_size


if __name__ == "__main__":
    project_name = "bart-mws-eval"  # newsela-transformer-bert-weights
    tracking_active = True
    base_file_path = "/glusterfs/dfs-gfs-dist/abeggluk/mws_bart/_0/evaluation/mle"

    experiment = None
    if tracking_active:
        # Add the following code anywhere in your machine learning file
        experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                                project_name=project_name,
                                workspace="abeggluk")

        experiment.log_other("evaluation_files_path", base_file_path)

    origin_groups_1, origin_groups_2, reference_groups_1, reference_groups_2, \
    prediction_groups_1, prediction_groups_2 = get_parallel_sentences(base_file_path)

    bleu_score_nltk_1, avg_sentence_bleu_scores_1, avg_meteor_scores_1, sari_score_1, best_beam_size_1 = \
        validate(origin_groups_1, reference_groups_1, prediction_groups_1, experiment)

    bleu_score_nltk_2, avg_sentence_bleu_scores_2, avg_meteor_scores_2, sari_score_2, best_beam_size_2 = \
        validate(origin_groups_2, reference_groups_2, prediction_groups_2, experiment)

    if experiment is not None:
        experiment.log_metric("bleu_score_nltk_1", float(bleu_score_nltk_1))
        #experiment.log_metric("bleu_score_local_1", float(bleu_score_local_1))
        experiment.log_metric("avg_sentence_bleu_scores_1", float(avg_sentence_bleu_scores_1))
        experiment.log_metric("avg_meteor_scores_1", float(avg_meteor_scores_1))
        experiment.log_metric("sari_score_1", float(sari_score_1))
        experiment.log_other("best_beam_size_1", best_beam_size_1)

        experiment.log_metric("bleu_score_nltk_2", float(bleu_score_nltk_2))
        # experiment.log_metric("bleu_score_local_2", float(bleu_score_local_2))
        experiment.log_metric("avg_sentence_bleu_scores_2", float(avg_sentence_bleu_scores_2))
        experiment.log_metric("avg_meteor_scores_2", float(avg_meteor_scores_2))
        experiment.log_metric("sari_score_2", float(sari_score_2))
        experiment.log_other("best_beam_size_2", best_beam_size_2)

    print("bleu_score_nltk 1 = ", bleu_score_nltk_1)
    #print("bleu_score_local 1 = ", bleu_score_local)
    print("avg_sentence_bleu_scores 1 = ", avg_sentence_bleu_scores_1)
    print("avg_meteor_scores 1 = ", avg_meteor_scores_1)
    print("sari_score 1 = ", sari_score_1)
    print("best_beam_size 1 = ", best_beam_size_1)

    print("\n")

    print("bleu_score_nltk 2 = ", bleu_score_nltk_2)
    # print("bleu_score_local 2 = ", bleu_score_local_2)
    print("avg_sentence_bleu_scores 2 = ", avg_sentence_bleu_scores_2)
    print("avg_meteor_scores 2 = ", avg_meteor_scores_2)
    print("sari_score 2 = ", sari_score_2)
    print("best_beam_size 2 = ", best_beam_size_2)
