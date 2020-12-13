from metrics.bleu import BleuMetric
from metrics.meteor import MeteorMetric
from metrics.sari import SARISentenceMetric

import numpy as np

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

import spacy

spacy_en = spacy.load('en')


def tokenize_en(text):
    text = text.replace("<unk>", "UNK")
    return [tok.text for tok in spacy_en.tokenizer(text)]


bleu = BleuMetric()
bleu_smoothing_function = SmoothingFunction()
sari = SARISentenceMetric()
meteor = MeteorMetric()


def read_from_file(filename):
    file = open(filename, "r")
    lines = [line.replace("\n", "") for line in file.readlines()]
    file.close()
    return lines


def get_parallel_sentences():
    print("Read files -")

    origin_groups = []
    reference_groups = []
    prediction_groups = []

    for i in range(5):
        filename = str(i + 1) + "_origin_sentences.txt"
        origins = read_from_file(filename)
        origin_groups.append(origins)

        filename = str(i + 1) + "_reference_sentences.txt"
        references = read_from_file(filename)
        reference_groups.append(references)

        filename = str(i + 1) + "_predicted_sentences.txt"
        predictions = read_from_file(filename)
        prediction_groups.append(predictions)

    print("Files read -")
    return origin_groups, reference_groups, prediction_groups


def evaluate_sentence_bleu_and_meteor(origins, tokenized_origins, references, tokenized_references, hypotheses, tokenized_hypotheses):
    sentence_bleu_scores = []
    meteor_scores = []

    zipped = zip(origins, tokenized_origins, references, tokenized_references, hypotheses, tokenized_hypotheses)

    for origin, tokenized_origin, reference, tokenized_reference, hypothesis, tokenized_hypothesis in zipped:

        sentence_bleu_score = sentence_bleu([tokenized_reference], tokenized_hypothesis, smoothing_function=bleu_smoothing_function.method3)
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
        #tokenized_sentences.append(bert_tokenizer.tokenize(sent))
    return tokenized_sentences


def validate_single_round(origins, references, predictions):

    tokenized_origins = tokenize(origins)
    tokenized_references = tokenize(references)
    tokenized_predictions = tokenize(predictions)

    list_of_tokenized_references = []

    for ref in tokenized_references:
        list_of_tokenized_references.append([ref])


    bleu_score_nltk = corpus_bleu(list_of_tokenized_references, tokenized_predictions, smoothing_function=bleu_smoothing_function.method3)
    bleu_score_local = bleu.evaluate_results(tokenized_predictions, list_of_tokenized_references)
    sari_score = sari.evaluate_results(tokenized_origins, tokenized_predictions, tokenized_references)

    avg_sentence_bleu_scores, avg_meteor_scores = evaluate_sentence_bleu_and_meteor(origins, tokenized_origins, references, tokenized_references, predictions, tokenized_predictions)

    return bleu_score_nltk, bleu_score_local, avg_sentence_bleu_scores, avg_meteor_scores, sari_score


def validate(origin_groups, reference_groups, prediction_groups):
    bleu_score_nltk = 0
    bleu_score_local = 0
    avg_sentence_bleu_scores = 0
    avg_meteor_scores = 0
    sari_score = 0

    for i in range(len(origin_groups)):
        origins = origin_groups[i]
        references = reference_groups[i]
        predictions = prediction_groups[i]

        bleu_score_nltk_s, bleu_score_local_s, avg_sentence_bleu_scores_s, avg_meteor_scores_s, sari_score_s = validate_single_round(origins, references, predictions)

        if bleu_score_nltk < bleu_score_nltk_s:
            bleu_score_nltk = bleu_score_nltk_s

        if bleu_score_local < bleu_score_local_s:
            bleu_score_local = bleu_score_local_s

        if avg_sentence_bleu_scores < avg_sentence_bleu_scores_s:
            avg_sentence_bleu_scores = avg_sentence_bleu_scores_s

        if avg_meteor_scores < avg_meteor_scores_s:
            avg_meteor_scores = avg_meteor_scores_s

        if sari_score < sari_score_s:
            sari_score = sari_score_s

    return bleu_score_nltk, bleu_score_local, avg_sentence_bleu_scores, avg_meteor_scores, sari_score


origin_groups, reference_groups, prediction_groups = get_parallel_sentences()

bleu_score_nltk, bleu_score_local, avg_sentence_bleu_scores, avg_meteor_scores, sari_score = validate(origin_groups, reference_groups, prediction_groups)

print("bleu_score_nltk = ", bleu_score_nltk)
print("bleu_score_local = ", bleu_score_local)
print("avg_sentence_bleu_scores = ", avg_sentence_bleu_scores)
print("avg_meteor_scores = ", avg_meteor_scores)
print("sari_score = ", sari_score)