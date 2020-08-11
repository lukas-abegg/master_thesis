from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from tqdm import tqdm
from metrics.bleu import BleuMetric
from metrics.sari import SARISentenceMetric
from metrics.meteor import MeteorMetric


class Evaluator:

    def __init__(self, predictor, save_filepath, test_iterator, logger, config, experiment=None):

        self.test_iterator = test_iterator
        self.predictor = predictor

        self.bleu = BleuMetric()
        self.sari = SARISentenceMetric()
        self.meteor = MeteorMetric()

        self.smoothing_function = SmoothingFunction()

        self.save_filepath = save_filepath
        self.logger = logger
        self.config = config
        self.experiment = experiment

    def evaluate_sentence_bleu(self, origins, references, predictions, hypotheses, list_of_references):
        with open(self.save_filepath, 'w') as file:
            for source, target, prediction, hypothesis, references in zip(origins, references, predictions, hypotheses,
                                                                          list_of_references):
                sentence_bleu_score = sentence_bleu(references, hypothesis,
                                                    smoothing_function=self.smoothing_function.method3)

                line = "{bleu_score}\t{source}\t{target}\t|\t{prediction}".format(
                    bleu_score=sentence_bleu_score,
                    source=source,
                    target=target,
                    prediction=prediction
                )
                file.write(line + '\n')

                self.logger.info(line)
                if self.experiment is not None:
                    self.experiment.log_metric("bleu_score", sentence_bleu_score)

    def evaluate_dataset(self):
        tokenize = lambda x: x.split()

        predictions = []
        origins = []
        references = []
        for source, target in tqdm(self.test_iterator):
            prediction = self.predictor.predict_one(source, num_candidates=1)[0]
            predictions.append(prediction)
            origins.append(source)
            references.append(target)

        hypotheses = [tokenize(prediction) for prediction in predictions]
        list_of_origins = [[tokenize(origin)] for origin in origins]
        list_of_references = [[tokenize(reference)] for reference in references]

        self.evaluate_sentence_bleu(origins, references, predictions, hypotheses, list_of_references)
        bleu_score_nltk = corpus_bleu(list_of_references, hypotheses,
                                      smoothing_function=self.smoothing_function.method3)
        bleu_score_local = self.bleu.evaluate_results(hypotheses, list_of_references)

        sari_score = self.sari.evaluate_results(list_of_origins, hypotheses, list_of_references)
        meteor_score = self.meteor.evaluate_results(hypotheses, list_of_references)

        if self.experiment is not None:
            self.experiment.log_metric("bleu_score_nltk", bleu_score_nltk)
            self.experiment.log_metric("bleu_score", bleu_score_local)
            self.experiment.log_metric("sari_score", sari_score)
            self.experiment.log_metric("meteor_score", meteor_score)

        return bleu_score_local, sari_score, meteor_score
