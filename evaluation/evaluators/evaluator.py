from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from tqdm import tqdm
from metrics.bleu import BleuMetric
from metrics.sari import SARISentenceMetric
from metrics.meteor import MeteorMetric
import nltk
nltk.download('wordnet')


class Evaluator:

    def __init__(self, predictor, save_filepath, test_iterator, logger, config, device, experiment=None, SRC=None, TRG=None):

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
        self.device = device

        self.SRC = SRC
        self.TRG = TRG

    def _evaluate_sentence_bleu_and_meteor(self, origins, tokenized_origins, references,
                                           tokenized_references, hypotheses, tokenized_hypotheses, round):
        with open(self.save_filepath, 'w') as file:
            for origin, tokenized_origin, reference, tokenized_reference, hypothesis, tokenized_hypothesis \
                    in zip(origins, tokenized_origins, references, tokenized_references, hypotheses, tokenized_hypotheses):

                sentence_bleu_score = sentence_bleu([tokenized_reference], tokenized_hypothesis,
                                                    smoothing_function=self.smoothing_function.method3)

                meteor_score = self.meteor.evaluate_results(hypothesis, [reference])

                line = "{bleu_score}\t{meteor_score}\t{origin}\t{reference}\t|\t{hypothesis}".format(
                    bleu_score=sentence_bleu_score,
                    meteor_score=meteor_score,
                    origin=origin,
                    reference=reference,
                    hypothesis=hypothesis
                )
                file.write(line + '\n')

                self.logger.info(line)
                if self.experiment is not None:
                    self.experiment.log_metric("bleu_score", sentence_bleu_score)
                    self.experiment.log_metric("meteor_score", meteor_score)
                    dict_bleu_meteor = {"round": {"round": round, "origin": origin, "reference": reference, "hypothesis": hypothesis,
                                                  "bleu_score": sentence_bleu_score, "meteor_score": meteor_score}}
                    self.experiment.log_others(dict_bleu_meteor)

    def _tokenize(self, x):
        tokenized = self.predictor.tokenizer.tokenize(x)
        prev = None
        merged = []
        for i in tokenized:
            if prev is None:
                prev = i
            elif i.startswith("##"):
                prev = prev+i.replace("##", "")
            else:
                merged.append(prev)
                prev = i
        merged.append(prev)
        return merged

    def evaluate_dataset(self, round):
        tokenize = lambda x: self._tokenize(x)

        hypotheses = []
        origins = []
        references = []

        step = 0
        desc = '  - (Evaluation)   '

        round = round + 1

        for batch in tqdm(self.test_iterator, mininterval=2, desc=desc, leave=False):
            step = step + 1

            sources = batch.src
            sources = sources.to(self.device)
            targets = batch.trg
            targets = targets.to(self.device)

            for i, source in enumerate(sources):
                target = targets[i]
                prediction = self.predictor.predict_one(source, only_one=True, tokenize=False)
                hypotheses.append(prediction)
                source = self.predictor.decode_predicted_string(source.tolist()).replace("[PAD]", "").strip()
                origins.append(source)
                target = self.predictor.decode_predicted_string(target.tolist()).replace("[PAD]", "").strip()
                references.append(target)

        tokenized_hypotheses = [tokenize(prediction) for prediction in hypotheses]
        tokenized_origins = [tokenize(origin) for origin in origins]
        tokenized_references = [tokenize(reference) for reference in references]

        self._evaluate_sentence_bleu_and_meteor(origins, tokenized_origins, references, tokenized_references, hypotheses,
                                                tokenized_hypotheses, round)

        list_of_tokenized_references = [[reference] for reference in tokenized_references]

        bleu_score_nltk = corpus_bleu(list_of_tokenized_references, tokenized_hypotheses,
                                      smoothing_function=self.smoothing_function.method3)
        bleu_score_local = self.bleu.evaluate_results(tokenized_hypotheses, list_of_tokenized_references)

        sari_score = self.sari.evaluate_results(tokenized_origins, tokenized_hypotheses, tokenized_references)

        if self.experiment is not None:
            self.experiment.log_metric("bleu_score_nltk", bleu_score_nltk)
            self.experiment.log_metric("bleu_score", bleu_score_local)
            self.experiment.log_metric("sari_score", sari_score)

        return bleu_score_nltk, sari_score
