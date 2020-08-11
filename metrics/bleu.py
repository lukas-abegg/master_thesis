from torchtext.data.metrics import bleu_score

""" 
Also think about SacreBleu

pip install sacrebleu...
"""


class BleuMetric:

    @staticmethod
    def evaluate_results(predictions, targets):
        """ 
        We assume the input and output as linear softmax. The prediction will be done here
        
        :param predictions: sentences as array of dim (NxP) where N = Batch Size and P = Predicted Sentence length
        :param targets: sentences as arrays of dim (NxT) where N = Batch Size and T = Target Sentence length
        :return: 
        """

        score = bleu_score(candidate_corpus=predictions, references_corpus=targets)

        return score
