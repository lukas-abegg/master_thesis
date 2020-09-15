from torchtext.data.metrics import bleu_score

""" 
Also think about SacreBleu

pip install sacrebleu...
"""


class BleuMetric:

    @staticmethod
    def evaluate_results(hypotheses, list_of_references):
        """ 
        We assume the input and output as linear softmax. The prediction will be done here
        
        :param hypotheses: sentences as array of dim (NxP) where N = Batch Size and P = Predicted Sentence length
        :param list_of_references: sentences as arrays of dim (Nx1xT) where N = Batch Size and T = Target Sentence length
        :return: 
        """

        score = bleu_score(candidate_corpus=hypotheses, references_corpus=list_of_references)

        return score
