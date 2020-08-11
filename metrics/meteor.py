from nltk.translate.meteor_score import meteor_score


class MeteorMetric:

    @staticmethod
    def evaluate_results(predictions, targets):
        """
        We assume the input and output as linear softmax. The prediction will be done here

        :param predictions: sentences as array of dim (NxP) where N = Batch Size and P = Predicted Sentence length
        :param targets: sentences as arrays of dim (NxT) where N = Batch Size and T = Target Sentence length
        :return:
        """

        score = meteor_score(hypothesis=predictions, references=targets)

        return score
