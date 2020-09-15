from nltk.translate.meteor_score import meteor_score


class MeteorMetric:

    @staticmethod
    def evaluate_results(hypothesis, references):
        """
        We assume the input and output as linear softmax. The prediction will be done here

        :param hypothesis: 1x sentence as array of dim (Px1) where P = Predicted Sentence length
        :param references: sentences as arrays of dim (NxT) where N = Batch Size and T = Target Sentence length
        :return:
        """

        score = meteor_score(hypothesis=hypothesis, references=references)

        return score
