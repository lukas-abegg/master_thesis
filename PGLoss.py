import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from metrics.sari import SARISentenceMetric


class PGLoss(torch.nn.Module):

    def __init__(self, ignore_index=None, size_average=False, reduce=True):
        super(PGLoss, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce

        self.lambda_r = 0.5
        self.lambda_q = 0.5

        self.bleu_smoothing_function = SmoothingFunction()
        self.sari = SARISentenceMetric()

    def forward(self, logprobs, labels, tokenized_origins, tokenized_predictions, tokenized_targets, reward, use_gpu):
        bsz, seqlen, _ = logprobs.size()
        loss = 0
        logprobs = logprobs.clone()

        if self.ignore_index is not None:
            logprobs[:, :, self.ignore_index] = 0

        for i in range(bsz):
            trg_labels = labels[i, :]
            row_idx = torch.LongTensor(range(seqlen))
            if use_gpu:
                row_idx = row_idx.cuda()

            bleu_i = self.calculate_bleu(tokenized_predictions[i], tokenized_targets[i])
            sari_i = self.calculate_sari(tokenized_origins[i], tokenized_predictions[i], tokenized_targets[i])
            reward_i = reward[i].item()

            loss_q = (self.lambda_q * bleu_i) + ((1 - self.lambda_q) * sari_i)
            loss_r = (self.lambda_r * reward_i) + ((1 - self.lambda_r) * loss_q)

            loss += self.calculate_loss(logprobs[i], row_idx, trg_labels, loss_r)

        if self.size_average:
            loss = loss / bsz

        return loss

    @staticmethod
    def calculate_loss(logprobs_i, row_idx, trg_labels, loss_r):
        trg_log_prob = logprobs_i[:, :][row_idx, trg_labels]
        trg_log_prob *= loss_r
        return -torch.sum(trg_log_prob)

    def calculate_bleu(self, tokenized_prediction_i, tokenized_target_i):
        sentence_bleu_score = sentence_bleu([tokenized_target_i], tokenized_prediction_i, smoothing_function=self.bleu_smoothing_function.method3)
        return sentence_bleu_score

    def calculate_sari(self, tokenized_origin_i, tokenized_prediction_i, tokenized_target_i):
        sari_score = self.sari.evaluate_results([tokenized_origin_i], [tokenized_prediction_i], [tokenized_target_i])
        return sari_score
