from torch import nn


class AccuracyMetric(nn.Module):

    def __init__(self, pad_index=1):
        super(AccuracyMetric, self).__init__()

        self.pad_index = pad_index

    def forward(self, outputs, targets):

        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs = outputs.contiguous().view(-1, vocabulary_size)

        predicts = outputs.argmax(dim=1)
        corrects = predicts == targets

        corrects.masked_fill_((targets == self.pad_index), 0)

        correct_count = corrects.sum().item()

        count = (targets != self.pad_index).sum().item()

        return correct_count, count
