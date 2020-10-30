import torch
from torch import nn


class TokenCrossEntropyLoss(nn.Module):

    def __init__(self, pad_index=1):
        super(TokenCrossEntropyLoss, self).__init__()

        self.pad_index = pad_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    def forward(self, outputs, targets):
        """
        outputs (FloatTensor): (batch_size, seq_len, vocabulary_size)
        targets (LongTensor): (batch_size, seq_len)
        """
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs = outputs.contiguous().view(-1, vocabulary_size)

        batch_loss = self.criterion(outputs, targets)

        count = (targets != self.pad_index).sum().item()

        return batch_loss, count


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, vocabulary_size, pad_index=1):
        assert 0.0 < label_smoothing <= 1.0

        super(LabelSmoothingLoss, self).__init__()

        self.pad_index = pad_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')

        smoothing_value = label_smoothing / (vocabulary_size - 2)  # exclude pad and true label
        smoothed_targets = torch.full((vocabulary_size,), smoothing_value)
        smoothed_targets[self.pad_index] = 0
        self.register_buffer('smoothed_targets', smoothed_targets.unsqueeze(0))  # (1, vocabulary_size)

        self.confidence = 1.0 - label_smoothing

    def forward(self, outputs, targets):
        """
        outputs (FloatTensor): (batch_size, seq_len, vocabulary_size)
        targets (LongTensor): (batch_size, seq_len)
        """
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_log_softmax = self.log_softmax(outputs)
        outputs_flat = outputs_log_softmax.contiguous().view(-1, vocabulary_size)

        smoothed_targets = self.smoothed_targets.repeat(targets.size(0), 1)
        smoothed_targets = smoothed_targets
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)
        smoothed_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)
        smoothed_targets.masked_fill_((targets == self.pad_index).unsqueeze(1), 0)
        # masked_targets: (batch_size * seq_len, vocabulary_size)

        loss = self.criterion(outputs_flat, smoothed_targets)

        count = (targets != self.pad_index).sum().item()

        return loss, count
