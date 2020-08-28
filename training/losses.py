import torch
from torch import nn


class TokenCrossEntropyLoss(nn.Module):

    def __init__(self, pad_index=0):
        super(TokenCrossEntropyLoss, self).__init__()

        self.pad_index = pad_index
        self.base_loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_index)

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_flat = outputs.view(batch_size * seq_len, vocabulary_size)
        print("outputs_flat", outputs_flat.shape)
        targets_flat = targets.view(batch_size * seq_len)
        print("targets_flat", targets_flat.shape)

        batch_loss = self.base_loss_function(outputs_flat, targets_flat)
        print("batch_loss", batch_loss)

        count = (targets != self.pad_index).sum().item()
        print("count", count)

        return batch_loss, count


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, vocabulary_size, pad_index=0):
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

        print("log_softmax")
        outputs_log_softmax = self.log_softmax(outputs)
        print("outputs_log_softmax")
        outputs_flat = outputs_log_softmax.view(batch_size * seq_len, vocabulary_size)
        print("targets_flat")
        targets_flat = targets.view(batch_size * seq_len)

        print('\ncurrent memory allocated after (sources, targets): {}'.format(
            torch.cuda.memory_allocated() / 1024 ** 2))
        print('max memory allocated (sources, targets): {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
        print('cached memory (sources, targets): {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
        print('total_memory of device: {}'.format(
            torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 2))

        smoothed_targets = self.smoothed_targets.repeat(targets_flat.size(0), 1)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)
        print("smoothed_targets")
        smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), self.confidence)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)
        print("smoothed_targets scatter_")
        smoothed_targets.masked_fill_((targets_flat == self.pad_index).unsqueeze(1), 0)
        # masked_targets: (batch_size * seq_len, vocabulary_size)
        print("smoothed_targets masked_fill_")

        print('\ncurrent memory allocated after (sources, targets): {}'.format(
            torch.cuda.memory_allocated() / 1024 ** 2))
        print('max memory allocated (sources, targets): {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
        print('cached memory (sources, targets): {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
        print('total_memory of device: {}'.format(
            torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 2))

        loss = self.criterion(outputs_flat, smoothed_targets)
        print("smoothed_targets loss")
        count = (targets != self.pad_index).sum().item()

        return loss, count
