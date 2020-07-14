import datetime
import json
import os
import time
import math

import numpy as np

import torch
from torch.autograd import Variable


def print_training_state(training_time, total_training_loss, total_validation_loss):
    print('| end of training | time: {:} | best train loss {:5.2f} | best valid loss {:8.2f}'.format(format_time(training_time), total_training_loss, total_validation_loss))


def print_validation_state(epoch, validation_time, total_validation_loss):
    print('| end of epoch {:3d} | time: {:} | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, format_time(validation_time), total_validation_loss, math.exp(total_validation_loss)))


def print_batch_state(log_interval, training_start_time, epoch_start_time, batch_start_time, scheduler, epoch, batch_idx, batch_len, train_len, total_loss, total_loss_batch):
    # Calculate elapsed time in minutes.
    elapsed_training = format_time(time.time() - training_start_time)
    elapsed_epoch = format_time(time.time() - epoch_start_time)
    elapsed_batch = format_time(time.time() - batch_start_time)

    progress_batch = (batch_idx + 1) * batch_len
    avg_loss = total_loss / (log_interval * (batch_idx + 1))
    current_loss = total_loss_batch / log_interval

    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | elapsed time for {:} batches: batch interval {} / epoch {} / training {} | loss: current interval {:5.2f} / overall avg {:5.2f} | ppl {:8.2f}'.format(
        log_interval, epoch, progress_batch, train_len, scheduler.get_lr()[0], elapsed_training, elapsed_epoch, elapsed_batch, current_loss, avg_loss, math.exp(current_loss)
    ))


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_config():
    with open(os.path.join("configs", "hyperparameter.json"), "r") as config_file:
        config = json.load(config_file)
    return config


def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if device == 0:
        np_mask = np_mask.cuda()
    return np_mask


def create_masks(src, trg, device):
    src_mask = (src != "[PAD]").unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != "[PAD]").unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, device)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask
