import sys

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


class DatasetProcessing(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.data_size = int(data['labels'].size(0))
        self.maxlen = max_len

        assert self.data['src'].size(0) == self.data['trg'].size(0) \
               and self.data['trg'].size(0) == self.data['labels'].size(0)

    def __getitem__(self, index):
        assert index < self.data_size

        source = self.data['src'][index].long()
        target = self.data['trg'][index].long()
        labels = self.data['labels'][index].long()

        return {
            'source': source,
            'target': target,
            'labels': labels
        }

    def __len__(self):
        return self.data_size

    def collater(self, samples):
        return DatasetProcessing.collate(samples, self.maxlen)

    @staticmethod
    def collate(samples, maxlen):
        if len(samples) == 0:
            return {}

        def merge(key):
            return DatasetProcessing.collate_tokens([s[key] for s in samples], maxlen)

        labels = torch.LongTensor([s['labels'] for s in samples])
        src_tokens = merge('source')
        target = merge('target')

        return {
            'src_tokens': src_tokens,
            'trg_tokens': target,
            'labels': labels
        }

    @staticmethod
    def collate_tokens(values, maxlen):
        max_input_size = max(v.size(0) for v in values)
        assert max_input_size == maxlen

        res = torch.stack(values, dim=0)

        return res


def train_dataloader(dataset, batch_size=32, sample_without_replacement=0, sort_by_source_size=False):

    batch_sampler = shuffled_batches_by_size(len(dataset), batch_size=batch_size,
                                             sample=sample_without_replacement,
                                             sort_by_source_size=sort_by_source_size)

    return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collater, batch_sampler=batch_sampler)


def eval_dataloader(dataset, batch_size=32):

    batch_sampler = batches_by_order(len(dataset), batch_size)

    return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collater, batch_sampler=batch_sampler)


def _make_batches(indices, batch_size):
    batch = []

    for idx in map(int, indices):
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(idx)

    if len(batch) > 0:
        yield batch


def batches_by_order(data_size, batch_size=32):
    """Returns batches of indices sorted by size. Sequences with different
    source lengths are not allowed in the same batch."""

    indices = np.arange(data_size)

    return list(_make_batches(indices, batch_size))


def shuffled_batches_by_size(data_size, batch_size=32, sample=0, sort_by_source_size=False):
    """Returns batches of indices, bucketed by size and then shuffled. Batches
    may contain sequences of different lengths."""
    if sample:
        indices = np.random.choice(data_size, sample, replace=False)
    else:
        indices = np.random.permutation(data_size)

    batches = list(_make_batches(indices, batch_size))

    if not sort_by_source_size:
        np.random.shuffle(batches)

    return batches


def prepare_training_data(data_iter, generator, tgt_vocab, bos_word, max_len, eos_word, blank_word, use_gpu):

    src_data_temp = []
    trg_data_temp = []
    labels_temp = []
    print("preparing discriminator data.")

    generator.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            sys.stdout.write('\r' + 'Finishing ' + str(i + 1) + '/' + str(len(data_iter)))
            sys.stdout.flush()

            # a tensor with max possible translation length
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg

            # change to shape (bs , max_seq_len)
            src = src.transpose(0, 1)

            # change to shape (bs , max_seq_len)
            trg = trg.transpose(0, 1)

            neg_tokens = []
            for origin_sentence in src:
                neg_tokens.append(greedy_decode_sentence(generator, origin_sentence, tgt_vocab, max_len, bos_word, eos_word, blank_word, use_gpu))

            neg_tokens = torch.stack(neg_tokens)
            pos_tokens = trg
            src_tokens = src

            assert neg_tokens.size() == pos_tokens.size()
            assert src_tokens.size() == pos_tokens.size()

            src_data_temp.append(src_tokens)
            trg_data_temp.append(pos_tokens)
            labels_temp.extend([1] * int(pos_tokens.size(0)))

            src_data_temp.append(src_tokens)
            trg_data_temp.append(neg_tokens)
            labels_temp.extend([0] * int(neg_tokens.size(0)))

        src_data_temp = torch.cat(src_data_temp, dim=0)
        trg_data_temp = torch.cat(trg_data_temp, dim=0)
        src_data_temp = src_data_temp.cpu().int()
        trg_data_temp = trg_data_temp.cpu().int()

        labels_temp = np.asarray(labels_temp)
        labels = torch.from_numpy(labels_temp)
        labels = labels.cpu()

        data = {'src': src_data_temp, 'trg': trg_data_temp, 'labels': labels}

        print('\n' + "preparing discriminator data done!")

    return data


def greedy_decode_sentence(generator, origin_sentence, tgt_vocab, max_len, bos_word, eos_word, blank_word, use_gpu=False):
    sentence_tensor = torch.unsqueeze(origin_sentence, 0)

    trg_init_tok = tgt_vocab.stoi[bos_word]
    trg = torch.LongTensor([[trg_init_tok]])

    if use_gpu:
        sentence_tensor = sentence_tensor.cuda()
        trg = trg.cuda()

    for i in range(max_len):
        size = trg.size(0)

        np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.cuda() if use_gpu else np_mask

        pred = generator(sentence_tensor.transpose(0, 1), trg, tgt_mask=np_mask)
        pred = F.softmax(pred)
        add_word = tgt_vocab.itos[pred.argmax(dim=2)[-1]]

        if use_gpu:
            trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
        else:
            trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]])))

        if add_word == eos_word:
            break

    trg = torch.nn.functional.pad(trg.transpose(0, 1).squeeze(0), (0, max_len - (len(trg))), value=tgt_vocab.stoi[blank_word])
    return trg