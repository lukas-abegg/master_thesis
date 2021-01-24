import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class DatasetProcessing(Dataset):
    def __init__(self, data, max_len_src, max_len_tgt):
        self.data = data
        self.data_size = int(data['labels'].size(0))
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt

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
        return DatasetProcessing.collate(samples, self.max_len_src, self.max_len_tgt)

    @staticmethod
    def collate(samples, max_len_src, max_len_tgt):
        if len(samples) == 0:
            return {}

        def merge(key, max_len):
            return DatasetProcessing.collate_tokens([s[key] for s in samples], max_len)

        labels = torch.LongTensor([s['labels'] for s in samples])
        src_tokens = merge('source', max_len_src)
        target = merge('target', max_len_tgt)

        return {
            'src_tokens': src_tokens,
            'trg_tokens': target,
            'labels': labels
        }

    @staticmethod
    def collate_tokens(values, max_len):
        max_input_size = max(v.size(0) for v in values)
        assert max_input_size == max_len

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


def prepare_training_data(data_iter, generator, tokenizer, beam_size, max_len_trg, use_gpu, device, max_batch=800):
    src_data_temp = []
    trg_data_temp = []
    labels_temp = []

    generator.eval()
    with torch.no_grad():
        i = 0
        desc = '  - (Generate Samples)   '
        for batch in tqdm(data_iter, desc=desc, leave=False):
            i = i + 1

            if i > max_batch:
                break

            # a tensor with max possible translation length
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg

            # change to shape (bs , max_seq_len)
            src = src.transpose(0, 1)

            # change to shape (bs , max_seq_len)
            trg = trg.transpose(0, 1)

            neg_tokens = []
            for origin_sentence in src:
                neg_tokens.append(
                    greedy_decode_sentence(generator, origin_sentence, tokenizer, beam_size, max_len_trg, device))

            neg_tokens = torch.stack(neg_tokens)
            pos_tokens = trg
            src_tokens = src

            assert neg_tokens.size() == pos_tokens.size()

            src_data_temp.append(src_tokens)
            trg_data_temp.append(pos_tokens)
            labels_temp.extend([1] * int(pos_tokens.size(0)))

            src_data_temp.append(src_tokens)
            trg_data_temp.append(neg_tokens)
            labels_temp.extend([0] * int(neg_tokens.size(0)))

        src_data_temp = torch.cat(src_data_temp, dim=0)
        trg_data_temp = torch.cat(trg_data_temp, dim=0)

        labels_temp = np.asarray(labels_temp)
        labels = torch.from_numpy(labels_temp)

        data = {'src': src_data_temp, 'trg': trg_data_temp, 'labels': labels}

    return data


def greedy_decode_sentence(generator, origin_sentence, tokenizer, beam_size, max_len, device):
    origin_sentence = torch.unsqueeze(origin_sentence, 0)
    origin_sentence = origin_sentence.to(device)

    pred_sent = generator.generate(origin_sentence, num_beams=beam_size, max_length=max_len, early_stopping=True)

    trg = torch.nn.functional.pad(pred_sent.squeeze(0), (0, max_len - pred_sent.size(1)),
                                  value=tokenizer.pad_token_id)
    return trg
