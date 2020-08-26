import logging
from functools import lru_cache

import torch
from torch.utils.data import DataLoader

from training.utils.dataset import TextDataset


class DataSplit:
    def __init__(self, train_set, validation_set, test_set, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.train_dataset = self.load_dataset(train_set)
        self.validation_dataset = self.load_dataset(validation_set)
        self.test_dataset = self.load_dataset(test_set)

    def load_dataset(self, data):
        return TextDataset(data.input, data.output, self.tokenizer, self.max_seq_length)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size, num_workers, shuffle):
        logging.debug('Initializing train-validation-test dataloaders')

        self.get_train_loader(batch_size, num_workers, shuffle)
        self.get_validation_loader(batch_size, num_workers, shuffle)
        self.get_test_loader(batch_size, num_workers, shuffle)
        return [self.train_loader, self.validation_loader, self.test_loader]

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size, num_workers, shuffle):
        logging.debug('Initializing train dataloader')

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size, num_workers, shuffle):
        logging.debug('Initializing validation dataloader')

        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return self.validation_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size, num_workers, shuffle):
        logging.debug('Initializing test dataloader')

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return self.test_loader
