import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from preprocessings.bert_tokenizing import BertPreprocessor


class TextDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_len=512):

        self.X = inputs
        self.y = outputs

        self.tokenizer: BertPreprocessor = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __encode_sentence(self, sentence):

        #tokenized, padded, attention_mask = self.tokenizer.encode_sentence(sentence, self.max_len)
        padded, attention_mask = self.tokenizer.encode_sentence(sentence, self.max_len)

        if len(padded) > self.max_len:
            #tokenized = padded[:self.max_len]
            padded = padded[:self.max_len]
            attention_mask = attention_mask[:self.max_len]

        return padded, attention_mask

    def __getitem__(self, index):

        X = self.__encode_sentence(self.X[index])
        y = self.__encode_sentence(self.y[index])

        return X, y
