import os
import json

import pandas as pd

from preprocessings.bert_tokenizing import BertPreprocessor
from preprocessings.models.datasplit import DataSplit
from preprocessings.models.vocab import Vocab


def load_training_config(bert_model):
    with open(os.path.join("configs", "load_newsela.json"), "r") as config_file:
        config = json.load(config_file)
    return config[bert_model]


def read_file_dataset(data_path):
    print("Reading lines for {}...".format(data_path))
    # Read the file and split into lines
    data = pd.read_csv(data_path, delimiter="\t", header=0, encoding="utf-8")
    return data.drop_duplicates()[:5]


def build_trg_vocab(tokenizer, train_set, validation_set, test_set, config):
    split_words = bool(config["split_words"])

    vocab = Vocab(tokenizer, split_words)

    for sentence in train_set.output:
        vocab.add_sentence(sentence)

    for sentence in validation_set.output:
        vocab.add_sentence(sentence)

    for sentence in test_set.output:
        vocab.add_sentence(sentence)

    return vocab


def load_dataset(bert_model_type, loaded_bert_model):
    training_config = load_training_config(bert_model_type)

    train_set_path = str(training_config["dataset_train"])
    train_set = read_file_dataset(train_set_path)

    validation_set_path = str(training_config["dataset_validation"])
    validation_set = read_file_dataset(validation_set_path)

    test_set_path = str(training_config["dataset_test"])
    test_set = read_file_dataset(test_set_path)

    tokenizer = BertPreprocessor(loaded_bert_model.tokenizer)
    max_seq_length = int(training_config["max_seq_length"])

    trg_vocab = build_trg_vocab(tokenizer, train_set, validation_set, test_set, training_config)

    splits = DataSplit(train_set, validation_set, test_set, tokenizer=tokenizer, max_seq_length=max_seq_length)

    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])
    shuffle = bool(training_config["shuffle"])

    return splits.get_split(batch_size=batch_size, num_workers=num_workers, shuffle=shuffle), trg_vocab
