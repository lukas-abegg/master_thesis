import os
import json
import codecs

import pandas as pd
import numpy as np


def load_training_config(bert_model):
    with open(os.path.join("../", "configs", "split_sets_wiki_simple.json"), "r") as config_file:
        config = json.load(config_file)
    return config[bert_model]


def read_file_dataset(data_path):
    print("Reading lines...")
    # Read the file and split into lines
    data = pd.read_csv(data_path, delimiter='\t', header=0, encoding='utf-8')
    return data.drop_duplicates()


def transform_to_numpy(data_health, data_wiki):
    data_health_np = data_health.to_numpy()
    data_wiki_np = data_wiki.to_numpy()

    data_health_size = len(data_health_np)
    data_health_indices = list(range(data_health_size))
    return data_health_indices, data_health_np, data_wiki_np


def split_train_valid_test_set(dataset_health, dataset_wiki, indices, test_split, validation_split):

    test_indices, train_indices = indices[:test_split], indices[test_split:]
    validation_indices, train_indices = train_indices[:validation_split], train_indices[validation_split:]

    train = np.concatenate([dataset_health[train_indices], dataset_wiki])
    validation = dataset_health[validation_indices]
    test = dataset_health[test_indices]

    return train, validation, test


def write_to_file(data, datafile):
    print("Writing file: {} with {} lines.".format(datafile, len(data)))
    data.to_csv(datafile + ".src", encoding='utf-8', columns=["input"], index=False, header=False)
    data.to_csv(datafile + ".dst", encoding='utf-8', columns=["output"], index=False, header=False)


def write_to_files(train, validation, test, datapath_train, datapath_validation, datapath_test):
    columns = ["input", "output"]
    train_df = pd.DataFrame.from_records(train, columns=columns)
    validation_df = pd.DataFrame.from_records(validation, columns=columns)
    test_df = pd.DataFrame.from_records(test, columns=columns)

    write_to_file(train_df, datapath_train)
    write_to_file(validation_df, datapath_validation)
    write_to_file(test_df, datapath_test)


if __name__ == "__main__":
    bert_model = "bio_bert"
    training_config = load_training_config(bert_model)

    data_health = read_file_dataset(os.path.join(training_config["data_path"], training_config["dataset_health"]))
    data_wiki = read_file_dataset(os.path.join(training_config["data_path"], training_config["dataset_wiki"]))
    data_wiki_without_health = data_wiki[
        np.logical_not(data_wiki.input.isin(data_health.input) & data_wiki.output.isin(data_health.output))]

    shuffle = bool(training_config["shuffle"])
    validation_split = int(training_config["validation_split"])
    test_split = int(training_config["test_split"])
    training_split = len(data_health) - validation_split - test_split

    print("Transform data...")
    data_health_indices, data_health_np, data_wiki_without_health_np = transform_to_numpy(data_health, data_wiki_without_health)

    if shuffle:
        print("Shuffle indices...")
        np.random.shuffle(data_health_indices)

    print("Split data...")
    train, validation, test = split_train_valid_test_set(data_health_np, data_wiki_without_health_np, data_health_indices, test_split, validation_split)

    print("Write data to files")
    dataset_train = str(training_config["dataset_train"])
    dataset_validation = str(training_config["dataset_validation"])
    dataset_test = str(training_config["dataset_test"])
    write_to_files(train, validation, test, dataset_train, dataset_validation, dataset_test)
