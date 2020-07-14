import os
import json
import codecs

import pandas as pd
import numpy as np


def load_training_config(bert_model):
    with open(os.path.join("../", "configs", "split_sets_newsela.json"), "r") as config_file:
        config = json.load(config_file)
    return config[bert_model]


def read_file_dataset(data_path):
    print("Reading lines...")
    # Read the file and split into lines
    data = pd.read_csv(data_path, delimiter="\t", header=None, encoding="utf-8", names=["doc_id", "docv_input", "docv_output", "input", "output"])
    print("unfiltered sentence pairs: ", str(data.size))
    data = data[~(data.input == "[CLS] nan [SEP]") & ~(data.output == "[CLS] nan [SEP]") & ~(data.input == "[CLS] [SEP]") & ~(data.output == "[CLS] [SEP]")]
    data.drop_duplicates()
    print("filtered sentence pairs: ", str(data.size))
    return data


def split_data(df_newsela):
    df_docs = df_newsela.doc_id
    df_docs = df_docs.drop_duplicates()

    df_docs_train = df_docs.head(1070)
    df_docs_valid = df_docs[~df_docs.isin(df_docs_train)]
    df_docs_valid = df_docs_valid.head(30)
    df_docs_test = df_docs[~df_docs.isin(df_docs_train) & ~df_docs.isin(df_docs_valid)]

    df_newsela_train = df_newsela[df_newsela.doc_id.isin(df_docs_train)]

    df_newsela_valid = df_newsela[df_newsela.doc_id.isin(df_docs_valid)]
    df_newsela_valid = df_newsela_valid.groupby(['doc_id', 'docv_input', 'input']).last().reset_index()

    df_newsela_test = df_newsela[df_newsela.doc_id.isin(df_docs_test)]
    df_newsela_test = df_newsela_test.groupby(['doc_id', 'docv_input', 'input']).last().reset_index()

    return df_newsela_train, df_newsela_valid, df_newsela_test


def write_to_file(data, datafile):
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    print("Writing file: {} with {} lines.".format(datafile, len(data)))
    data.to_csv(datafile, sep=delimiter, encoding='utf-8', index=False)


def write_to_files(train, validation, test, datapath_train, datapath_validation, datapath_test):
    columns = ["input", "output"]

    write_to_file(train, datapath_train)
    write_to_file(validation, datapath_validation)
    write_to_file(test, datapath_test)


if __name__ == "__main__":
    bert_model = "bio_bert"
    #bert_model = "bert_base"
    training_config = load_training_config(bert_model)

    data = read_file_dataset(os.path.join(training_config["data_path"], training_config["dataset"]))

    print("Split data...")
    train, validation, test = split_data(data)

    print("Write data to files")
    dataset_train = str(training_config["dataset_train"])
    dataset_validation = str(training_config["dataset_validation"])
    dataset_test = str(training_config["dataset_test"])
    write_to_files(train, validation, test, dataset_train, dataset_validation, dataset_test)
