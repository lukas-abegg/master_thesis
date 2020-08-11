import os
import json

""" Configs """


def load_hyperparameter_config(base_dir):
    with open(os.path.join(base_dir, "configs", "hyperparameter.json"), "r") as config_file:
        config = json.load(config_file)
    return config


def load_dataset_config(base_dir, dataset):
    if dataset == 'newsela':
        with open(os.path.join(base_dir, "configs", "load_newsela.json"), "r") as config_file:
            config = json.load(config_file)
    else:
        with open(os.path.join(base_dir, "configs", "load_wiki_simple.json"), "r") as config_file:
            config = json.load(config_file)
    return config


def load_evaluation_config(base_dir):
    with open(os.path.join(base_dir, "configs", "evaluation.json"), "r") as config_file:
        config = json.load(config_file)
    return config


def load_prediction_config(base_dir):
    with open(os.path.join(base_dir, "configs", "prediction.json"), "r") as config_file:
        config = json.load(config_file)
    return config

