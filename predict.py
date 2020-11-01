import os
import sys
import traceback
from os.path import dirname, abspath

import torch

from build_models.choose_model import build_model
from configs.load_config import load_dataset_config, \
    load_prediction_config, load_hyperparameter_config_val
from evaluation.predictors.predictor_advanced_beam import Predictor as Predictor_with_advanced_beam
from evaluation.predictors.predictor_simple_beam import Predictor
from evaluation.predictors.predictor_greedy import Predictor as Predictor_greedy
from layers.utils.bert_loader import BertModelLoader


def load_predictor(model, beam_approach, prediction_config, dataset_config, hyperparameter_config, tokenizer, device):
    checkpoint_filepath = os.path.join(BASE_DIR, prediction_config["model_dir"], prediction_config["model_file"])
    max_length = dataset_config[hyperparameter_config["bert_model"]]["max_seq_length"]
    beam_size = prediction_config["beam_size"]
    num_candidates = prediction_config['num_candidates']

    if beam_approach == "advanced":
        predictor = Predictor_with_advanced_beam(
            model=model,
            checkpoint_filepath=checkpoint_filepath,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            beam_size=beam_size,
            num_candidates=num_candidates
        )
    elif beam_approach == "simple":
        predictor = Predictor(
            model=model,
            checkpoint_filepath=checkpoint_filepath,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            beam_size=beam_size,
            num_candidates=num_candidates
        )
    else:
        predictor = Predictor_greedy(
            model=model,
            checkpoint_filepath=checkpoint_filepath,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length
        )
    return predictor


def load_origins(filepath):
    # Open a file: file
    origins = []
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            origins.append(line.strip())
            line = fp.readline()
            cnt += 1
    print(cnt, ' lines read from', filepath)
    return origins


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    torch.backends.cudnn.benchmark = True

    # Load Config
    BASE_DIR = dirname(abspath(__file__))
    prediction_config = load_prediction_config(BASE_DIR)
    hyperparameter_config = load_hyperparameter_config_val(BASE_DIR, prediction_config["load_config_file"])
    bert_model_name = hyperparameter_config["bert_model"]
    dataset_name = hyperparameter_config["dataset"]
    dataset_config = load_dataset_config(BASE_DIR, dataset_name)

    # Base dir for Loading
    base_dir_load = os.path.join(BASE_DIR, prediction_config['base_dir_load'])

    # Load Bert
    print('Loading bert...')
    bert_model_loader = BertModelLoader(bert_model_name, BASE_DIR, base_dir_load)
    tokenizer = bert_model_loader.tokenizer

    # Load Data
    origins = load_origins(os.path.join(BASE_DIR, prediction_config['origins_path']))

    source_vocab_length = prediction_config['source_vocab_length']
    target_vocab_length = prediction_config['target_vocab_length']

    SRC = None
    TRG = None

    # Load Transformer
    print('Loading transformer...')
    model = build_model(hyperparameter_config, dataset_config, bert_model_loader.model, source_vocab_length, target_vocab_length, SRC, TRG, tokenizer, device)

    # Load Predictor
    try:
        predictor = load_predictor(model, prediction_config['beam_approach'], prediction_config, dataset_config,
                                   hyperparameter_config, tokenizer, device)
    except Exception:
        print("Unexpected error by loading predictor:", traceback.format_exc())
        raise

    # Predict
    try:
        for origin in origins:
            print(f'Predict target for origin: {origin}')
            predictions = predictor.predict_one(origin)
            for index, candidate in enumerate(predictions):
                print(f'Candidate {index + 1} : {candidate}')
    except Exception as e:
        print("Unexpected error by running evaluation:", traceback.format_exc())
        raise

    sys.exit()
