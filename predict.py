import os
import sys
import traceback
from os.path import dirname, abspath

import torch

from build_models.choose_model import build_model
from configs.load_config import load_hyperparameter_config, load_dataset_config, \
    load_prediction_config
from evaluation.predictors.predictor_transformer_gru import Predictor as Predictor_with_GRU
from evaluation.predictors.predictor_transformer import Predictor
from layers.utils.bert_loader import BertModelLoader


def load_predictor(model, prediction_config, dataset_config, tokenizer, device):
    if model == "transformer_with_gru":
        predictor = Predictor_with_GRU(
            model=model,
            checkpoint_filepath=os.path.join(BASE_DIR, prediction_config['checkpoint']),
            tokenizer=tokenizer,
            device=device,
            max_length=dataset_config['max_length'],
            beam_size=prediction_config['beam_size']
        )
    else:
        predictor = Predictor(
            model=model,
            checkpoint_filepath=os.path.join(BASE_DIR, prediction_config['checkpoint']),
            tokenizer=tokenizer,
            device=device,
            max_length=dataset_config['max_length'],
            beam_size=prediction_config['beam_size']
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
    #
    torch.backends.cudnn.benchmark = True
    #
    # Load Config
    BASE_DIR = dirname(abspath(__file__))
    prediction_config = load_prediction_config(BASE_DIR)
    hyperparameter_config = load_hyperparameter_config(BASE_DIR)
    bert_model_name = prediction_config["bert_model"]
    dataset_name = prediction_config["dataset"]
    dataset_config = load_dataset_config(BASE_DIR, dataset_name)

    # Load Bert
    print('Loading bert...')
    bert_model_loader = BertModelLoader(bert_model_name, "")
    tokenizer = bert_model_loader.tokenizer

    # Load Transformer
    print('Loading transformer...')
    model = build_model(hyperparameter_config, dataset_config, bert_model_loader.model)

    # Load Data
    origins = load_origins(os.path.join(BASE_DIR, prediction_config['origins_path']))

    # Load Predictor
    try:
        predictor = load_predictor(hyperparameter_config['model'], prediction_config, dataset_config, tokenizer, device)
    except Exception:
        print("Unexpected error by loading predictor:", traceback.format_exc())
        raise

    # Predict
    try:
        for origin in origins:
            print(f'Predict target for origin: {origin}')
            for index, candidate in enumerate(
                    predictor.predict_one(origin, num_candidates=prediction_config['num_candidates'])):
                print(f'Candidate {index} : {candidate}')
    except Exception as e:
        print("Unexpected error by running evaluation:", traceback.format_exc())
        raise

    sys.exit()
