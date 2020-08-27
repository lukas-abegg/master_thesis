import os
import sys
import traceback
from datetime import datetime
from os.path import dirname, abspath

import torch

from evaluation.evaluator import Evaluator
from evaluation.predictors.predictor_simple_beam import Predictor
from evaluation.predictors.predictor_advanced_beam import Predictor as Predictor_with_advanced_beam
from layers.utils.bert_loader import BertModelLoader

from configs.load_config import load_evaluation_config, load_dataset_config, load_hyperparameter_config
from evaluation.evaluation_utils import init_logger, load_data
from build_models.choose_model import build_model
from utils.tracking import load_tracking, stop_tracking


def load_predictor(beam_approach, evaluation_config, dataset_config, tokenizer, device):
    if beam_approach == "advanced_beam":
        predictor = Predictor_with_advanced_beam(
            model=model,
            checkpoint_filepath=os.path.join(BASE_DIR, evaluation_config['checkpoint']),
            tokenizer=tokenizer,
            device=device,
            max_length=dataset_config['max_length'],
            beam_size=evaluation_config['beam_size']
        )
    else:
        predictor = Predictor(
            model=model,
            checkpoint_filepath=os.path.join(BASE_DIR, evaluation_config['checkpoint']),
            tokenizer=tokenizer,
            device=device,
            max_length=dataset_config['max_length'],
            beam_size=evaluation_config['beam_size']
        )
    return predictor


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    torch.backends.cudnn.benchmark = True

    # Load Config
    BASE_DIR = dirname(abspath(__file__))
    evaluation_config = load_evaluation_config(BASE_DIR)
    hyperparameter_config = load_hyperparameter_config(BASE_DIR)
    bert_model_name = evaluation_config["bert_model"]
    dataset_name = evaluation_config["dataset"]
    dataset_config = load_dataset_config(BASE_DIR, dataset_name)

    # Start Tracking
    tracking_active = evaluation_config["tracking_active"]
    experiment = None

    if tracking_active:
        experiment = load_tracking(evaluation_config)

    # Load Logger
    logger, run_name = init_logger(evaluation_config)

    # Save base dir
    base_dir_save = os.path.join(BASE_DIR, hyperparameter_config['save_base_dir'])

    # Load Bert
    logger.info('Loading bert...')
    bert_model_loader = BertModelLoader(bert_model_name, BASE_DIR, base_dir_save)
    tokenizer = bert_model_loader.tokenizer

    # Load Transformer
    logger.info('Loading transformer...')
    model = build_model(hyperparameter_config, dataset_config, bert_model_loader.model)

    # Load Dataset
    logger.info('Loading dataset...')
    test_iterator, length_set = load_data(dataset_name, dataset_config, evaluation_config,
                                          tokenizer, device)

    # Load Predictor
    try:
        predictor = load_predictor(hyperparameter_config['beam_approach'], evaluation_config, dataset_config, tokenizer, device)
    except Exception:
        logger.error("Unexpected error by loading predictor:", traceback.format_exc())
        stop_tracking(experiment)
        raise

    timestamp = datetime.now()
    eval_filepath = 'logs/eval-{config}-time={timestamp}.csv'.format(
        config=evaluation_config,
        timestamp=timestamp.strftime("%Y_%m_%d_%H_%M_%S"))

    eval_filepath = os.path.join(BASE_DIR, eval_filepath),

    # Load Evaluator
    try:
        evaluator = Evaluator(
            predictor=predictor,
            save_filepath=eval_filepath,
            test_iterator=test_iterator,
            logger=logger,
            config=evaluation_config,
            experiment=experiment
        )
    except Exception as e:
        logger.error("Unexpected error by loading evaluator:", traceback.format_exc())
        stop_tracking(experiment)
        raise

    # Evaluate
    try:
        if experiment is not None:
            with experiment.validate():
                bleu_score, sari_score, meteor_score = evaluator.evaluate_dataset()
        else:
            bleu_score, sari_score, meteor_score = evaluator.evaluate_dataset()

    except Exception as e:
        logger.error("Unexpected error by running evaluation:", traceback.format_exc())
        stop_tracking(experiment)
        raise

    # Stop Training
    stop_tracking(experiment)

    sys.exit()
