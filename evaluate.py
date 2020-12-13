import os
import sys
import traceback
from datetime import datetime
from os.path import dirname, abspath

import torch

from evaluation.evaluators.evaluator import Evaluator
from evaluation.predictors.predictor_simple_beam import Predictor
from evaluation.predictors.predictor_advanced_beam import Predictor as Predictor_with_advanced_beam
from evaluation.predictors.predictor_greedy import Predictor as Predictor_greedy
from layers.utils.bert_loader import BertModelLoader

from configs.load_config import load_evaluation_config, load_dataset_config, \
    load_hyperparameter_config_val
from evaluation.evaluation_utils import init_logger, load_data
from build_models.choose_model import build_model
from tracking import load_tracking, stop_tracking


def load_predictor(base_dir_load, model, beam_approach, evaluation_config, dataset_config, hyperparameter_config, tokenizer, device, SRC, TRG):
    checkpoint_filepath = os.path.join(base_dir_load, evaluation_config["model_dir"], evaluation_config["model_file"])
    max_length = dataset_config[hyperparameter_config["bert_model"]]["max_seq_length"]
    beam_size = evaluation_config["beam_size"]
    num_candidates = evaluation_config['num_candidates']

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


def execute_evaluations(evaluator, evaluation_config, logger, experiment=None):
    num_evals = evaluation_config['num_evals']

    bleu_scores = []
    sari_scores = []

    for i in range(num_evals):
        print("Evaluation Round: {}".format(i))
        bleu_score, sari_score = evaluator.evaluate_dataset(i)

        bleu_scores.append(bleu_score)
        sari_scores.append(sari_score)

    best_bleu_scores, _ = torch.as_tensor(bleu_scores).topk(1)
    best_sari_scores, _ = torch.as_tensor(sari_scores).topk(1)

    print("Best Bleu Score: {}".format(best_bleu_scores.item()))
    print("Best Sari Score: {}".format(best_sari_scores.item()))

    if experiment is not None:
        experiment.log_metric("best_bleu", best_bleu_scores.item())
        experiment.log_metric("best_sari", best_sari_scores.item())


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Found ", torch.cuda.device_count(), " GPU devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device ", device, " for task")

    torch.backends.cudnn.benchmark = True

    # Load Config
    BASE_DIR = dirname(abspath(__file__))
    evaluation_config = load_evaluation_config(BASE_DIR)

    # Base dir for Loading
    if evaluation_config["mode"] == "local":
        base_dir_load = os.path.join(BASE_DIR, evaluation_config['base_dir_load'])
    else:
        base_dir_load = os.path.join(evaluation_config['base_dir_load'])

    hyperparameter_config = load_hyperparameter_config_val(base_dir_load, evaluation_config["load_config_file"])
    bert_model_name = hyperparameter_config["bert_model"]
    dataset_name = hyperparameter_config["dataset"]
    dataset_config = load_dataset_config(BASE_DIR, dataset_name)

    # Start Tracking
    tracking_active = evaluation_config["tracking_active"]
    experiment = None

    if tracking_active:
        experiment = load_tracking(evaluation_config)

    # Load Logger
    logger, run_name = init_logger(evaluation_config, hyperparameter_config)

    # Load Bert
    logger.info('Loading bert...')
    bert_model_loader = BertModelLoader(bert_model_name, BASE_DIR, base_dir_load)
    tokenizer = bert_model_loader.tokenizer

    # Load Dataset
    logger.info('Loading dataset...')
    test_iterator, length_set, SRC, TRG = load_data(base_dir_load, dataset_name, dataset_config, hyperparameter_config,
                                                    tokenizer)

    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TRG.vocab)

    # Load Transformer
    logger.info('Loading transformer...')
    model = build_model(hyperparameter_config, dataset_config, bert_model_loader.model, source_vocab_length, target_vocab_length, SRC, TRG, tokenizer, device)

    # Load Predictor
    try:
        predictor = load_predictor(base_dir_load, model, evaluation_config['beam_approach'], evaluation_config, dataset_config,
                                   hyperparameter_config, tokenizer, device)
    except Exception:
        logger.error("Unexpected error by loading predictor:", traceback.format_exc())
        stop_tracking(experiment)
        raise

    timestamp = datetime.now()
    eval_filepath = '{config}-time={timestamp}.csv'.format(
        config=evaluation_config['save_metrics'],
        timestamp=timestamp.strftime("%Y_%m_%d_%H_%M_%S"))

    eval_filepath = os.path.join(BASE_DIR, eval_filepath)

    # Load Evaluator
    try:
        evaluator = Evaluator(
            predictor=predictor,
            save_filepath=eval_filepath,
            test_iterator=test_iterator,
            logger=logger,
            config=evaluation_config,
            device=device,
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
                execute_evaluations(evaluator, evaluation_config, logger, experiment)
        else:
            execute_evaluations(evaluator, evaluation_config, logger)

    except Exception as e:
        logger.error("Unexpected error by running evaluation:", traceback.format_exc())
        stop_tracking(experiment)
        raise

    # Stop Training
    stop_tracking(experiment)

    sys.exit()
