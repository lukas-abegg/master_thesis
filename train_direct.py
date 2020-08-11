import sys
import traceback
from os.path import dirname, abspath

import torch
from torch.optim import AdamW, lr_scheduler

from layers.utils.bert_loader import BertModelLoader
from training.direct_training.trainer import EpochTrainer
from training.losses import TokenCrossEntropyLoss, LabelSmoothingLoss
from metrics.accuracy import AccuracyMetric

from build_models.choose_model import build_model
from configs.load_config import load_hyperparameter_config, load_dataset_config
from training.training_utils import load_data, init_logger
from utils.tracking import load_tracking, stop_tracking


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    torch.backends.cudnn.benchmark = True

    # Load Config
    BASE_DIR = dirname(abspath(__file__))
    hyperparameter_config = load_hyperparameter_config(BASE_DIR)
    bert_model_name = hyperparameter_config["bert_model"]
    dataset_name = hyperparameter_config["dataset"]
    dataset_config = load_dataset_config(BASE_DIR, dataset_name)

    # Start Tracking
    tracking_active = hyperparameter_config['tracking_active']
    experiment = None

    if tracking_active:
        experiment = load_tracking(hyperparameter_config)

    # Load Logger
    logger, run_name = init_logger(hyperparameter_config)

    # Load Bert
    logger.info('Loading bert...')
    bert_model_loader = BertModelLoader(bert_model_name, "")
    tokenizer = bert_model_loader.tokenizer

    # Load Transformer
    logger.info('Loading transformer...')
    model = build_model(hyperparameter_config, dataset_config, bert_model_loader.model)

    # Load Dataset
    logger.info('Loading dataset...')
    train_iterator, valid_iterator, lengths_sets = load_data(dataset_name, dataset_config, hyperparameter_config,
                                                             tokenizer, device)

    # Load Loss, Accuracy, Optimizer
    if hyperparameter_config['loss'] == "tokencrossentropy":
        loss_function = TokenCrossEntropyLoss(pad_index=tokenizer.pad_token_id)
    else:
        loss_function = LabelSmoothingLoss(label_smoothing=hyperparameter_config['label_smoothing'],
                                           vocabulary_size=len(tokenizer.vocab),
                                           pad_index=tokenizer.pad_token_id)

    accuracy_function = AccuracyMetric()
    optimizer = AdamW(model.parameters(), lr=hyperparameter_config['optimizer_lr'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameter_config['scheduler_step_size'],
                                           gamma=hyperparameter_config['scheduler_gamma'])

    # Train Model
    try:
        trainer = EpochTrainer(
            base_dir=BASE_DIR,
            model=model,
            train_iterator=train_iterator,
            val_iterator=valid_iterator,
            lengths_sets=lengths_sets,
            loss_function=loss_function,
            metric_function=accuracy_function,
            optimizer=optimizer,
            scheduler=exp_lr_scheduler,
            logger=logger,
            run_name=run_name,
            save_config=hyperparameter_config['save_config'],
            save_checkpoint=hyperparameter_config['save_checkpoint'],
            config=hyperparameter_config,
            experiment=experiment
        )

        if experiment is not None:
            with experiment.train():
                trainer.run(hyperparameter_config['epochs'])
        else:
            trainer.run(hyperparameter_config['epochs'])

    except Exception as e:
        logger.error("Unexpected error by loading trainer:", traceback.format_exc())
        stop_tracking(experiment)
        raise

    # Stop Training
    stop_tracking(experiment)

    sys.exit()
