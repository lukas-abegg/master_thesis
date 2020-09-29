import torch
import numpy as np
from tqdm import tqdm

from os.path import join, exists
from os import makedirs
from datetime import datetime
import json

from comet_ml import Experiment


class EpochTrainer:

    def __init__(self, device, base_dir, model,
                 train_iterator, val_iterator, lengths_sets,
                 loss_function, metric_function,
                 optimizer, scheduler,
                 logger, run_name,
                 save_config,
                 config, experiment: Experiment = None):

        self.base_dir = base_dir

        self.config = config
        self.device = device
        self.model = model
        self.model.to(self.device)

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.lengths_sets = lengths_sets

        self.loss_function = loss_function.to(self.device)
        self.metric_function = metric_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grads = self.config['clip_grads']

        self.logger = logger
        self.checkpoint_dir = join(self.base_dir, 'checkpoints', run_name)
        self.experiment = experiment

        logger.info("Save checkpoints in {}".format(self.checkpoint_dir))

        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)

        if save_config is None:
            config_filepath = join(self.checkpoint_dir, 'configs.json')
        else:
            config_filepath = join(self.base_dir, save_config)
        with open(config_filepath, 'w') as config_file:
            json.dump(self.config, config_file)

        self.print_every = self.config['print_every']
        self.save_every = self.config['save_every']

        self.epoch = 1
        self.step = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_filepath = None

        self.save_format = 'epoch={epoch:0>3}-val_loss={val_loss:<.3}-val_metrics={val_metrics}.pth'

        self.log_format = (
            "Epoch: {epoch:>3} "
            "Progress: {progress:<.1%} "
            "Elapsed: {elapsed} "
            "Batches/second: {per_second:<.1} "
            "Batches per Epoch: {batches_per_epoch:>3} "
            "Train Loss: {train_loss:<.6} "
            "Val Loss: {val_loss:<.6} "
            "Train Metrics: {train_metrics} "
            "Val Metrics: {val_metrics} "
            "Learning rate: {current_lr:<.4} "
        )

    def run_epoch(self, dataloader, mode='train'):
        batch_losses = []
        batch_counts = []
        batch_metrics = []

        if mode == "train":
            desc = '  - (Training)   '
        else:
            desc = '  - (Validation)   '

        for batch in tqdm(dataloader, mininterval=2, desc=desc, leave=False):
            self.step = self.step + 1

            sources = batch.src
            sources = sources.to(self.device)
            targets = batch.trg
            targets = targets.to(self.device)

            if self.experiment is not None:
                self.experiment.set_step(self.step)

            outputs = self.model(sources, targets)

            batch_loss, batch_count = self.loss_function(outputs, targets)

            if mode == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            batch_loss_item = batch_loss.item()

            batch_losses.append(batch_loss_item)
            batch_counts.append(batch_count)

            batch_metric, batch_metric_count = self.metric_function(outputs, targets)
            batch_metrics.append(batch_metric)

            assert batch_count == batch_metric_count

            if self.experiment is not None:
                batch_loss_to_log = batch_loss_item / batch_count
                self.experiment.log_metric("batch_loss", batch_loss_to_log)
                self.experiment.log_metric("batch_accuracy", batch_metric)

        epoch_loss = sum(batch_losses) / sum(batch_counts)
        epoch_accuracy = sum(batch_metrics) / sum(batch_counts)
        epoch_perplexity = float(np.exp(epoch_loss))
        epoch_metrics = [epoch_perplexity, epoch_accuracy]

        return epoch_loss, epoch_metrics

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            if self.experiment is not None:
                self.experiment.set_epoch(self.epoch)

            """ Start Training for Epoch"""
            self.model.train()

            epoch_start_time = datetime.now()

            train_epoch_loss, train_epoch_metrics = self.run_epoch(self.train_iterator, mode='train')
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = datetime.now()

            """ Start Validation for Epoch"""
            self.model.eval()

            with torch.no_grad():
                val_epoch_loss, val_epoch_metrics = self.run_epoch(self.val_iterator, mode='val')

            """ Log Metrics for Epoch"""
            if self.epoch % self.print_every == 0 and self.logger:
                batches = self.lengths_sets[0]
                per_second = batches / ((epoch_end_time - epoch_start_time).seconds + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                log_message = self.log_format.format(epoch=self.epoch,
                                                     progress=self.epoch / epochs,
                                                     per_second=per_second,
                                                     batches_per_epoch=batches,
                                                     train_loss=train_epoch_loss,
                                                     val_loss=val_epoch_loss,
                                                     train_metrics=[round(metric, 4) for metric in train_epoch_metrics],
                                                     val_metrics=[round(metric, 4) for metric in val_epoch_metrics],
                                                     current_lr=current_lr,
                                                     elapsed=self._elapsed_time()
                                                     )
                if self.experiment is not None:
                    self.experiment.log_metric("epoch_train_loss", train_epoch_loss)
                    self.experiment.log_metric("epoch_val_loss", val_epoch_loss)

                    log_train_epoch_perplexity, log_train_epoch_accuracy = [round(metric, 4) for metric in train_epoch_metrics]
                    log_val_epoch_perplexity, log_val_epoch_accuracy = [round(metric, 4) for metric in val_epoch_metrics]
                    self.experiment.log_metric("epoch_perplexity_train", log_train_epoch_perplexity)
                    self.experiment.log_metric("epoch_accuracy_train", log_train_epoch_accuracy)
                    self.experiment.log_metric("epoch_perplexity_val", log_val_epoch_perplexity)
                    self.experiment.log_metric("epoch_accuracy_val", log_val_epoch_accuracy)
                    self.experiment.log_metric("current_lr", current_lr)

                self.logger.info(log_message)

            """ Save Model """
            if self.epoch % self.save_every == 0:
                self._save_model(train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics)

            if self.experiment is not None:
                self.experiment.log_epoch_end(epoch_cnt=self.epoch)

    def _save_model(self, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics):

        checkpoint_filename = self.save_format.format(
            epoch=self.epoch,
            val_loss=val_epoch_loss,
            val_metrics='-'.join(['{:<.3}'.format(v) for v in val_epoch_metrics])
        )

        checkpoint_filepath = join(self.checkpoint_dir, checkpoint_filename)

        save_state = {
            'epoch': self.epoch,
            'train_loss': train_epoch_loss,
            'train_metrics': train_epoch_metrics,
            'val_loss': val_epoch_loss,
            'val_metrics': val_epoch_metrics,
            'checkpoint': checkpoint_filepath,
        }

        if self.epoch > 1:
            torch.save(self.model.state_dict(), checkpoint_filepath)
            self.history.append(save_state)

        representative_val_metric = val_epoch_metrics[0]
        if self.best_val_metric is None or self.best_val_metric > representative_val_metric:
            self.best_val_metric = representative_val_metric
            self.val_loss_at_best = val_epoch_loss
            self.train_loss_at_best = train_epoch_loss
            self.train_metrics_at_best = train_epoch_metrics
            self.val_metrics_at_best = val_epoch_metrics
            self.best_checkpoint_filepath = checkpoint_filepath

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_filepath))
            self.logger.info("Current best model is {}".format(self.best_checkpoint_filepath))

            if self.experiment is not None:
                self.experiment.log_other("current_best_model", self.best_checkpoint_filepath)

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds
