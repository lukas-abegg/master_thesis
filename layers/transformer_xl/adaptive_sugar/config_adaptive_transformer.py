
PARAMS_CONFIG = {
    # env-specific
    'env_params': {
        '--distributed': {
            'action': 'store_true',
            'default': False,
            'help': 'enable distributed training.'
                    '(otherwise will use all available GPUs with dataparallel)',
            'dest': 'distributed'
        },
        '--local_rank': {
            'type': int,
            'default': 0,
            'help': 'used in distributed training',
            'dest': 'local_rank'
        },
    },
    # data-specific
    'data_params': {
        '--data': {
            'type': str,
            'default': 'data/text8',
            'help': 'data location '
                    '(must contain train.txt, valid.txt and test.txt)',
            'dest': 'data_path'
        },
        '--data-unit': {
            'type': str,
            'default': 'bpc',
            'choices': ['bpc', 'ppl'],
            'help': 'loss unit to log',
            'dest': 'data_unit'
        },
    },
    # model-specific
    'model_params': {
        '--hid-sz': {
            'type': int,
            'default': 256,
            'help': 'hidden size (i.e. model size)',
            'dest': 'hidden_size'
        },
        '--inner-hid-sz': {
            'type': int,
            'default': 1024,
            'help': 'inner hidden size of FF layer',
            'dest': 'inner_hidden_size'
        },
        '--nlayers': {
            'type': int,
            'default': 8,
            'help': 'number of layers',
            'dest': 'nb_layers'
        },
        '--block-sz': {
            'type': int,
            'default': 64,
            'help': 'block size '
                    '(the length of sequence to process in parallel)',
            'dest': 'block_size'
        },
        '--nheads': {
            'type': int,
            'default': 2,
            'help': 'number of self-attention heads',
            'dest': 'nb_heads'
        },
        '--attn-span': {
            'type': int,
            'default': 32,
            'help': 'length of the attention span',
            'dest': 'attn_span'
        },
        '--dropout': {
            'type': float,
            'default': 0.2,
            'help': 'dropout rate of ReLU and attention',
            'dest': 'dropout'
        },
        '--emb-dropout': {
            'type': float,
            'default': 0.,
            'help': 'the dropout rate applied on I/O embeddings',
            'dest': 'emb_dropout'
        },
    },
    # optimization-specific
    'optim_params': {
        '--lr': {
            'type': float,
            'default': 0.03,
            'help': 'learning rate',
            'dest': 'lr'
        },
        '--momentum': {
            'type': float,
            'default': 0.9,
            'help': 'SGD momentum',
            'dest': 'momentum'
        },
        '--optim': {
            'type': str,
            'default': 'sgd',
            'help': 'optimization method: sgd | adagrad',
            'dest': 'optim'
        },
        '--lr-warmup': {
            'type': int,
            'default': 0,
            'help': 'linearly increase LR from 0 '
                    'during first lr_warmup updates',
            'dest': 'lr_warmup'
        },
        '--grad-clip': {
            'type': float,
            'default': 0,
            'help': '[only works with adagrad!] '
                    'clip gradient of each module parameters by a given '
                    'value',
            'dest': 'grad_clip'
        },
    },
    # trainer-specific
    'trainer_params': {
        '--batch-sz': {
            'type': int,
            'default': 64,
            'help': 'batch size',
            'dest': 'batch_size'
        },
        '--batch-split': {
            'type': int,
            'default': 1,
            'help': 'split a batch into smaller parts to fit in GPU memory',
            'dest': 'batch_split'
        },
        '--nbatches': {
            'type': int,
            'default': 1000,
            'help': 'number of batches in each iteration',
            'dest': 'nb_batches_per_iter'
        },
        '--niter': {
            'type': int,
            'default': 1000,
            'help': 'number of iterations to train',
            'dest': 'nb_iter'
        },
        '--checkpoint': {
            'type': str,
            'default': '',
            'help': 'path to save/load model',
            'dest': 'checkpoint_path'
        },
        '--full-eval-mode': {
            'action': 'store_true',
            'default': False,
            'help': 'do evaluation on the whole validation and the test data',
            'dest': 'full_eval_mode'
        },
    },
    # adaptive I/O specific params
    'adapt_io_params': {
        '--adapt-io': {
            'action': 'store_true',
            'default': False,
            'help': 'enable adaptive input and output representations',
            'dest': 'adapt_io_enabled'
        },
        '--adapt-io-tied': {
            'action': 'store_true',
            'default': False,
            'help': 'tie the input parameters with the output parameters',
            'dest': 'adapt_io_tied'
        },
        '--adapt-io-divval': {
            'type': int,
            'default': 4,
            'help': 'dimension division value',
            'dest': 'adapt_io_divval'
        },
        '--adapt-io-cutoffs': {
            'type': int,
            'default': [20000, 40000, 200000],
            'help': 'cutoffs values',
            'dest': 'adapt_io_cutoffs'
        },
    },
    # adaptive attention span specific params
    'adapt_span_params': {
        '--adapt-span': {
            'action': 'store_true',
            'default': False,
            'help': 'enable adaptive attention span',
            'dest': 'adapt_span_enabled'
        },
        '--adapt-span-loss': {
            'type': float,
            'default': 0,
            'help': 'the loss coefficient for span lengths',
            'dest': 'adapt_span_loss'
        },
        '--adapt-span-ramp': {
            'type': int,
            'default': 32,
            'help': 'ramp length of the soft masking function',
            'dest': 'adapt_span_ramp'
        },
        '--adapt-span-init': {
            'type': float,
            'default': 0,
            'help': 'initial attention span ratio',
            'dest': 'adapt_span_init'
        },
        '--adapt-span-cache': {
            'action': 'store_true',
            'default': False,
            'help': 'adapt cache size as well to reduce memory usage',
            'dest': 'adapt_span_cache'
        },
    },
    # persistent memory specific params
    'pers_mem_params': {
        '--pers-mem-size': {
            'type': int,
            'default': 0,
            'help': 'the number of persistent memory vectors',
            'dest': 'pers_mem_size'
        },
    },
}


import os
import math
import argparse

import torch


def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:  # e.g., 'model_params'
        for param_flag, param_config in params_config[params_category].items():
            # e.g., param_flag = '--block-sz'
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)


def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)
    return {
        params_category: {
            param_config['dest']:
                namespace.__getattribute__(param_config['dest'])
            for param_config in params_config[params_category].values()
        }
        for params_category in params_config
    }


##############################################################################
# ENVIRONMENT
##############################################################################

def _torch_distributed_init_process_group(local_rank):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print('my rank={} local_rank={}'.format(rank, local_rank))
    torch.cuda.set_device(local_rank)
    return {
        'rank': rank,
        'world_size': world_size,
    }


def set_up_env(env_params):
    if env_params['distributed']:
        env_params.update(
            _torch_distributed_init_process_group(
                local_rank=env_params['local_rank']))
    env_params['device'] = torch.device('cuda')


##############################################################################
# OPTIMIZER AND SCHEDULER
##############################################################################

def _get_grad_requiring_params(model):
    nb_parameters = 0
    grad_requiring_params = []
    for param in model.parameters():
        if param.requires_grad:
            nb_parameters += param.numel()
            grad_requiring_params.append(param)
    print('nb_parameters={:.2f}M'.format(nb_parameters / 1e6))
    return grad_requiring_params


##############################################################################
# CHECKPOINT
##############################################################################

def _load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger,
                     distributed):
    print('loading from a checkpoint at {}'.format(checkpoint_path))
    if distributed:
        # the model is saved from gpu0 so we need to map it to CPU first
        checkpoint_state = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint_state = torch.load(checkpoint_path)
    iter_init = checkpoint_state['iter_no'] + 1  # next iteration
    model.load_state_dict(checkpoint_state['model'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])
    logger.load_state_dict(checkpoint_state['logger'])
    if 'scheduler_iter' in checkpoint_state:
        # we only need the step count
        scheduler.step(checkpoint_state['scheduler_iter'])
    return iter_init


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger,
                    distributed):
    if checkpoint_path and os.path.exists(checkpoint_path):
        return _load_checkpoint(checkpoint_path=checkpoint_path,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                logger=logger,
                                distributed=distributed)
    return 0


def save_checkpoint(checkpoint_path, iter_no, model,
                    optimizer, scheduler, logger):
    if checkpoint_path:
        checkpoint_state = {
            'iter_no': iter_no,  # last completed iteration
            'model': model.state_dict(),
            'logger': logger.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint_state['scheduler_iter'] = scheduler.last_epoch
        torch.save(checkpoint_state, checkpoint_path)


##############################################################################
# LOGGER
##############################################################################

class Logger:
    def __init__(self, data_unit):
        self.data_unit = data_unit
        self._state_dict = dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def _log(self, title, value):
        if title not in self._state_dict:
            self._state_dict[title] = []
        self._state_dict[title].append(value)

    def log_iter(self, iter_no, nb_batches_per_iter, loss_train, loss_val,
                 elapsed, model):
        step = (iter_no + 1) * nb_batches_per_iter
        self._log(title='step', value=step)
        msg = 'steps: {}'.format(step)
        if self.data_unit == 'bpc':
            train_bpc = float(loss_train / math.log(2))
            val_bpc = float(loss_val / math.log(2))
            msg += '\ttrain: {:.3f}bpc\tval: {:.3f}bpc'.format(train_bpc, val_bpc)
            self._log(title='train_bpc', value=train_bpc)
            self._log(title='val_bpc', value=val_bpc)
        else:
            train_ppl = math.exp(loss_train)
            val_ppl = math.exp(loss_val)
            msg += '\ttrain: {:.2f}ppl\tval: {:.2f}ppl'.format(train_ppl, val_ppl)
            self._log(title='train_ppl', value=train_ppl)
            self._log(title='val_ppl', value=val_ppl)
        msg += '\tms/batch: {:.1f}'.format(elapsed)

        if model.module.layers[0].attn.attn.adapt_span_enabled:
            avg_spans = []
            max_spans = []
            for layer in model.module.layers:
                avg_spans.append(
                    layer.attn.attn.adaptive_span.get_current_avg_span())
                max_spans.append(
                    layer.attn.attn.adaptive_span.get_current_max_span())
            span_avg = float(sum(avg_spans)) / len(avg_spans)
            span_max = float(max(max_spans))
            self._log('span_avg', span_avg)
            self._log('span_max', span_max)
            msg += "\tspan_avg: {:.0f}\tspan_max: {:.0f}".format(span_avg, span_max)

        print(msg)


def launch(env_params,
           model_params,
           adapt_io_params,
           adapt_span_params,
           pers_mem_params,
           optim_params,
           data_params,
           trainer_params):
    # ENVIRONMENT (device, distributed, etc.)
    set_up_env(env_params)
    device = env_params['device']
    distributed = env_params['distributed']

    print('model_params:\t', model_params)
    print('optim_params:\t', optim_params)
    print('data_params:\t', data_params)
    print('trainer_params:\t', trainer_params)
    print('adapt_io_params:\t', adapt_io_params)
    print('adapt_span_params:\t', adapt_span_params)
    print('pers_mem_params:\t', pers_mem_params)


if __name__ == '__main__':

    launch(**get_params(params_config=PARAMS_CONFIG))