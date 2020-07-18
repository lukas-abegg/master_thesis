import sys

import torch
from torch.backends import cudnn

from layers.bert_embedding_layer import BertEmbedding
from layers.transformer import Transformer
from loading_models.bert_loader import BertModelLoader
from preprocessings.load_wiki_simple import load_dataset
from training.direct_training import train
from utils.training_utils import load_config

# import comet_ml in the top of your file
from comet_ml import Experiment


def get_model(config, bert_model):

    n_layers = config["transformer_n_layers"]
    d_model = config["transformer_d_model"]
    attn_heads = config["transformer_attn_heads"]
    dropout = config["transformer_dropout"]
    adaptive_span = config["transformer_adaptive_span"]
    span_length = config["transformer_span_length"]
    transformer_with_gru = config["transformer_with_gru"]

    vocab_size = 2000

    # Load Networks
    embedding_layer = BertEmbedding(bert_model)
    model = Transformer(vocab_size, d_model, n_layers, attn_heads, embedding_layer, dropout)

    return model


def load_tracking(hyper_params):
    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                            project_name="project",
                            workspace="workspace")

    experiment.log_parameters(hyper_params)
    experiment.display()
    return experiment


def stop_tracking(experiment):
    experiment.end()


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Load Config
    config = load_config()

    # Load Bert
    bert_model_name = config["bert_model"]
    bert_model_loader = BertModelLoader(bert_model_name, "")

    # Load Transformer Model
    #model = get_model(config, bert_model_loader.model)

    # Load Dataset
    loaders, vocab = load_dataset(bert_model_name, bert_model_loader)
    training_loader, validation_loader, test_loader = loaders[0], loaders[1], loaders[2]

    # Start Tracking
    tracking_active = config["tracking_active"]
    experiment = None

    if tracking_active:
        experiment = load_tracking(config)

    try:
        train(None, training_loader, validation_loader, config, device, experiment=experiment)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

    if tracking_active:
        stop_tracking(experiment)
