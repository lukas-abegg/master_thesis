import os

import torch

from build_models.transformer_baseline import build_model as build_baseline
from build_models.bert_weights_embeddings_transformer import build_model as build_bert_weights_embeddings
from build_models.advanced_beam.transformer_baseline_advanced import build_model as build_baseline_advanced
from build_models.advanced_beam.transformer_with_bert_embedding_advanced import build_model as build_transformer_with_bert_embedding_advanced
from build_models.bert_encoding_transformer import build_model as build_bert_encoding_transformer


def build_model(hyperparameter_config, dataset_config, bert_model, bert_tokenizer):
    model_name = hyperparameter_config['model']

    if model_name == "baseline":
        return build_baseline(hyperparameter_config, dataset_config, bert_tokenizer)
    elif model_name == "bert_encoding":
        return build_bert_encoding_transformer(hyperparameter_config, dataset_config, bert_model, bert_tokenizer)
    elif model_name == "bert_weights_embeddings":
        return build_bert_weights_embeddings(hyperparameter_config, dataset_config, bert_model, bert_tokenizer)
    elif model_name == "baseline_advanced":
        return build_baseline_advanced(hyperparameter_config, dataset_config, bert_tokenizer)
    elif model_name == "transformer_with_bert_embedding_advanced":
        return build_transformer_with_bert_embedding_advanced(hyperparameter_config, dataset_config, bert_model, bert_tokenizer)
    else:
        return build_baseline(hyperparameter_config, dataset_config, bert_tokenizer)


def load_model(model, base_dir, hyperparameter_config):
    PATH = os.path.join(base_dir, hyperparameter_config['model_path'])
    model.load_state_dict(torch.load(PATH))
    return model
