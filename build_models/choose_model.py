from build_models.transformer_baseline import build_model as build_baseline
from build_models.transformer_with_bert_embedding import build_model as build_transformer_with_bert_embedding


def build_model(hyperparameter_config, dataset_config, bert_model):
    model_name = hyperparameter_config['model']

    if model_name == "baseline":
        return build_baseline(hyperparameter_config, dataset_config, bert_model)
    elif model_name == "transformer_with_bert_embedding":
        return build_transformer_with_bert_embedding(hyperparameter_config, dataset_config, bert_model)
    else:
        return build_baseline(hyperparameter_config, dataset_config, bert_model)
