from layers.advanced_beam_nn.transformer_with_gru import Transformer


def build_model(hyperparameter_config, dataset_config, bert_model):
    n_layers = hyperparameter_config["transformer_n_layers"]
    ninp = hyperparameter_config["transformer_ninp"]
    nhidden = hyperparameter_config["transformer_nhidden"]
    nheads = hyperparameter_config["transformer_nheads"]
    dropout = hyperparameter_config["transformer_dropout"]
    use_gate = hyperparameter_config["transformer_use_gate"]

    vocab_size = len(bert_model.vocab)
    max_len = dataset_config['max_seq_length']

    # Load Networks
    model = Transformer(vocab_size, ninp, nhidden, nheads, n_layers, dropout, None, max_len)
    return model
