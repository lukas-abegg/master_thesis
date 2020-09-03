from layers.bert_weights_embeddings_transformer import TransformerModel as Transformer


def build_model(hyperparameter_config, dataset_config, bert_model, bert_tokenizer):
    n_layers = hyperparameter_config["transformer_n_layers"]
    ninp = hyperparameter_config["transformer_ninp"]
    nhidden = hyperparameter_config["transformer_nhidden"]
    nheads = hyperparameter_config["transformer_nheads"]
    dropout = hyperparameter_config["transformer_dropout"]
    use_gate = hyperparameter_config["transformer_use_gate"]

    vocab_size = len(bert_tokenizer.vocab)
    max_len = dataset_config[hyperparameter_config["bert_model"]]["max_seq_length"]

    # Load Networks
    model = Transformer(vocab_size, ninp, nheads, nhidden, n_layers, dropout, bert_model, max_len)
    return model
