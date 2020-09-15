from layers.bert_embedding_layer import BertEmbedding
from layers.advanced_beam_nn.transformer_with_gru import Transformer


def build_model(hyperparameter_config, dataset_config, bert_model, bert_tokenizer):
    n_layers = hyperparameter_config["transformer_n_layers"]
    ninp = hyperparameter_config["transformer_ninp"]
    nhidden = hyperparameter_config["transformer_nhidden"]
    nheads = hyperparameter_config["transformer_nheads"]
    dropout = hyperparameter_config["transformer_dropout"]
    use_gate = hyperparameter_config["transformer_use_gate"]

    vocab_size = len(bert_tokenizer.vocab)
    max_len = dataset_config['max_seq_length']

    # Load Networks
    embedding_layer = BertEmbedding(bert_model)
    model = Transformer(vocab_size, ninp, nhidden, nheads, n_layers, dropout, embedding_layer, max_len)

    return model
