import torch
import sys

from loading_models.bert_loader import BertModelLoader
from preprocessings.bert_tokenizing import BertPreprocessor


class BertEmbedding(torch.nn.Module):
    def __init__(self, bert_model):
        super(BertEmbedding, self).__init__()
        self.bert = bert_model
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.bert.eval()

    def forward(self, input_ids, attention_mask):

        sequence_output, _, hidden_states = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask)

        print('\n==== Hidden Layers Output ====\n')

        print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
        layer_i = 0
        print("Number of batches:", len(hidden_states[layer_i]))
        batch_i = 0
        print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        token_i = 0
        print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
        # `hidden_states` is a Python list.
        print('      Type of hidden_states: ', type(hidden_states))
        # Each layer in the list is a torch tensor.
        print('Tensor shape for each layer: ', hidden_states[0].size())

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)
        print("Size:", token_embeddings.size())
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        print("Size:", token_embeddings.size())
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)
        print("Size:", token_embeddings.size())

        # `token_embeddings` is a [22 x 12 x 768] tensor.
        mode = "concat"

        if mode == "concat":
            return self.concatenate(token_embeddings)
        elif mode == "sum":
            return self.summing_up(token_embeddings)
        else:
            raise ValueError("The embedding mode (%s) for Bert is not implemented." % mode)

    @staticmethod
    def concatenate(token_embeddings):

        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor

            # Concatenate the vectors (that is, append them together) from the last
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)

        print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

        return token_vecs_cat

    @staticmethod
    def summing_up(token_embeddings):
        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)

            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

        return token_vecs_sum

    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False

        # Similar to:
        # self.bert.weight.requires_grad = False
        # self.bert.bias.requires_grad = False


def print_params(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


if __name__ == "__main__":
    # Model computations
    bert_model = "bio_bert"
    # Load Bert
    loaded_bert_model = BertModelLoader(bert_model, "..")
    # Load Networks
    model = BertEmbedding(loaded_bert_model.model)
    print_params(model)

    tokenizer = BertPreprocessor(loaded_bert_model.tokenizer)
    padded_sentence, attention_mask = tokenizer.encode_sentence("Hello, my dog is cute", max_len=512)
    padded_sentence = padded_sentence.unsqueeze(0)  # Batch size 1
    attention_mask = attention_mask.unsqueeze(0)  # Batch size 1
    outputs = model(padded_sentence, attention_mask)
    sys.exit()

