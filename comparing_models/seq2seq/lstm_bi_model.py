from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, num_layers, dropout, src_pad_idx, tgt_pad_idx,
                 use_cuda=False):
        super(LSTMModel, self).__init__()

        self.use_cuda = use_cuda

        # Initialize encoder and decoder
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            embed_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=src_pad_idx
        )
        self.decoder = LSTMDecoder(
            output_dim=output_dim,
            embed_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_cuda=use_cuda,
            pad_idx=tgt_pad_idx
        )

    def forward(self, src, trg):
        # encoder_output: (seq_len, batch, hidden_size * num_directions)
        # _encoder_hidden: (num_layers * num_directions, batch, hidden_size)
        # _encoder_cell: (num_layers * num_directions, batch, hidden_size)
        encoder_out = self.encoder(src)

        decoder_out, attn_scores = self.decoder(trg, encoder_out)

        return decoder_out

    @staticmethod
    def get_normalized_probs(net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        vocab = net_output.size(-1)
        net_output1 = net_output.view(-1, vocab)
        if log_probs:
            return F.log_softmax(net_output1, dim=1).view_as(net_output)
        else:
            return F.softmax(net_output1, dim=1).view_as(net_output)


class LSTMEncoder(nn.Module):
    """LSTM encoder."""

    def __init__(self, input_dim, embed_dim, hid_dim, num_layers=1, dropout=0.1, pad_idx=0):
        super(LSTMEncoder, self).__init__()
        self.num_layers = num_layers

        self.padding_idx = pad_idx
        self.embed_tokens = Embedding(input_dim, embed_dim, self.padding_idx)

        self.dropout = dropout

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
        )

        self.hid_dim = hid_dim

    def forward(self, src_tokens):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # apply LSTM
        h0 = Variable(x.data.new(self.num_layers, bsz, self.hid_dim).zero_())
        c0 = Variable(x.data.new(self.num_layers, bsz, self.hid_dim).zero_())
        x, (final_hiddens, final_cells) = self.lstm(x, (h0, c0), )

        x = F.dropout(x, p=self.dropout, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.hid_dim]

        return x, final_hiddens, final_cells


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super(AttentionLayer, self).__init__()

        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(2 * output_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        attn_scores = F.softmax(attn_scores.t(), dim=1).t()  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hid_dim, num_layers, dropout, use_cuda=False, pad_idx=0):
        super(LSTMDecoder, self).__init__()
        self.use_cuda = use_cuda

        self.output_dim = output_dim

        self.dropout = dropout

        self.hid_dim = hid_dim

        self.padding_idx = pad_idx
        self.embed_tokens = Embedding(embed_dim, embed_dim, self.padding_idx)

        self.layers = nn.ModuleList([
            LSTMCell(embed_dim + hid_dim if layer == 0 else hid_dim, hid_dim)
            for layer in range(num_layers)
        ])
        self.attention = AttentionLayer(hid_dim, hid_dim)

        self.fc_out = Linear(hid_dim, output_dim, dropout=dropout)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out
        srclen = encoder_outs.size(0)

        x = self.embed_tokens(prev_output_tokens)  # (bsz, seqlen, embed_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)  # (seqlen, bsz, embed_dim)

        # initialize previous states (or get from cache during incremental generation)
        # cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        # initialize previous states (or get from cache during incremental generation)
        cached_state = get_incremental_state(self, incremental_state, 'cached_state')

        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            _, encoder_hiddens, encoder_cells = encoder_out
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            input_feed = Variable(x.data.new(bsz, self.hid_dim).zero_())

        attn_scores = Variable(x.data.new(srclen, seqlen, bsz).zero_())
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
            out = F.dropout(out, p=self.dropout, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        set_incremental_state(
            self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hid_dim)
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        x = self.fc_out(x)

        return x, attn_scores

    @staticmethod
    def max_positions():
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def reorder_incremental_state(self, incremental_state, new_order):
        cached_state = get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        if not isinstance(new_order, Variable):
            new_order = Variable(new_order)
        new_state = tuple(map(reorder_state, cached_state))
        set_incremental_state(self, incremental_state, 'cached_state', new_state)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value
