import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

from layers.transformer_xl.utils.adaptive_embed import AdaptiveEmbedding
from layers.transformer_xl.utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from layers.utils.gru import GRUGate


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inp):
        core_out = self.CoreNet(inp)

        return core_out


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # Get query, key and value for each token (NOTE SOME Inefficiency since
        # don't need query for any of the memory. Parallelization must make up for it
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    @staticmethod
    def _parallelogram_mask(h, w, left=False):
        # UserWarning: masked_fill_ received a mask with dtype torch.uint8,
        # this behavior is now deprecated,please use a mask with dtype torch.bool instead.
        # changed .byte() to .bool()
        # mask = torch.ones((h, w)).byte()
        mask = torch.ones((h, w)).bool()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    @staticmethod
    def _shift(x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    @staticmethod
    def _rel_shift(x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, use_stable_version=False):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        # if using stable version, then want layernorm of memory as well before MHA
        if mems is not None:
            cat = torch.cat([mems, w], 0)

            w_heads = self.qkv_net(cat) if not use_stable_version else self.qkv_net(self.layer_norm(cat))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(w) if not use_stable_version else self.qkv_net(self.layer_norm(w))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:  # THIS IS WHAT IS Usually executed
                # print('Attentionscore shape: ',attn_score.shape)
                # print('MASK SHAPE: ', attn_mask[:,:,:,None].shape)
                # print('MASK EL 1: ', attn_mask[:,:,0])
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # print('ATTENTION SCORE: ', attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection + layer normalization
        # output = self.layer_norm(w + attn_out)

        return attn_out


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, use_gate, use_stable_version,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.use_gate = use_gate
        self.use_stable_version = use_stable_version

        if self.use_gate:
            self.gate_mha = GRUGate(d_model)
            self.gate_mlp = GRUGate(d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward_orig(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.layer_norm1(dec_inp + output)
        output2 = self.pos_ff(output)
        output2 = self.layer_norm2(output + output2)
        return output2

    def forward_stable(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        # Layer norm will be applied at start of MHA module on both dec_inp2 and mems
        # dec_inp2 = self.layer_norm1(dec_inp)
        # First Layer norm will be applied within dec_attn

        dec_inp2 = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                 attn_mask=dec_attn_mask,
                                 mems=mems, use_stable_version=self.use_stable_version)

        # NOTE: In stable transformer they apply Relu before the layernorm/gate (in appendix C.3)
        if self.use_gate:
            dec_inp2 = self.gate_mha(dec_inp, F.relu(dec_inp2))
        else:
            dec_inp2 = dec_inp + F.relu(dec_inp2)

        dec_inp3 = self.layer_norm2(dec_inp2)

        dec_inp3 = self.pos_ff(dec_inp3)

        if self.use_gate:
            dec_inp3 = self.gate_mlp(dec_inp2, F.relu(dec_inp3))
        else:
            dec_inp3 = F.relu(dec_inp3) + dec_inp2

        return dec_inp3

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        if self.use_stable_version:
            return self.forward_stable(dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask, mems)

        return self.forward_orig(dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask, mems)


class Transformer(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False],
                 tgt_len=None, ext_len=0, mem_len=1,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, clamp_len=-1,
                 use_gate=True, use_stable_version=True):
        super(Transformer, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.adapt_inp = adapt_inp

        # TODO: Check if we can replace that with embedding layer from BERT
        if not self.adapt_inp:
            self.word_embed = nn.Embedding(n_token, d_embed)
        else:
            self.word_embed = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    use_stable_version=use_stable_version, use_gate=use_gate,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                    dropatt=dropatt)
            )

        if not self.adapt_inp:
            self.out_emb = nn.Linear(d_embed, n_token)
        else:
            # When using adaptive softmax and embedding, the embedding dimension is divided by div_val from bin $i$ to bin $i+1$.
            # This saves both GPU memory and the parameter budget
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs=cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_embed.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_embed.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_embed.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def init_gru_bias(self):
        for l in self.layers:
            l.gate_mha.init_bias()
            l.gate_mlp.init_bias()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    # NOTE: qlen looks to be number of characters in one example
    #       mlen is memory size
    def _update_mems(self, hids, mems, q_len, m_len):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `m_len + q_len` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `q_len` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `m_len + q_len - self.ext_len - self.mem_len`
        # to `m_len + q_len - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = m_len + max(0,
                                  q_len - 0 - self.ext_len)  # ext_len looks to usually be 0 (in their experiments anyways

            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()

        # Encoder Part
        word_embed = self.word_embed(dec_inp)

        if mems is not None:
            mlen = mems[0].size(0)
            # print('HERE: mlen: {}, len mems: {}, mems[0] shape: {}'.format(mlen, len(mems),mems[0].shape))
        else:
            mlen = 0

        klen = mlen + qlen

        # create the mask taking in consideration the mlen as well. All memory should be attended by the first query
        if self.same_length:
            all_ones = word_embed.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen)
                             + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_embed.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]

        hids = []
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_embed.device,
                               dtype=word_embed.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_embed)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        # SEEMS THAT THEY store memory per layer which makes sense to attend to (for ex if at first layer, if we were
        # applying attention to memory and this new data, this would give us the same result.
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                             self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems):
        if not mems:
            # print('INITIALIZED MEMS')
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]

        if not self.adapt_inp:
            loss = F.log_softmax(self.out_emb(pred_hid), dim=-1)
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    from utils.data_utils import LMOrderedIterator

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = Transformer(args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout,
                            dropatt=args.dropout, tie_weight=True,
                            d_embed=d_embed, div_val=div_val,
                            tie_projs=tie_projs,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                            cutoffs=cutoffs).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]