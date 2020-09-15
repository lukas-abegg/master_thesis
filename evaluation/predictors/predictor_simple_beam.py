""" This module will handle the text generation with beam search. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer


class Predictor(nn.Module):
    """ Load a trained model and translate in beam search fashion. """

    def __init__(self, model, checkpoint_filepath, tokenizer: BertTokenizer, device, max_length=512, beam_size=4, num_candidates=2):
        super(Predictor, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.num_candidates = num_candidates

        self.model = model
        self.device = device
        self.model.to(self.device)

        self.tokenizer = tokenizer

        # Model parameter
        self.max_seq_len = max_length
        self.src_pad_idx = tokenizer.pad_token_id
        self.trg_pad_idx = tokenizer.pad_token_id
        self.trg_bos_idx = tokenizer.cls_token_id
        self.trg_eos_idx = tokenizer.sep_token_id

        checkpoint = torch.load(checkpoint_filepath, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[self.trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((self.beam_size, self.max_seq_len), self.trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, self.max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_key_padding_mask):
        trg_seq, trg_key_padding_mask, trg_mask = self._prepare_output(trg_seq)
        trg_seq = trg_seq.to(self.device)
        trg_key_padding_mask = trg_key_padding_mask.to(self.device)
        trg_mask = trg_mask.to(self.device)

        dec_output = self.model.decoder(trg_seq, enc_output, tgt_mask=trg_mask)
                                        # tgt_key_padding_mask=trg_key_padding_mask,
                                        # memory_key_padding_mask=src_key_padding_mask)
        dec_output = dec_output.to(self.device)

        if hasattr(self.model, 'x_logit_scale'):
            dec_output = F.softmax(self.model.trg_word_prj(dec_output) * self.model.x_logit_scale, dim=-1)
        else:
            dec_output = F.softmax(self.model.trg_word_prj(dec_output), dim=-1)

        return dec_output

    def _get_init_state(self, src_seq, src_key_padding_mask):
        enc_output = self.model.encoder(src_seq, src_key_padding_mask)
        enc_output = enc_output.to(self.device)
        dec_output = self._model_decode(self.init_seq, enc_output, src_key_padding_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(self.beam_size)

        scores = torch.log(best_k_probs).view(self.beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(1, self.beam_size, 1)
        src_key_padding_mask = src_key_padding_mask.repeat(self.beam_size, 1)

        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    @staticmethod
    def generate_key_padding_mask(seq):
        # x: (batch_size, seq_len)
        pad_mask = seq == 0  # (batch_size, seq_len)
        return pad_mask

    def _prepare_input(self, src_seq, tokenize):
        if tokenize:
            src_seq = self.tokenizer.encode(src_seq, add_special_tokens=True, max_length=self.max_seq_len)
        # src_seq = F.pad(torch.as_tensor(src_seq), pad=(0, self.max_length - len(src_seq)), mode='constant', value=self.tokenizer.pad_token_id)
        src_seq = torch.as_tensor(src_seq)
        src_mask = self.generate_key_padding_mask(src_seq)

        src_seq = src_seq.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)

        return src_seq, src_mask

    def _prepare_output(self, out_seq):
        # out_seq = F.pad(torch.as_tensor(out_seq), pad=(0, self.max_length - out_seq.size(1)), mode='constant',
        # value=self.tokenizer.pad_token_id)
        out_seq = torch.as_tensor(out_seq)
        out_key_padding_mask = self.generate_key_padding_mask(out_seq)
        out_mask = self.model.generate_square_subsequent_mask(out_seq)

        return out_seq, out_key_padding_mask, out_mask

    def decode_predicted_string(self, seq_ids):
        return self.tokenizer.decode(seq_ids)

    def predict_one(self, src_seq, only_one=False, tokenize=True):
        # Only accept batch size equals to 1 in this function.
        with torch.no_grad():

            source_tensor, source_key_padding_mask = self._prepare_input(src_seq, tokenize)
            source_tensor = source_tensor.to(self.device)
            source_key_padding_mask = source_key_padding_mask.to(self.device)

            enc_output, gen_seq, scores = self._get_init_state(source_tensor, source_key_padding_mask)

            ans_idx = 0  # default
            for step in range(2, self.max_seq_len):  # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, source_key_padding_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == self.trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, self.max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == self.beam_size:
                    # TODO: Try different terminate conditions.
                    scores.to(self.device)
                    seq_lens.to(self.device)
                    self.num_candidates.to(self.device)
                    self.alpha.to(self.device)
                    _, ans_idx_seq = scores.div(seq_lens.float() ** self.alpha).topk(self.num_candidates)
                    break

        candidates = [gen_seq[ans_idx][:seq_lens[ans_idx]].tolist() for ans_idx in ans_idx_seq]
        hs = [self.decode_predicted_string(h) for h in candidates]

        if only_one:
            return hs[0]
        else:
            return hs
