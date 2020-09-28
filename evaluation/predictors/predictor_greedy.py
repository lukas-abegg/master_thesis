""" This module will handle the text generation with greedy search. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer


class Predictor(nn.Module):
    """ Load a trained model and translate in beam search fashion. """

    def __init__(self, model, checkpoint_filepath, tokenizer: BertTokenizer, device, max_length=512):
        super(Predictor, self).__init__()
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

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

        checkpoint = torch.load(checkpoint_filepath, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[self.trg_bos_idx]]))

    def _model_decode(self, trg_seq, enc_output, src_key_padding_mask):
        trg_seq, trg_key_padding_mask, trg_mask = self._prepare_output(trg_seq)
        trg_seq = trg_seq.to(self.device)
        trg_key_padding_mask = trg_key_padding_mask.to(self.device)
        trg_mask = trg_mask.to(self.device)

        dec_output = self.model.decoder(trg_seq, enc_output, tgt_mask=trg_mask,
                                        tgt_key_padding_mask=trg_key_padding_mask,
                                        memory_key_padding_mask=src_key_padding_mask)
        dec_output = dec_output.to(self.device)

        if hasattr(self.model, 'x_logit_scale'):
            dec_output = self.softmax(self.model.trg_word_prj(dec_output) * self.model.x_logit_scale)
        else:
            dec_output = self.softmax(self.model.trg_word_prj(dec_output))

        return dec_output

    def _get_init_state(self, src_seq, src_key_padding_mask):
        enc_output = self.model.encoder(src_seq, src_key_padding_mask)
        enc_output = enc_output.to(self.device)
        dec_output = self._model_decode(self.init_seq, enc_output, src_key_padding_mask)

        best_k_prob, best_k_idx = dec_output[:, -1, :].topk(1)
        gen_seq = torch.cat([self.init_seq[0], best_k_idx[0]], dim=0).unsqueeze(0)
        return gen_seq, enc_output

    @staticmethod
    def _get_the_best_score_and_idx(gen_seq, dec_output):
        _, best_idx = dec_output[:, -1, :].topk(1)

        # Set the best tokens in this beam search step
        gen_seq = torch.cat([gen_seq, best_idx], dim=1)

        return gen_seq

    @staticmethod
    def generate_key_padding_mask(seq):
        # x: (batch_size, seq_len)
        pad_mask = seq == 0  # (batch_size, seq_len)
        return pad_mask

    def _prepare_input(self, src_seq, tokenize):
        if tokenize:
            src_seq = self.tokenizer.encode(src_seq, add_special_tokens=True, max_length=self.max_seq_len)
        src_seq = torch.as_tensor(src_seq)
        src_mask = self.generate_key_padding_mask(src_seq)

        src_seq = src_seq.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)

        return src_seq, src_mask

    def _prepare_output(self, out_seq):
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

            gen_seq, enc_output = self._get_init_state(source_tensor, source_key_padding_mask)

            for step in range(2, self.max_seq_len):  # decode up to max length
                gen_seq = gen_seq.to(self.device)
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, source_key_padding_mask)
                gen_seq = self._get_the_best_score_and_idx(gen_seq[:, :step], dec_output)

                if (gen_seq[0, step] == self.trg_eos_idx).item():
                    break

        hs = self.decode_predicted_string(gen_seq[0].tolist())

        if only_one:
            return hs
        else:
            return [hs]
