from transformers import BertTokenizer

from layers.utils.beam_search_advanced import Beam

import torch
import torch.nn.functional as F


class Predictor:

    def __init__(self, model, checkpoint_filepath, tokenizer: BertTokenizer, device, max_length=30, beam_size=8, num_candidates=5):
        self.model = model
        self.max_length = max_length
        self.beam_size = beam_size
        self.device = device
        self.num_candidates = num_candidates

        checkpoint = torch.load(checkpoint_filepath, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.tokenizer = tokenizer
        #self.attentions = None
        self.hypothesises = None

    def decode_predicted_string(self, seq_ids):
        return self.tokenizer.decode(seq_ids)

    @staticmethod
    def generate_key_padding_mask(seq):
        # x: (batch_size, seq_len)
        pad_mask = seq == 0  # (batch_size, seq_len)
        return pad_mask

    def _prepare_input(self, src_seq, tokenize):
        if tokenize:
            src_seq = self.tokenizer.encode(src_seq, add_special_tokens=True, max_length=self.max_length)
        #src_seq = F.pad(torch.as_tensor(src_seq), pad=(0, self.max_length - len(src_seq)), mode='constant', value=self.tokenizer.pad_token_id)
        src_seq = torch.as_tensor(src_seq)
        src_mask = self.generate_key_padding_mask(src_seq)

        src_seq = src_seq.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)

        return src_seq, src_mask

    def _prepare_output(self, out_seq):
        #out_seq = F.pad(torch.as_tensor(out_seq), pad=(0, self.max_length - out_seq.size(1)), mode='constant',
                        #value=self.tokenizer.pad_token_id)
        out_seq = torch.as_tensor(out_seq)
        out_mask = self.generate_key_padding_mask(out_seq)

        return out_seq, out_mask

    def predict_one(self, source, only_one=False, tokenize=True):
        with torch.no_grad():
            source_tensor, source_key_padding_mask = self._prepare_input(source, tokenize)
            source_tensor = source_tensor.to(self.device)
            source_key_padding_mask = source_key_padding_mask.to(self.device)
            #source_tensor = torch.tensor(source_preprocessed[0]).unsqueeze(0)  # why unsqueeze?

            #sources_mask = self.pad_masking(source_tensor, source_tensor.size(1))
            #memory_mask = self.pad_masking(source_tensor, 1)
            memory = self.model.encoder(source_tensor, source_key_padding_mask)
            memory = memory.to(self.device)

            #decoder_state = self.model.decoder.init_decoder_state()

            # Repeat beam_size times
            #memory_beam = memory.detach().repeat(self.beam_size, 1, 1)  # (beam_size, seq_len, hidden_size)
            memory = memory.repeat(1, self.beam_size, 1)
            source_key_padding_mask = source_key_padding_mask.repeat(self.beam_size, 1)

            beam = Beam(beam_size=self.beam_size, min_length=0, n_top=self.num_candidates, ranker=None,
                        start_token_id=self.tokenizer.cls_token_id, end_token_id=self.tokenizer.sep_token_id)

            for _ in range(self.max_length):

                new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
                new_inputs, trg_key_padding_mask = self._prepare_output(new_inputs)
                new_inputs = new_inputs.to(self.device)
                trg_key_padding_mask = trg_key_padding_mask.to(self.device)

                decoder_outputs = self.model.decoder(new_inputs, memory, tgt_key_padding_mask=trg_key_padding_mask,
                                                     memory_key_padding_mask=source_key_padding_mask)
                decoder_outputs = decoder_outputs.to(self.device)
                #attention = self.model.decoder.decoder_layers[-1].memory_attention_layer.attention
                #beam.advance(decoder_outputs.squeeze(1), attention)

                if hasattr(self.model, 'x_logit_scale'):
                    outputs = self.model.trg_word_prj(decoder_outputs) * self.model.x_logit_scale
                else:
                    outputs = self.model.trg_word_prj(decoder_outputs)

                outputs = outputs.to(self.device)
                outputs_softmax = F.softmax(outputs, dim=-1)
                beam.advance(outputs_softmax.squeeze(1))

                beam_current_origin = beam.get_current_origin()  # (beam_size, )
                #decoder_state.beam_update(beam_current_origin)

                if beam.done():
                    break

            scores, ks = beam.sort_finished(minimum=self.num_candidates)
            #hypothesises, attentions = [], []
            hypothesises = []
            for i, (times, k) in enumerate(ks[:self.num_candidates]):
                #hypothesis, attention = beam.get_hypothesis(times, k)
                hypothesis = beam.get_hypothesis(times, k)
                hypothesises.append(hypothesis)
                #attentions.append(attention)

            #self.attentions = attentions
            self.hypothesises = [[token.item() for token in h] for h in hypothesises]
            hs = [self.decode_predicted_string(h) for h in self.hypothesises]
            hs = list(reversed(hs))

            if only_one:
                return hs[0]
            else:
                return hs
