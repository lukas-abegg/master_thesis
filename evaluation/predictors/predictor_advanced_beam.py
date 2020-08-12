from transformers import BertTokenizer

from layers.utils.beam import Beam

import torch


class Predictor:

    def __init__(self, model, checkpoint_filepath, tokenizer: BertTokenizer, device, max_length=30, beam_size=8):
        self.model = model
        self.max_length = max_length
        self.beam_size = beam_size
        self.device = device

        checkpoint = torch.load(checkpoint_filepath, map_location=device)
        self.model.load_state_dict(checkpoint)

        self.tokenizer = tokenizer
        self.attentions = None
        self.hypothesises = None

    def pad_masking(x, target_len):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        padded_positions = x == 0  # (batch_size, seq_len)
        pad_mask = padded_positions.unsqueeze(1).expand(batch_size, target_len, seq_len)
        return pad_mask

    def _decode_predicted_string(self, seq_ids):
        return self.tokenizer.decode(seq_ids)

    def predict_one(self, source, num_candidates=5):
        with torch.no_grad():
            source_preprocessed = self.tokenizer.encode(source)
            source_tensor = torch.tensor(source_preprocessed).unsqueeze(0)  # why unsqueeze?

            sources_mask = self.pad_masking(source_tensor, source_tensor.size(1))
            memory_mask = self.pad_masking(source_tensor, 1)
            memory = self.model.encoder(source_tensor, sources_mask)

            decoder_state = self.model.decoder.init_decoder_state()

            # Repeat beam_size times
            memory_beam = memory.detach().repeat(self.beam_size, 1, 1)  # (beam_size, seq_len, hidden_size)

            beam = Beam(beam_size=self.beam_size, min_length=0, n_top=num_candidates, ranker=None)

            for _ in range(self.max_length):

                new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
                decoder_outputs, decoder_state = self.model.decoder(new_inputs, memory_beam,
                                                                    memory_mask,
                                                                    state=decoder_state)

                attention = self.model.decoder.decoder_layers[-1].memory_attention_layer.attention
                beam.advance(decoder_outputs.squeeze(1), attention)

                beam_current_origin = beam.get_current_origin()  # (beam_size, )
                decoder_state.beam_update(beam_current_origin)

                if beam.done():
                    break

            scores, ks = beam.sort_finished(minimum=num_candidates)
            hypothesises, attentions = [], []
            for i, (times, k) in enumerate(ks[:num_candidates]):
                hypothesis, attention = beam.get_hypothesis(times, k)
                hypothesises.append(hypothesis)
                attentions.append(attention)

            self.attentions = attentions
            self.hypothesises = [[token.item() for token in h] for h in hypothesises]
            hs = self._decode_predicted_string(self.hypothesises)
            return list(reversed(hs))
