from transformers import BertTokenizer
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
        self.hypothesises = None

    def predict_one(self, source, num_candidates=1):
        input_ids = self.tokenizer.encode(source)
        beam_output = self.model.generate(
            input_ids,
            max_length=self.max_length,
            num_beams=self.beam_size,
            num_return_sequences=num_candidates,
            early_stopping=True
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in beam_output]
