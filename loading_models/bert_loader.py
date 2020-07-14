import os
import json
import sys

from transformers import BertModel, BertTokenizer, BertConfig


class BertModelLoader(object):
    def __init__(self, model_name, base_dir):
        self.config = self.load_config(base_dir, model_name)
        self.models_path = os.path.join(base_dir, "models")
        self.bert_path = os.path.join(self.models_path, self.config["model_path"])
        self.do_lower_case = bool(self.config["do_lower_case"])
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    @staticmethod
    def load_config(base_dir, model_name):
        with open(os.path.join(base_dir, "configs", "bert_config.json"), "r") as config_file:
            config = json.load(config_file)
        return config[model_name]

    def load_tokenizer(self):
        return BertTokenizer.from_pretrained(self.bert_path, do_lower_case=self.do_lower_case)

    def load_model(self):
        config = BertConfig.from_pretrained(self.bert_path, output_hidden_states=True)
        model = BertModel.from_pretrained(self.bert_path, config=config)
        return model


if __name__ == "__main__":
    print("Loading...")
    bert_model = "bio_bert"
    model = BertModelLoader(bert_model, "..")
    print("Model loaded.")
    sys.exit()
