import numpy as np
import torch

from transformers import BertTokenizer


class BertPreprocessor:
    def __init__(self, tokenizer):
        self.tokenizer: BertTokenizer = tokenizer

    @staticmethod
    def mark_text(text):
        try:
            return "[CLS] " + text + " [SEP]"
        except:
            return "[CLS] [SEP]"

    def tokenize_into_words(self, text):
        tokenized_text, _ = self.tokenize_text_per_word(text)
        tokenized_text_into_words = []
        for word in tokenized_text:
            if len(word) == 1:
               tokenized_text_into_words.append(word[0])
            else:
                concatenated_word = ""
                for token in word:
                    concatenated_word = concatenated_word + token.replace("##", "")
                tokenized_text_into_words.append(concatenated_word)

        return tokenized_text_into_words

    def tokenize_text(self, text):
        marked_text = self.mark_text(text)
        tokenized_text = self.tokenizer.tokenize(marked_text)
        ids_encoded = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return tokenized_text, ids_encoded

    def tokenize_text_per_word(self, text):
        tokenized_text, ids_encoded = self.tokenize_text(text)
        tokens = []
        ids = []
        last_word_token = []
        last_word_ids = []
        for i, token in enumerate(tokenized_text):
            if token.startswith("##"):
                last_word_token.append(token)
                last_word_ids.append(ids_encoded[i])
            else:
                if len(last_word_token):
                    tokens.append(last_word_token)
                    ids.append(last_word_ids)
                last_word_token = [token]
                last_word_ids = [ids_encoded[i]]
        if len(last_word_token):
            tokens.append(last_word_token)
            ids.append(last_word_ids)

        return tokens, ids

    def encode_sentences(self, sentences, max_len):
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tokenized_sentences = sentences.apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))

        # Padding: fill up with 0's for paddings
        padded_sentences = np.array([i + [0] * (max_len - len(i)) for i in tokenized_sentences.values])

        # Masking: Set attention 1 for id's and 0's for paddings
        attention_masks = np.where(padded_sentences != 0, 1, 0)

        padded_sentences = torch.tensor(padded_sentences)
        attention_masks = torch.tensor(attention_masks)

        return tokenized_sentences, padded_sentences, attention_masks

    def encode_sentence_per_word(self, sentence, max_len):
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tokenized_sentence, ids_sentence = self.tokenize_text_per_word(sentence)
        tokenized_sentence = np.asarray(tokenized_sentence)

        # Padding: fill up with 0's for paddings
        padded_sentence = np.asarray(ids_sentence + [0] * (max_len - len(ids_sentence)))

        # Masking: Set attention 1 for id's and 0's for paddings
        attention_mask = np.where(padded_sentence != 0, 1, 0)

        # Transform to Tensors
        padded_sentence = torch.tensor(padded_sentence)
        attention_mask = torch.tensor(attention_mask)

        # return tokenized_sentence, padded_sentence, attention_mask
        return padded_sentence, attention_mask

    def encode_sentence(self, sentence, max_len):
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tokenized_sentence, ids_sentence = self.tokenize_text(sentence)
        tokenized_sentence = np.asarray(tokenized_sentence)

        # Padding: fill up with 0's for paddings
        padded_sentence = np.asarray(ids_sentence + [0] * (max_len - len(ids_sentence)))

        # Masking: Set attention 1 for id's and 0's for paddings
        attention_mask = np.where(padded_sentence != 0, 1, 0)

        # Transform to Tensors
        padded_sentence = torch.tensor(padded_sentence)
        attention_mask = torch.tensor(attention_mask)

        # return tokenized_sentence, padded_sentence, attention_mask
        return padded_sentence, attention_mask
