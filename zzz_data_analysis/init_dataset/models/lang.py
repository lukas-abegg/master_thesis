import os
import codecs

import pandas as pd


class Lang(object):
    def __init__(self, name, tokenizer):
        self.name = name
        self.tokenizer = tokenizer
        self.allowed_max_length = 512
        self.word_2_count = {}
        self.n_words = 0
        self.n_words_unique = 0  # Count CLS and SEP
        self.max_length = 0
        self.max_length_words = 0
        self.min_length = 100
        self.min_length_words = 100
        self.df_tokenized_texts = pd.DataFrame(columns=["index", "text"])
        self.df_unknown = pd.DataFrame(columns=["index", "text"])

    def tokenize_sentence(self, sentence, index):
        tokenized_text, ids_encoded = self.tokenizer.tokenize_text(sentence)
        tokenized_into_words = self.tokenizer.tokenize_into_words(sentence)

        self.df_tokenized_texts.loc[len(self.df_tokenized_texts)] = [index, tokenized_text]
        return tokenized_text, tokenized_into_words

    def filter_sentence(self, tokenized_text):
        return len(tokenized_text) <= self.allowed_max_length

    def add_sentence(self, sentence, index):
        tokenized_text, tokenized_into_words = self.tokenize_sentence(sentence, index)

        allowed_length = self.filter_sentence(tokenized_text)

        if allowed_length:
            if "[UNK]" in tokenized_text:
                self.df_unknown.loc[len(self.df_unknown)] = [index, tokenized_text]

            for word in tokenized_text:
                if word not in ["[CLS]", "[SEP]"]:
                    self.add_word(word)

            self.max_length = max(self.max_length, len(tokenized_text))
            self.max_length_words = max(self.max_length_words, len(tokenized_into_words))
            self.min_length = min(self.min_length, len(tokenized_text))
            self.min_length_words = min(self.min_length_words, len(tokenized_into_words))

        return allowed_length

    def add_word(self, word):
        self.n_words += 1

        if word not in self.word_2_count:
            self.word_2_count[word] = 1
            self.n_words_unique += 1
        else:
            self.word_2_count[word] += 1

    def write_unknowns_to_file(self, filepath, filename_base):
        delimiter = '\t'
        # Unescape the delimiter
        delimiter = str(codecs.decode(delimiter, "unicode_escape"))

        print("{} sentences found with unknown ([UNK]) words.".format(len(self.df_unknown)))
        self.df_unknown.to_csv(os.path.join(filepath, filename_base+"unknown.txt"), sep=delimiter, encoding='utf-8', index=False)

    def get_unknown(self):

        def get_val(val):
            found_sum = 0
            for key, value in self.word_2_count.items():
                if key.startswith(val):
                    found_sum = found_sum + value

            print("Key with value {}: {} times found.".format(val, found_sum))

        print("Key '[UNK]':")
        get_val("[UNK]")
        print("Key '##':")
        get_val("##")
        print("Key '#':")
        get_val("#")
