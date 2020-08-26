from preprocessings.bert_tokenizing import BertPreprocessor


class Vocab:
    def __init__(self, tokenizer, split_words):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0
        self.split_words = split_words
        self.tokenizer: BertPreprocessor = tokenizer

    def add_sentence(self, sentence):
        if self.split_words:
            for word, _ in self.tokenizer.tokenize_text(sentence):
                self.add_word(word)
        else:
            for word in self.tokenizer.tokenize_into_words(sentence):
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
