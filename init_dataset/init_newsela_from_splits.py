import os
import pandas as pd
import codecs
import json
import sys

from loading_models.bert_loader import BertModelLoader
from init_dataset.models.lang import Lang
from init_dataset.preprocessing import cleanup
from preprocessings.bert_tokenizing import BertPreprocessor


# trim and unicode transformation
def normalize_sentence(s, config):
    do_lower_case = bool(config["do_lower_case"])
    remove_brackets_with_inside = bool(config["remove_brackets_with_inside"])
    apply_regex_rules = bool(config["apply_regex_rules"])
    s = cleanup.normalize_sentence(s, do_lower_case, remove_brackets_with_inside, apply_regex_rules)
    return s


# load input and output into own Lang class to get some general information about them
def get_langs(data, tokenizer, config):
    lang_simple = "simple"
    lang_domain = "domain"

    if bool(config["normalize_text"]):
        # Split every line into pairs and normalize
        data["input"] = data["input"].apply(lambda x: normalize_sentence(x, config))

    input_lang = Lang(lang_simple, tokenizer)
    output_lang = Lang(lang_domain, tokenizer)

    print("Counting words...")

    trimmed_data = data.loc[data.apply(
        lambda x: input_lang.add_sentence(x["input"], x.name) & output_lang.add_sentence(x["output"], x.name),
        axis=1)]

    return input_lang, output_lang, trimmed_data


def read_file(filepath_to_load_train_src, filepath_to_load_train_dst, filepath_to_load_valid_src, filepath_to_load_valid_dst, filepath_to_load_test_src, filepath_to_load_test_dst):
    print("Reading lines...")
    # Read the file and split into lines
    df_newsela_train_src = pd.read_csv(filepath_to_load_train_src, delimiter="\t", header=None, encoding="utf-8", names=["input"])
    df_newsela_train_dst = pd.read_csv(filepath_to_load_train_dst, delimiter="\t", header=None, encoding="utf-8", names=["output"])
    df_newsela_train = pd.concat([df_newsela_train_src, df_newsela_train_dst], axis=1, sort=False)

    df_newsela_valid_src = pd.read_csv(filepath_to_load_valid_src, delimiter="\t", header=None, encoding="utf-8", names=["input"])
    df_newsela_valid_dst = pd.read_csv(filepath_to_load_valid_dst, delimiter="\t", header=None, encoding="utf-8", names=["output"])
    df_newsela_valid = pd.concat([df_newsela_valid_src, df_newsela_valid_dst], axis=1, sort=False)

    df_newsela_test_src = pd.read_csv(filepath_to_load_test_src, delimiter="\t", header=None, encoding="utf-8", names=["input"])
    df_newsela_test_dst = pd.read_csv(filepath_to_load_test_dst, delimiter="\t", header=None, encoding="utf-8", names=["output"])
    df_newsela_test = pd.concat([df_newsela_test_src, df_newsela_test_dst], axis=1, sort=False)

    frames = [df_newsela_train, df_newsela_valid, df_newsela_test]
    dataset_merged = pd.concat(frames)

    return df_newsela_train, df_newsela_valid, df_newsela_test, dataset_merged


def drop_duplicates(data):
    return data.drop_duplicates()


def prepare_data(data, tokenizer, config):
    data_to_prepare = drop_duplicates(data)
    print("Read %s sentence pairs" % len(data_to_prepare))
    return get_langs(data_to_prepare, tokenizer, config)


def load_config(bert_model):
    with open(os.path.join("..", "configs", "init_newsela_from_splits.json"), "r") as config_file:
        config = json.load(config_file)
    return config[bert_model]


def print_dataset_details(input_lang, output_lang, newsela_path, filename_base):
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    print("Shortest tokenized sentence in Input: {} words, Output: {} words".format(input_lang.min_length,
                                                                                    output_lang.min_length))
    print("Shortest word-split sentence in Input: {} words, Output: {} words".format(input_lang.min_length_words,
                                                                                     output_lang.min_length_words))
    print("Longest tokenized sentence in Input: {} words, Output: {} words".format(input_lang.max_length,
                                                                                   output_lang.max_length))
    print("Longest word-split sentence in Input: {} words, Output: {} words".format(input_lang.max_length_words,
                                                                                    output_lang.max_length_words))

    print("Check for unknown keys in Input")
    input_lang.get_unknown()
    input_lang.write_unknowns_to_file(newsela_path, filename_base + "_input_")
    print("Check for unknown keys in Output")
    output_lang.get_unknown()
    output_lang.write_unknowns_to_file(newsela_path, filename_base + "_output_")


def encode_sentence_samples(tokenizer, sentences, max_length):
    print("Encode sample sentences with tokenizer:")
    tokenized_sentences, padded_sentences, attention_masks = tokenizer.encode_sentences(sentences, max_len=max_length)

    print("Shapes of attention mask:", attention_masks.shape)

    # Print sentence 0, now as a list of IDs.
    print("Original: ", sentences[0])
    print("Token IDs:", tokenized_sentences[0])
    print("Padding Mask:", padded_sentences[0])
    print("Attention Mask:", attention_masks[0])


def filter_data_after_preprocessing(data, input_lang, output_lang):
    return data[~data.index.isin(input_lang.df_unknown.index) & ~data.index.isin(output_lang.df_unknown.index)]


def write_preprocessed_data_to_file(train, validation, test, newsela_path, filename_base):
    write_to_file(train, newsela_path, "train_" + filename_base)
    write_to_file(validation, newsela_path, "valid_" + filename_base)
    write_to_file(test, newsela_path, "test_" + filename_base)


def write_to_file(data, newsela_path, filename_base):
    datafile = os.path.join(newsela_path, "preprocessed_data", filename_base + ".txt")

    print("Writing file: {} with {} lines.".format(datafile, len(data)))
    data.to_csv(datafile + ".src", encoding='utf-8', columns=["input"], index=False, header=False)
    data.to_csv(datafile + ".dst", encoding='utf-8', columns=["output"], index=False, header=False)


def write_dataset_config_to_file(data, input_lang, output_lang, newsela_path, filename_base, dataset_name,
                                 bert_model, config):
    datafile = os.path.join(newsela_path, "preprocessed_data", filename_base + ".json")
    data = {
        dataset_name: {
            "bert_model": bert_model,
            "length_dataset": data.shape[0],
            "max_length_tokens_input": input_lang.max_length,
            "max_length_words_input": input_lang.max_length_words,
            "min_length_tokens_input": input_lang.min_length,
            "min_length_words_input": input_lang.min_length_words,
            "max_length_tokens_output": output_lang.max_length,
            "max_length_words_output": output_lang.max_length_words,
            "min_length_tokens_output": output_lang.min_length,
            "min_length_words_output": output_lang.min_length_words,
            "dataset_normalized": bool(config["normalize_text"])
        }
    }

    with open(datafile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    bert_model = "bio_bert"
    config = load_config(bert_model)
    tokenizer = BertPreprocessor(BertModelLoader(bert_model, "../").tokenizer)

    newsela_path = config["newsela_path"]

    print("Load Dataset...")
    print("Newsela Corpus:")
    print("------------------------------")

    newsela_data_train_src = config["newsela_data_train_src"]
    newsela_data_train_dst = config["newsela_data_train_dst"]
    newsela_data_valid_src = config["newsela_data_valid_src"]
    newsela_data_valid_dst = config["newsela_data_valid_dst"]
    newsela_data_test_src = config["newsela_data_test_src"]
    newsela_data_test_dst = config["newsela_data_test_dst"]

    filepath_to_load_train_src = os.path.join(newsela_path, newsela_data_train_src)
    filepath_to_load_train_dst = os.path.join(newsela_path, newsela_data_train_dst)
    filepath_to_load_valid_src = os.path.join(newsela_path, newsela_data_valid_src)
    filepath_to_load_valid_dst = os.path.join(newsela_path, newsela_data_valid_dst)
    filepath_to_load_test_src = os.path.join(newsela_path, newsela_data_test_src)
    filepath_to_load_test_dst = os.path.join(newsela_path, newsela_data_test_dst)

    dataset_train, dataset_valid, dataset_test, dataset_merged = read_file(filepath_to_load_train_src, filepath_to_load_train_dst, filepath_to_load_valid_src, filepath_to_load_valid_dst, filepath_to_load_test_src, filepath_to_load_test_dst)

    input_lang, output_lang, data = prepare_data(dataset_merged, tokenizer, config)
    print_dataset_details(input_lang, output_lang, newsela_path, "newsela_" + bert_model)
    encode_sentence_samples(tokenizer, data.loc[:5, "input"], input_lang.max_length)
    filtered_data = filter_data_after_preprocessing(data, input_lang, output_lang)
    write_preprocessed_data_to_file(dataset_train, dataset_valid, dataset_test, newsela_path, "newsela_" + bert_model)
    write_dataset_config_to_file(filtered_data, input_lang, output_lang, newsela_path,
                                   "newsela_" + bert_model, "newsela", bert_model, config)

    print("Dataset processed.")

    sys.exit()
