import os
from argparse import ArgumentParser

import pandas as pd
import codecs
import json
import sys

from layers.utils.bert_loader import BertModelLoader
from zzz_data_analysis.init_dataset.models.lang import Lang
from zzz_data_analysis.init_dataset.preprocessing import cleanup


# trim and unicode transformation
from zzz_data_analysis.init_dataset.preprocessing.bert_tokenizing import BertPreprocessor


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


def read_file_health(datapath):
    print("Reading lines...")
    # Read the file and split into lines
    return pd.read_csv(datapath, delimiter="\t", header=None, encoding="utf-8", names=["input", "output"])


def read_file_wiki(datapath):
    print("Reading lines...")
    # Read the file and split into lines
    data = pd.read_csv(datapath, delimiter="\t", header=None, encoding="utf-8", names=["input", "output", "matching"])
    data = data.drop("matching", axis=1)
    return data


def drop_duplicates(data):
    return data.drop_duplicates()


def prepare_data(data, tokenizer, config):
    data_to_prepare = drop_duplicates(data)
    print("Read %s sentence pairs" % len(data_to_prepare))
    return get_langs(data_to_prepare, tokenizer, config)


def load_config(bert_model):
    with open(os.path.join("..", "configs", "init_wiki_simple.json"), "r") as config_file:
        config = json.load(config_file)
    return config[bert_model]


def print_dataset_details(input_lang, output_lang, wiki_simple_path, filename_base):
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
    input_lang.write_unknowns_to_file(wiki_simple_path, filename_base + "_input_")
    print("Check for unknown keys in Output")
    output_lang.get_unknown()
    output_lang.write_unknowns_to_file(wiki_simple_path, filename_base + "_output_")


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


def write_preprocessed_data_to_file(data, wiki_simple_path, filename_base):
    delimiter = "\t"
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    datafile = os.path.join(wiki_simple_path, "preprocessed_data", filename_base + ".txt")

    print("Writing newly formatted file.")
    data.to_csv(datafile, sep=delimiter, encoding="utf-8", index=False)


def write_dataset_config_to_file(data, input_lang, output_lang, wiki_simple_path, filename_base, dataset_name,
                                 bert_model, config):
    datafile = os.path.join(wiki_simple_path, "preprocessed_data", filename_base + ".json")
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
    """ This programm analyses the wiki_simple corpus data with the given bert model and its tokenizer and writes as output:
    " - preprocessed src and dst files
    " - a json configs file with the information about the data (e.g., wiki_f_health_bert_base.json)
    " - a file with the unknow words for the input and output sentences (e.g., wiki_f_health_bert_base_input_unknown.txt and wiki_f_health_bert_base_output_unknown.txt)
    """

    parser = ArgumentParser(description='Train Transformer')
    parser.add_argument('--bert_model', type=str, default=None)
    args = parser.parse_args()

    if args.config is not None:
        bert_model = args.config
    else:
        raise ValueError('Input arguments are missing')

    config = load_config(bert_model)
    tokenizer = BertPreprocessor(BertModelLoader(bert_model, "../../").tokenizer)

    wiki_simple_path = config["wiki_simple_path"]

    print("Load Dataset...")
    print("Expert fully aligned medical subset:")
    print("------------------------------")

    wiki_f_health = config["wiki_f_health"]
    filepath_to_load = os.path.join(wiki_simple_path, wiki_f_health)

    input_lang, output_lang, data = prepare_data(read_file_health(filepath_to_load), tokenizer, config)
    print_dataset_details(input_lang, output_lang, wiki_simple_path, "wiki_f_health_" + bert_model)
    encode_sentence_samples(tokenizer, data.loc[:5, "input"], input_lang.max_length)
    filtered_data = filter_data_after_preprocessing(data, input_lang, output_lang)
    write_preprocessed_data_to_file(filtered_data, wiki_simple_path, "wiki_f_health_" + bert_model)
    write_dataset_config_to_file(filtered_data, input_lang, output_lang, wiki_simple_path,
                                 "wiki_f_health_" + bert_model, "wiki_simple", bert_model, config)

    print("Dataset processed.")

    print("Automatically fully aligned f_wiki:")
    print("------------------------------")

    wiki_f_wiki = config["wiki_f_wiki"]
    filepath_to_load = os.path.join(wiki_simple_path, wiki_f_wiki)

    input_lang, output_lang, data = prepare_data(read_file_wiki(filepath_to_load), tokenizer, config)
    print_dataset_details(input_lang, output_lang, wiki_simple_path, "wiki_f_wiki_" + bert_model)
    encode_sentence_samples(tokenizer, data.loc[:5, "input"], input_lang.max_length)
    filtered_data = filter_data_after_preprocessing(data, input_lang, output_lang)
    write_preprocessed_data_to_file(filtered_data, wiki_simple_path, "wiki_f_wiki_" + bert_model)
    write_dataset_config_to_file(filtered_data, input_lang, output_lang, wiki_simple_path, "wiki_f_wiki_" + bert_model,
                                 "wiki_simple", bert_model, config)

    print("Dataset processed.")

    sys.exit()
