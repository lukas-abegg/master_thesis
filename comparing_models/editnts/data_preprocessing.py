import numpy as np
import pandas as pd
import data
from nltk import pos_tag
from comparing_models.editnts.label_edits import sent2edit


def remove_lrb(sent_string):
    # sent_string = sent_string.lower()
    frac_list = sent_string.split('-lrb-')
    clean_list = []
    for phrase in frac_list:
        if '-rrb-' in phrase:
            clean_list.append(phrase.split('-rrb-')[1].strip())
        else:
            clean_list.append(phrase.strip())
    clean_sent_string = ' '.join(clean_list)
    return clean_sent_string


def replace_lrb(sent_string):
    sent_string = sent_string.lower()
    # new_sent= sent_string.replace('-lrb-','(').replace('-rrb-',')')
    new_sent = sent_string.replace('-lrb-', '').replace('-rrb-', '')
    return new_sent


def process_raw_data(comp_txt, simp_txt):
    comp_txt = [line.lower().split() for line in comp_txt]
    simp_txt = [line.lower().split() for line in simp_txt]
    # df_comp = pd.read_csv('data/%s_comp.csv'%dataset,  sep='\t')
    # df_simp= pd.read_csv('data/%s_simp.csv'%dataset,  sep='\t')
    assert len(comp_txt) == len(simp_txt)
    df = pd.DataFrame(
        {'comp_tokens': comp_txt,
         'simp_tokens': simp_txt,
         })

    def add_edits(df):
        """
        :param df: a Dataframe at least contains columns of ['comp_tokens', 'simp_tokens']
        :return: df: a df with an extra column of target edit operations
        """
        comp_sentences = df['comp_tokens'].tolist()
        simp_sentences = df['simp_tokens'].tolist()
        pair_sentences = list(zip(comp_sentences, simp_sentences))

        edits_list = [sent2edit(l[0], l[1]) for l in
                      pair_sentences]  # transform to edits based on comp_tokens and simp_tokens
        df['edit_labels'] = edits_list
        return df

    def add_pos(df, unk):
        src_sentences = df['comp_tokens'].tolist()
        pos_sentences = [pos_tag(sent) for sent in src_sentences]
        df['comp_pos_tags'] = pos_sentences

        pos_vocab = data.POSvocab()
        pos_ids_list = []
        for sent in pos_sentences:
            pos_ids = [pos_vocab.w2i[w[1]] if w[1] in pos_vocab.w2i.keys() else pos_vocab.w2i[unk] for w in sent]
            pos_ids_list.append(pos_ids)
        df['comp_pos_ids'] = pos_ids_list
        return df

    df = add_pos(df)
    df = add_edits(df)
    return df


def editnet_data_to_editnetID(df, output_path):
    """
    this function reads from df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    and add vocab ids for comp_tokens, simp_tokens, and edit_labels
    :param df: df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    :param output_path: the path to store the df
    :return: a dataframe with df.columns=['comp_tokens', 'simp_tokens', 'edit_labels',
                                            'comp_ids','simp_id','edit_ids',
                                            'comp_pos_tags','comp_pos_ids'])
    """
    out_list = []
    vocab = data.Vocab()
    vocab.add_vocab_from_file('./vocab_data/vocab.txt', 30000)

    def prepare_example(example, vocab, unk):
        """
        :param example: one row in pandas dataframe with feild ['comp_tokens', 'simp_tokens', 'edit_labels']
        :param vocab: vocab object for translation
        :return: inp: original input sentence,
        """
        comp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[unk] for i in example['comp_tokens']])
        simp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[unk] for i in example['simp_tokens']])
        edit_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[unk] for i in example['edit_labels']])
        return comp_id, simp_id, edit_id  # add a dimension for batch, batch_size =1

    for i, example in df.iterrows():
        print(i)
        comp_id, simp_id, edit_id = prepare_example(example, vocab)
        ex = [example['comp_tokens'], comp_id,
              example['simp_tokens'], simp_id,
              example['edit_labels'], edit_id,
              example['comp_pos_tags'], example['comp_pos_ids']
              ]
        out_list.append(ex)
    outdf = pd.DataFrame(out_list, columns=['comp_tokens', 'comp_ids', 'simp_tokens', 'simp_ids',
                                            'edit_labels', 'new_edit_ids', 'comp_pos_tags', 'comp_pos_ids'])
    outdf.to_pickle(output_path)
    print('saved to %s' % output_path)
