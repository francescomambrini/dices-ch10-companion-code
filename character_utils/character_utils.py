import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfTransformer
import os
from collections import defaultdict
import pickle
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


_dir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(_dir, '../data')

df_il = pd.read_csv(os.path.join(datadir, 'full_char_matrix_il.csv'), index_col=0)
df_od = pd.read_csv(os.path.join(datadir, 'full_char_matrix_od.csv'), index_col=0)
df_gen = pd.read_csv(os.path.join(datadir, 'full_char_matrix_general.csv'), 
                     index_col=0)

def load_sents(path):
    import conllu

    """
    Load the CoNLLU file with the direct-speech sentences 
    """

    with open(path, 'r') as f:
        sents = conllu.parse(f.read())
    return sents

def create_normalized_per1000_matrix(df, nr_chr, custom_word_list):
    transformer = TfidfTransformer(norm='l1',
                               use_idf=False,
                               smooth_idf=False
                              )
    X = transformer.fit_transform(df) * 1000
    normdf = pd.DataFrame(X.toarray(), columns=df.columns, 
                         index=df.index)
    char_freqs = df.sum(axis=1).sort_values(ascending=False)
    top_char = char_freqs.head(nr_chr).index
    return normdf.loc[top_char, custom_word_list]

def create_zsc_matrix(df, nr_chr, custom_word_list):
    transformer = TfidfTransformer(norm='l1',
                               use_idf=False,
                               smooth_idf=False
                              )
    X = transformer.fit_transform(df)
    normdf = pd.DataFrame(X.toarray(), columns=df.columns, 
                         index=df.index)
    char_freqs = df.sum(axis=1).sort_values(ascending=False)
    top_char = char_freqs.head(nr_chr).index
    zdf1k = pd.DataFrame(scale(normdf, with_mean=True), index=normdf.index, 
                     columns=normdf.columns)
    
    zdf = zdf1k.loc[top_char, custom_word_list]
    return zdf

def distance_matrix(zdf):
    distances = pdist(zdf.values, metric="cityblock") / len(zdf.columns)
    sqdistmat = squareform(distances)
    distdf = pd.DataFrame(sqdistmat, index=zdf.index, columns=zdf.index)
    return distdf

def correlation_matrix(df, nr_chr, top_word_list):
    transformer = TfidfTransformer(norm='l1',
                               use_idf=False,
                               smooth_idf=False
                              )
    X = transformer.fit_transform(df)
    normdf = pd.DataFrame(X.toarray(), columns=df.columns, 
                         index=df.index)
    char_freqs = df.sum(axis=1).sort_values(ascending=False)
    top_char = char_freqs.head(nr_chr).index
    top_df = normdf.loc[top_char]
    top_df = top_df[top_word_list]
    corrmat = top_df.transpose().corr()
    return corrmat

def create_speech_dic(sent_list):
    """
    Create a defaultdict with character as key 
    and a list of sentences as value
    """
    char_dic = defaultdict(list)  
    for s in sent_list:
        spk = s.metadata['Speaker']
        char_dic[spk].append(s)
    return char_dic

def do_pca(df):
    pca = PCA(n_components=2)
    pca_corr_mat = pca.fit_transform(df)
    pca_res = pd.DataFrame(data = pca_corr_mat, columns = ['vector 1', 'vector 2'])#, 'vector 3'])#, 'vector 4'])
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    return pca_res

def plot_pca(pca_res_df, char_df):
    plt.figure()
    plt.figure(figsize=(12,10))
    plt.axes(xlabel='Vector 1', ylabel='Vector 2')

    for i in pca_res_df.index:
        x, y = pca_res_df.loc[i]["vector 1"] * -1, pca_res_df.loc[i]["vector 2"]
        plt.scatter(x, y, s=50)
        word = char_df.iloc[i].name
        plt.text(x+.005, y+.005, word, fontsize=16)
    plt.show()

def print_occurrence(sent_list, lemma, basecolor='grey'):
    from termcolor import colored

    for s in sent_list:
        matches = [i for i,t in enumerate(s) if t['lemma'] == lemma]
        last_index = len(s) -1
        if matches:
            print(colored(str(s[matches[0]]['misc']['Ref']), 'green'), end=" ")
            for i, tok in enumerate(s):
                end = "\n" if i == last_index else " "
                c = basecolor if i not in matches else 'blue'
                print(colored(tok['form'], c), end=end)

def plot_words(zdf, words, width=0.3, outfig=None, dpi=300):
    fig, ax = plt.subplots(figsize=(20, 10))
    N = len(zdf.index)
    ind = np.arange(N)    # the x locations for the groups
    width = 0.3         # the width of the bars
    ind = np.arange(N)    # the x locations for the groups
    # the width of the bars
    ps = []
    for i, w in enumerate(words):
        offset = width * i
        ps.append(ax.bar(ind + offset, zdf[w], width))
    # p1 = ax.bar(ind, zdf.loc[words[0]], width)
    # p2 = ax.bar(ind + width, filt.loc[char2], width)

    ax.legend([p for p in ps], words)
    ax.set_title(f'Z-scores for {", ".join(words)}')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels([i for i in zdf.index])


    ax.autoscale_view()

    plt.xticks(rotation=45)
    if outfig:
        plt.savefig(outfig, dpi=dpi, facecolor='white')
    
    plt.show()
    


def plot_chars(zdf, char1, char2, lemmas, outfig=None, dpi=300):
    lemmas = '|'.join([str(l) for l in lemmas])
    filt = zdf.filter(regex=lemmas)
    N = len(filt.columns)

    fig, ax = plt.subplots(figsize=(20, 10))

    ind = np.arange(N)    # the x locations for the groups
    width = 0.3         # the width of the bars
    p1 = ax.bar(ind, filt.loc[char1], width)
    p2 = ax.bar(ind + width, filt.loc[char2], width)

    ax.legend((p1[0], p2[0]), (char1, char2))
    nr_lemma = len(lemmas.split('|'))
    ax.set_title(f'Scores for the top {nr_lemma} most frequent lemmas grouped by character')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels([l.split('_')[0] for l in filt.columns])


    ax.autoscale_view()

    plt.xticks(rotation=45)
    if outfig:
        plt.savefig(outfig, dpi=dpi, facecolor='white')

    plt.show()
    
