# import basic lib
import json
import numpy as np
import os
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(use_memory_fs=False)
import string
import sys
# import logging lib
import logging
import src.log as log
# import time lib
from time import time
#import nlp lib
import re
from pythainlp.tokenize import word_tokenize
# import custom lib
import src.utils as utils

punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.«»"
punc2 = string.punctuation
double_space = " ​ "

def tokenize(args, root='data'):
    corpus, sub_corpus, s, t = args
    _, method, w2v, n_grams, fragment_size, sentence_len, s2v, s2v_args_list, _, _, _, _, _, _ = utils.get_experiment_setting(exp_code)
    logger = log.get_logger(__name__)
    corpus_dir = f"{root}/tokenized/{corpus}/{sub_corpus}/{tokenize_method}"
    utils.make_dir(corpus_dir)
    s_fwd_output_dir = f"{root}/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{s}-{t}.{s}.csv"
    t_fwd_output_dir = f"{root}/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{s}-{t}.{t}.csv"
    g_fwd_output_dir = f"{root}/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{s}-{t}.gold.csv"

    s_bwd_output_dir = f"{root}/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{t}-{s}.{s}.csv"
    t_bwd_output_dir = f"{root}/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{t}-{s}.{t}.csv"
    g_bwd_output_dir = f"{root}/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{t}-{s}.gold.csv"

    s_check = os.path.isfile(s_fwd_output_dir) or os.path.isfile(s_bwd_output_dir)
    t_check = os.path.isfile(t_fwd_output_dir) or os.path.isfile(t_bwd_output_dir)
    g_check = os.path.isfile(g_fwd_output_dir) or os.path.isfile(g_bwd_output_dir)

    if not (s_check and t_check and g_check):
        tic = time()
        sdf, tdf, gdf = load_raw(corpus, sub_corpus, s, t)
        toc = time()
        logger.info(f"loading raw data from {corpus} - {sub_corpus} in {toc-tic:.2f} second(s)")

        if not s_check:
            tic = time()
            sdf = sdf.parallel_apply(tokenize_sentence, axis=1)
            sdf = sdf.dropna()
            sdf.to_csv(s_fwd_output_dir, index=False, sep='\t')
            toc = time()
            logger.info(f"prepare {s} dataset in {toc-tic:.2f} second(s)")
            logger.info(f"sample: {sdf}")
        
        if not t_check:
            tic = time()
            tdf = tdf.parallel_apply(tokenize_sentence, axis=1)
            tdf = tdf.dropna()
            tdf.to_csv(t_fwd_output_dir, index=False, sep = '\t')
            toc = time()
            logger.info(f"prepare {t} dataset in {toc-tic:.2f} second(s)")
            logger.info(f"sample: {tdf}")
        
        if not g_check:
            tic = time()
            gdf = gdf.loc[gdf[s].isin(sdf['id'].values) & gdf[t].isin(tdf['id'].values)]
            gdf.to_csv(g_fwd_output_dir, index=False, sep='\t')
            toc = time()
            logger.info(f"prepare gold in {toc-tic:.2f} second(s)")
            logger.info(f"sample: {gdf}")


def tokenize_sentence(row):
    global punc, punc2, double_space
    sid, lang = row.index
    sentence = row.pop(lang)
    sentence = sentence.strip('\n')
    sentence = sentence.lower()
    # remove punctuation
    sentence = re.sub(r"[%s%s]+"%(punc, punc2), '', sentence)
    sentence = sentence.strip(' ')
    sentence = re.sub(r"[%s]+"%(double_space), ' ', sentence)

    if not lang in ['th']:
        sentence = sentence.split(' ')
    else:
        sentence= word_tokenize(sentence, engine='newmm')
        sentence = [i for i in sentence if i != ' ']
        if len(sentence) == 0:
            row[sid], row[lang] = None, None

    if len(sentence) == 1 and sentence[0]=='':
        row[sid], row[lang] = None, None
    else:
        row[lang] = sentence
    return row