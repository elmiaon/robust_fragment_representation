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

def CLSR_tokenize(setting_code, corpus, sub_corpus, s, t, chunksize=10000):
    _, tokenize_method, _, _, _ = utils.get_RFR_CLSR_setting(setting_code)
    logger = log.get_logger(__name__)

    # check the reformatted dataest
    if os.path.isfile(f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{s}.csv") and \
       os.path.isfile(f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{t}.csv"):
        input_dir = {
            s: f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{s}.csv",
            t: f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{t}.csv"
        }
    elif os.path.isfile(f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{s}.csv") and \
         os.path.isfile(f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{t}.csv"):
        input_dir = {
            s: f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{s}.csv",
            t: f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{t}.csv"
        }
    else:
        raise FileExistsError(f"There is no prepared {corpus}-{sub_corpus}")

    # define output directory
    utils.make_dir(f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}")
    output_dir_fwd = {
        s: f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{s}-{t}.{s}.csv",
        t: f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{s}-{t}.{t}.csv"
    }

    output_dir_bwd = {
        s: f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{t}-{s}.{s}.csv",
        t: f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{t}-{s}.{t}.csv"
    }

    # check output to skip
    fwd_check = os.path.isfile(output_dir_fwd[s]) and os.path.isfile(output_dir_fwd[t])
    bwd_check = os.path.isfile(output_dir_bwd[s]) and os.path.isfile(output_dir_bwd[t])

    if not (fwd_check or bwd_check):
        
        if tokenize_method == 'RFR':
            tokenizer = RFR_tokenize
        else:
            raise ValueError(f"invalid tokenizer")

        for lang in [s, t]:
            in_dir = input_dir[lang]
            out_dir = output_dir_fwd[lang]

            if not os.path.isfile(out_dir):
                tokenized_list = []

                for idx_chunk, chunk in enumerate(pd.read_csv(in_dir, sep='\t', chunksize=chunksize)):
                    tokenized_list.append(tokenizer(chunk))
                    print(f"finish {s}-{t}.{lang} part {idx_chunk+1} tokenization")
                    
                df = pd.concat(tokenized_list, ignore_index=True)
                df.to_csv(out_dir, sep='\t', index=False)
            
                logger.info(f"finish {corpus}-{sub_corpus}.{s}-{t}.{lang} tokenzation")
                logger.info(f"sample:\n{df}")
            
            else:
                logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t}.{lang} tokenzation")
    else:
        logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t} tokenization")

def RFR_tokenize(df):
    sid, lang = df.columns
    df[lang] = df[lang].parallel_apply(RFR_tokenize_sentence, args=(lang,))
    df = df.dropna()
    return df

def RFR_tokenize_sentence(sentence, lang):
    global punc, punc2, double_space
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
        if len(sentence)==0:
            return None

    if len(sentence) == 1 and sentence[0]=='':
        return None
    else:
        return sentence