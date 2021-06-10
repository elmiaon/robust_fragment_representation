##################################################
### import                                     ###
##################################################
# basic lib
import json
import numpy as np
import os
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(use_memory_fs=False)
import string
# logging lib
import logging
import src.log as log
# time lib
from time import time
# nlp lib
import re
from pythainlp.tokenize import word_tokenize
# custom lib
import src.utils as utils

##################################################
### define global                              ###
##################################################
punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.«»" # special punctuation especially in Japanese
punc2 = string.punctuation # common punctuation
double_space = " ​ " # double space in Thai


##################################################
### CLSR_tokenize                              ###
##################################################
def CLSR_tokenize(setting_code:int, corpus:str, sub_corpus:str, s:str, t:str, chunksize=10000:int):
    '''
    tokenize sentences in the corpus
    input : 
        setting_code(int) - setting_code to get the experiment parameter
        corpus(str), sub_corpus(str) - corpus and sub corpus to be tokenzied
        s(str), t(str) - source and target language to be tokenized, respectively
    output: tokenized dataset files(csv) - saved in data/tokenized/ directory
    '''
    _, tokenize_method, _, _, _ = utils.get_RFR_CLSR_setting(setting_code)
    logger = log.get_logger(__name__)

    # check the reformatted dataset to be tokenized
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
    else: # error if there is no reformatted file
        raise FileExistsError(f"There is no reformatted {corpus}-{sub_corpus}")

    # define output directory
    utils.make_dir(f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}")
    output_dir_fwd = { # output from {s}->{t}
        s: f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{s}-{t}.{s}.csv",
        t: f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{s}-{t}.{t}.csv"
    }

    output_dir_bwd = { # output from {t}->{s}
        s: f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{t}-{s}.{s}.csv",
        t: f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}/{t}-{s}.{t}.csv"
    }

    # check output to skip
    fwd_check = os.path.isfile(output_dir_fwd[s]) and os.path.isfile(output_dir_fwd[t])
    bwd_check = os.path.isfile(output_dir_bwd[s]) and os.path.isfile(output_dir_bwd[t])

    if not (fwd_check or bwd_check):
        
        # select tokenizer
        if tokenize_method == 'RFR':
            tokenizer = RFR_tokenize
        else:
            raise ValueError(f"invalid tokenizer")

        # tokenize {s} and {t} corpus
        for lang in [s, t]:
            in_dir = input_dir[lang] # define input dir for the specific language
            out_dir = output_dir_fwd[lang] # define output dir for the specific language

            if not os.path.isfile(out_dir):
                tokenized_list = [] # define to store tokenized chunk

                for idx_chunk, chunk in enumerate(pd.read_csv(in_dir, sep='\t', chunksize=chunksize)): # read csv chunk-by-chunk
                    tokenized_list.append(tokenizer(chunk)) # tokenize each chunk
                    print(f"finish {s}-{t}.{lang} part {idx_chunk+1} tokenization") # print to tell the status of each chunk
                    
                df = pd.concat(tokenized_list, ignore_index=True) # concatenate all chunk
                df.to_csv(out_dir, sep='\t', index=False) # save the tokenized
            
                logger.info(f"finish {corpus}-{sub_corpus}.{s}-{t}.{lang} tokenzation") # log the time
                logger.info(f"sample:\n{df}") # show some sample
            
            else:
                logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t}.{lang} tokenzation") # the tokenized step for specific language is skipped
    else:
        logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t} tokenization") # the tokenized step is skipped

##################################################
### tokenize method                            ###
##################################################

### RFR ###
def RFR_tokenize(df:pd.dataFrame):
    '''
    RFR tokenize for input dataframe
    input : df(dataframe) - dataframe to be tokenized with [id, {language}] as columns
    output: tokenized dataframe(df) - tokenized dataframe with dropped Nan
    '''
    sid, lang = df.columns # get each column to indicate the input language
    df[lang] = df[lang].parallel_apply(RFR_tokenize_sentence, args=(lang,)) # tokenize each sentence
    df = df.dropna() # drop Nan row
    return df

def RFR_tokenize_sentence(sentence:str, lang:str):
    '''
    RFR tokenize for input sentence
    input : 
        sentence(str) - sentence to be tokenized
        lang(str) - language indicator
    output: tokenized sentences(list) - tokenized dataframe with dropped Nan
    '''
    global punc, punc2, double_space # define global varible in the function
    sentence = sentence.strip('\n') # strip new line
    sentence = sentence.lower() # lower the characters in the setnence
    # remove punctuation
    sentence = re.sub(r"[%s%s]+"%(punc, punc2), '', sentence) # remove punc, punc2
    sentence = sentence.strip(' ') # remove puncs probably create some space at the start and end of sentence
    sentence = re.sub(r"[%s]+"%(double_space), ' ', sentence) # replace double space with single space

    if lang == 'th' # use special tokenizer for Thai
        sentence= word_tokenize(sentence, engine='newmm')
        sentence = [i for i in sentence if i != ' ']
        if len(sentence)==0:
            return None
    else: # for other languages tokenize using whitespace
        sentence = sentence.split(' ')
    
    if len(sentence) == 1 and sentence[0]=='': # remove the empty tokenized sentence
        return None
    else: # return the normal tokenized sentence
        return sentence