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
### check input and output                     ###
##################################################
def check_output(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str):
    '''
    checking output for skip
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    tokenize_method: string.
    s: source language
    t: target language

    returns
    -------
    skip: bool. skip boolean to skip the tokenized process
    output_dir: list. output dict for save the tokenized sentences
    '''
    output_corpus_dir = f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}" # define output dir
    utils.make_dir(f"{output_corpus_dir}") # create output dir
    output_dir_fwd = { # output from {s}->{t}
        s: f"{output_corpus_dir}/{s}-{t}.{s}.csv",
        t: f"{output_corpus_dir}/{s}-{t}.{t}.csv"
    }

    output_dir_bwd = { # output from {t}->{s}
        s: f"{output_corpus_dir}/{t}-{s}.{s}.csv",
        t: f"{output_corpus_dir}/{t}-{s}.{t}.csv"
    }

    # check output to skip
    if os.path.isfile(output_dir_fwd[s]) and os.path.isfile(output_dir_fwd[t]):
        return True, output_dir_fwd
    elif os.path.isfile(output_dir_bwd[s]) and os.path.isfile(output_dir_bwd[t]):
        return True, output_dir_bwd
    else:
        return False, output_dir_fwd

def check_input(corpus:str, sub_corpus:str, s:str, t:str):
    '''
    checking input
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    s: source language
    t: target language

    returns
    -------
    input_dir: list. output dict for save the tokenized sentences
    '''
    # check the reformatted dataset to be tokenized
    input_corpus_dir = f"data/reformatted/{corpus}/{sub_corpus}" # define input dir
    
    input_dir_fwd = {
        s: f"{input_corpus_dir}/{s}-{t}.{s}.csv",
        t: f"{input_corpus_dir}/{s}-{t}.{t}.csv"
    }
    
    input_dir_bwd = {
        s: f"{input_corpus_dir}/{t}-{s}.{s}.csv",
        t: f"{input_corpus_dir}/{t}-{s}.{t}.csv"
    }

    if os.path.isfile(input_dir_fwd[s]) and os.path.isfile(input_dir_fwd[t]):
        return input_dir_fwd
    elif os.path.isfile(input_dir_bwd[s]) and os.path.isfile(input_dir_bwd[t]):
        return input_dir_bwd
    else: # error if there is no tokenized file
        raise FileExistsError(f"There is no tokenized {corpus}-{sub_corpus}")
    
##################################################
### CLSR_tokenize                              ###
##################################################
def CLSR(setting_code:int, corpus:str, sub_corpus:str, s:str, t:str, chunksize:int=10000):
    '''
    tokenize sentences in the corpus
    input : 
        setting_code(int) - setting_code to get the experiment parameter
        corpus(str), sub_corpus(str) - corpus and sub corpus to be tokenzied
        s(str), t(str) - source and target language to be tokenized, respectively
    output: tokenized dataset files(csv) - saved in data/tokenized/ directory
    '''
    _, tokenize_method, _, _, _ = utils.get_experiment_setting(setting_code)
    logger = log.get_logger(__name__)

    # ckeck output existance
    skip, output_dir = check_output(corpus, sub_corpus, s, t, tokenize_method)

    if not skip:

        # check the reformatted dataset to be tokenized
        input_dir = check_input(corpus, sub_corpus, s, t)

        # tokenize {s} and {t} corpus
        for lang in [s, t]:
            in_dir = input_dir[lang] # define input dir for the specific language
            out_dir = output_dir[lang] # define output dir for the specific language

            if not os.path.isfile(out_dir):
                tokenized_list = [] # define to store tokenized chunk

                for idx_chunk, chunk in enumerate(pd.read_csv(in_dir, sep='\t', chunksize=chunksize)): # read csv chunk-by-chunk
                    tokenized_list.append(CLSR_tokenizer(chunk, tokenize_method)) # tokenize each chunk
                    print(f"finish {s}-{t}.{lang} part {idx_chunk+1} tokenization") # print to tell the status of each chunk
                    
                df = pd.concat(tokenized_list, ignore_index=True) # concatenate all chunk
                df.to_csv(out_dir, sep='\t', index=False) # save the tokenized
            
                logger.info(f"finish {corpus}-{sub_corpus}.{s}-{t}.{lang} tokenzation")
                logger.info(f"sample:\n{df}") # show some sample
            
            else:
                logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t}.{lang} tokenzation") # the tokenize step for specific language is skipped
    else:
        logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t} tokenization") # the tokenize step is skipped

##################################################
### tokenize method                            ###
##################################################

### CLSR_tokenizer ###
def CLSR_tokenizer(df, tokenize_method:str):
    '''
    CLSR tokenize for input dataframe

    parameters
    ----------
    df: dataframe. dataframe to be tokenized with [id, {language}] as columns
    tokenize_method: string

    returns
    -------
    tokenized_dataframe: dataframe. tokenized dataframe with dropped Nan
    '''
    sid, lang = df.columns # get each column to indicate the input language

    # select tokenizer
    if tokenize_method == 'RFR':
        tokenizer = RFR_tokenize_sentence
        args = [lang]
    else:
        raise ValueError(f"invalid tokenizer")

    df[lang] = df[lang].parallel_apply(tokenizer, args=[*args]) # tokenize each sentence
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

    if lang == 'th': # use special tokenizer for Thai
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