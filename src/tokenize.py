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

def tokenize(tokenize_method:str, corpus:str, sub_corpus:str, s:str, t:str):
    '''
    tokenize the reformatted corpus

    parameters
    ----------
    tokenize_method: str. method to tokenize the reformatted corpus
    corpus: str. corpus name
    sub_corpus: str. sub corpus name
    s: str. source language
    t: str. target language

    returns
    -------
    None

    * Note: there is not return but the function save result in data/tokenized/ directory
    '''
    skip, output_dir = check_output(corpus, sub_corpus, s, t, tokenize_method)

    if not skip:

        input_dir = check_input(corpus, sub_corpus, s, t)

        if tokenize_method == 'RFRt':
            execute_tokenize = RFRt
        else:
            raise ValueError(f"Invalid tokenize method")
        
        execute_tokenize(input_dir, output_dir)
    
    else:
        logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t} tokenization") # the tokenize step is skipped

##################################################
### RFRt                                       ###
##################################################
def RFRt(input_dir:dict, output_dir:dict, chunksize:int=10000):
    '''
    tokenize sentences in the corpus
    
    parameters
    ----------
    input_dir: dict. input directory with language as key
    output_dir: dict. output directory with language as key
    chunksize: int. default=10000. 
    '''

    logger = log.get_logger(__name__)
    # tokenize {s} and {t} corpus
    for lang in input_dir:
        in_dir = input_dir[lang] # define input dir for the specific language
        out_dir = output_dir[lang] # define output dir for the specific language

        if not os.path.isfile(out_dir):
            tokenized_list = [] # define to store tokenized chunk

            for idx_chunk, chunk in enumerate(pd.read_csv(in_dir, sep='\t', chunksize=chunksize)): # read csv chunk-by-chunk
                chunk[lang] = chunk[lang].parallel_apply(RFR_tokenize_sentence, args=[lang,])
                tokenized_list.append(chunk) # tokenize each chunk
                print(f"finish {lang} tokenization part {idx_chunk+1}") # print to tell the status of each chunk
                
            df = pd.concat(tokenized_list, ignore_index=True) # concatenate all chunk
            df.to_csv(out_dir, sep='\t', index=False) # save the tokenized
        
            logger.info(f"finish {lang} tokenzation")
            logger.info(f"sample:\n{df}") # show some sample
        
        else:
            logger.info(f"skip {lang} tokenzation") # the tokenize step for specific language is skipped

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