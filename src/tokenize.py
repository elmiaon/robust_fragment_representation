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
from src.tokenizer.RFRt import RFRt

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
### interface                                  ###
##################################################

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
    logger = log.get_logger(__name__)
    
    skip, output_dir = check_output(corpus, sub_corpus, s, t, tokenize_method)

    if not skip:

        input_dir = check_input(corpus, sub_corpus, s, t)

        if tokenize_method == 'RFRt':
            RFRt(input_dir, output_dir)
        else:
            raise ValueError(f"Invalid tokenize method")
    
    else:
        logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t} tokenization") # the tokenize step is skipped

