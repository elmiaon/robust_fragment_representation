##################################################
### import                                     ###
##################################################
# basic lib
import os
# logging lib
import logging
import src.log as log
# custom lib
import src.utils as utils
from src.representation.RFRr import RFRr

##################################################
### check input and output                     ###
##################################################
def check_output(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str, represent_method:list):
    '''
    checking output for skip
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    tokenize_method: string.
    represent_method: list. [represent method name, setting code]
    s: source language
    t: target language

    returns
    -------
    skip: bool. skip boolean to skip the tokenized process
    output_dir: list. output dict for save the tokenized sentences
    '''
    output_corpus_dir = f"data/represented/{corpus}/{sub_corpus}/{''.join(represent_method)}" # define output dir
    utils.make_dir(f"{output_corpus_dir}") # create output dir
    output_dir_fwd = { # output from {s}->{t}
        s: f"{output_corpus_dir}/{tokenize_method}.{s}-{t}.{s}.csv",
        t: f"{output_corpus_dir}/{tokenize_method}.{s}-{t}.{t}.csv"
    }

    output_dir_bwd = { # output from {t}->{s}
        s: f"{output_corpus_dir}/{tokenize_method}.{t}-{s}.{s}.csv",
        t: f"{output_corpus_dir}/{tokenize_method}.{t}-{s}.{t}.csv"
    }

    # check output to skip
    if os.path.isfile(output_dir_fwd[s]) and os.path.isfile(output_dir_fwd[t]):
        return True, output_dir_fwd
    elif os.path.isfile(output_dir_bwd[s]) and os.path.isfile(output_dir_bwd[t]):
        return True, output_dir_bwd
    else:
        return False, output_dir_fwd

def check_input(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str):
    '''
    checking input
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    s: source language
    t: target language
    tokenize_method: string.

    returns
    -------
    input_dir: list. output dict for save the tokenized sentences
    '''
    # check the tokenized dataset to be represented
    input_corpus_dir = f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}" # define input dir

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
    else: # error if there is no reformatted file
        raise FileExistsError(f"There is no reformatted {corpus}-{sub_corpus}")

##################################################
### interface                                  ###
##################################################
def represent(tokenize_method:str, represent_method:str, corpus:str, sub_corpus:str, s:str, t:str):
    '''
    represent the tokenized corpus

    parameters
    ----------
    tokenize_method: str. method to tokenize the reformatted corpus
    represent_method: list. list of [method, setting_code] to represent the tokenized corpus
    corpus: str. corpus name
    sub_corpus: str. sub corpus name
    s: str. source language
    t: str. target language

    returns
    -------
    None

    * Note: there is not return but the function save result in data/represented/ directory
    '''
    logger = log.get_logger(__name__)

    skip, output_dir = check_output(corpus, sub_corpus, s, t, tokenize_method, represent_method)

    if not skip:
        input_dir = check_input(corpus, sub_corpus, s, t, tokenize_method)

        represent_method_key, represent_setting_code = represent_method # unpack represent_method

        if represent_method_key == 'RFRr':
            RFRr(represent_setting_code, input_dir, output_dir)
        else:
            raise ValueError(f"invalid represent method")
    
    else:
        logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t} representation") # the represent step is skipped