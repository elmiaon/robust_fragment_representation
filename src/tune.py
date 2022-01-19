##################################################
### import                                     ###
##################################################
# basic lib
from ast import literal_eval
import itertools
import json
import numpy as np
import os
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(use_memory_fs=False)
import sys
# logging lib
import logging
import src.log as log
# time lib
from time import time
# multiprocess lib
import multiprocessing as mp
PROCESS_NUM = mp.cpu_count()-2
# custom lib
import src.utils as utils
from src.aggregator.RFRa import tune as tune_RFRa
from src.aggregator.RFRa import random_tune as tune_RFRa_rand
import src.cal_score as cal_score

##################################################
### check input and output                     ###
##################################################

def check_output(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str, represent_method:list, retrieve_method:list, aggregate_method:list):
    '''
    checking output for skip
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    s: source language
    t: target language
    tokenize_method: string.
    represent_method: list. [represent method name, setting code]
    retrieve_method: list. [retrieve method name, k]
    aggregate_method: list. [aggregate method name, setting code]

    returns
    -------
    skip: bool. skip boolean to skip the tokenized process
    output_dir: list. output dict for save the tokenized sentences
    '''
    output_corpus_dir = f"data/tuned/{corpus}/{sub_corpus}/{''.join(aggregate_method)}" # define output dir
    utils.make_dir(f"{output_corpus_dir}") # create output dir
    output_dir = f"{output_corpus_dir}/{tokenize_method}.{''.join(represent_method)}.{''.join(retrieve_method)}.{s}-{t}.csv"

    # check output to skip
    if os.path.isfile(output_dir):
        return True, output_dir
    else:
        return False, output_dir

def check_input(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str, represent_method:list, retrieve_method:list):
    '''
    checking input
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    s: source language
    t: target language
    tokenize_method: string.
    represent_method: list. [represent method name, setting code]

    returns
    -------
    input_dir: list. output dict for save the tokenized sentences
    '''
    # check the tokenized dataset to be represented
    input_corpus_dir = f"data/retrieved/{corpus}/{sub_corpus}/{''.join(retrieve_method)}" # define input dir

    input_dir = f"{input_corpus_dir}/{tokenize_method}.{''.join(represent_method)}.{s}-{t}.csv"
    
    if not os.path.isfile(input_dir):
        raise ValueError(f"There is no retrieve {corpus}-{sub_corpus}.{s}-{t}")

    if os.path.isfile(f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.gold.csv"):
        gold_dir = f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.gold.csv"
    elif os.path.isfile(f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.gold.csv"):
        gold_dir = f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.gold.csv"
    else:
        raise ValueError(f"There is no reformatted gold file")
    
    return input_dir, gold_dir

##################################################
### interface                                  ###
##################################################
def tune(tokenize_method:str, represent_method:list, retrieve_method:list, aggregate_method:str, corpus:str, sub_corpus:str, s:str, t:str):
    '''
    tuning the parameter for aggregator

    parameters
    ----------
    tokenize_method: str. method to tokenize the reformatted corpus
    represent_method: list. list of [method, setting_code] to represent the tokenized corpus
    retrieve_method: list. list of [similarity_function, top_k] to retrieve k-NN similar target fragments
    aggregate_method: list. list of [method, setting_code] to aggregate the retrieved fragments
    corpus: str. corpus name
    sub_corpus: str. sub corpus name
    s: str. source language
    t: str. target language

    returns
    -------
    None

    * Note: there is not return but the function save result in data/retrieved/ directory
    '''

    logger = log.get_logger(__name__)

    skip, output_dir = check_output(corpus, sub_corpus, s, t, tokenize_method, represent_method, retrieve_method, aggregate_method) # check output existance

    if not skip:

        input_dir, gold_dir = check_input(corpus, sub_corpus, s, t, tokenize_method, represent_method, retrieve_method) # check and get exist input dir

        aggregate_method_key, aggregate_setting_code = aggregate_method # unpack aggregate_method

        if aggregate_method_key == 'RFRa':
            scores_df = tune_RFRa(aggregate_setting_code, input_dir, gold_dir, output_dir)
        elif aggregate_method_key == 'RFRa_rand':
            scores_df = tune_RFRa_rand(aggregate_setting_code, input_dir, gold_dir, output_dir)
        else:
            raise ValueError(f"invalid aggregator tuning method")
    
    else:
        scores_df = pd.read_csv(output_dir, sep='\t')
    
    at_1 = scores_df.loc[scores_df['n']==1]
    params_set = at_1.loc[at_1['align_f1']==at_1['align_f1'].max()]
    params = at_1.loc[at_1['align_f1']==at_1['align_f1'].max()].tail(1).to_numpy().reshape(-1)
    return (  params[:4]  ) # return k, beta, fil, p_thres