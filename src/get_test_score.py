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
import src.aggregator as aggregator
import src.cal_score as cal_score

##################################################
### check input and output                     ###
##################################################

def check_output(corpus:str, sub_corpus:str, tune_corpus:str, tune_sub_corpus:str, s:str, t:str, tokenize_method:str, represent_method:list, retrieve_method:list, aggregate_method:list):
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
    score_output_corpus_dir = f"data/tested/{tokenize_method}.{'s'.join(represent_method)}.{'k'.join(retrieve_method)}.{'s'.join(aggregate_method)}" # define output dir
    utils.make_dir(f"{score_output_corpus_dir}") # create output dir
    score_output_dir = f"{score_output_corpus_dir}/test_{corpus}_{sub_corpus}.tune_{tune_corpus}_{tune_sub_corpus}.{s}-{t}.csv"

    ans_output_corpus_dir = f"data/ans/{tokenize_method}.{'s'.join(represent_method)}.{'k'.join(retrieve_method)}.{'s'.join(aggregate_method)}"
    utils.make_dir(f"{ans_output_corpus_dir}")
    ans_output_dir = f"{ans_output_corpus_dir}/test_{corpus}_{sub_corpus}.tune_{tune_corpus}_{tune_sub_corpus}.{s}-{t}.csv"

    analysis_output_corpus_dir = f"data/analysis/{tokenize_method}.{'s'.join(represent_method)}.{'k'.join(retrieve_method)}.{'s'.join(aggregate_method)}/test_{corpus}_{sub_corpus}.tune_{tune_corpus}_{tune_sub_corpus}/{s}-{t}" # define output dir
    utils.make_dir(f"{analysis_output_corpus_dir}") #create analysis output dir
    TP_output_dir = f"{analysis_output_corpus_dir}/TP.csv"
    TN_output_dir = f"{analysis_output_corpus_dir}/TN.csv"
    FP_output_dir = f"{analysis_output_corpus_dir}/FP.csv"
    FN_output_dir = f"{analysis_output_corpus_dir}/FN.csv"
    FA_output_dir = f"{analysis_output_corpus_dir}/FA.csv"
    # check output to skip
    skip = os.path.isfile(score_output_dir) and \
           os.path.isfile(ans_output_dir) and \
           os.path.isfile(TP_output_dir) and \
           os.path.isfile(TN_output_dir) and \
           os.path.isfile(FP_output_dir) and \
           os.path.isfile(FN_output_dir) and \
           os.path.isfile(FA_output_dir)

    if skip:
        return True, score_output_dir, ans_output_dir, TP_output_dir, TN_output_dir, FP_output_dir, FN_output_dir, FA_output_dir
    else:
        return False, score_output_dir, ans_output_dir, TP_output_dir, TN_output_dir, FP_output_dir, FN_output_dir, FA_output_dir

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
    input_corpus_dir = f"data/retrieved/{corpus}/{sub_corpus}/{'k'.join(retrieve_method)}" # define input dir

    input_dir = f"{input_corpus_dir}/{tokenize_method}.{'s'.join(represent_method)}.{s}-{t}.csv"
    
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
### get test score                             ###
##################################################
def get_test_score(setting_code:int, corpus:str, sub_corpus:str,tune_corpus:str, tune_sub_corpus:str, s:str, t:str, params):
    '''
    get test score for the test set

    parameters
    ----------
    setting_code: int. setting coder to get the experiment parameters
    corpus: string. corpus to be tested
    sub_corpus: string. sub corpus to be tested
    tune_corpus: string. tuned params' corpus name
    tune_sub_corpus: tuned params' sub corpus name
    s: string. source langugae to be tuned
    t: string. target language to be tuned
    params: np array. parameters to get the score composed of
        k: int. k of k-NN to be included in aggration
        beta: int. spiking coefficient
        fil: float. minimum entropy portion to be keep
        p_thres: float. probability thresold
        at: list. to calculate score@n for n in at

    returns
    -------
    test_score: csv. test score for each n in at

    '''

    logger = log.get_logger(__name__)

    _, tokenize_method, represent_method, retrieve_method, aggregate_method = utils.get_experiment_setting(setting_code) # get settings

    skip, score_output_dir, ans_output_dir, TP_outout_dir, TN_outout_dir, FP_outout_dir, FN_outout_dir, FA_outout_dir = check_output(corpus, sub_corpus, tune_corpus, tune_sub_corpus, s, t, tokenize_method, represent_method, retrieve_method, aggregate_method) # check output existance

    if not skip:

        input_dir, gold_dir = check_input(corpus, sub_corpus, s, t, tokenize_method, represent_method, retrieve_method) # check and get exist input dir

        method, aggregate_setting_code = aggregate_method # unpack aggregate_method

        k, beta, fil, p_thres, at = aggregator.get_aggregate_setting(method, aggregate_setting_code)

        k, beta, fil, p_thres = params

        gold_df = pd.read_csv(gold_dir, sep='\t') # load gold
        gold_df = gold_df[[s, t]] # reorder columns
        gold_df.columns = ['id', 'pair'] # rename gold_df columns
        # print(f"gold_df:\n{gold_df}")

        retrieved_df = pd.read_csv(input_dir, sep='\t', converters={'candidates': literal_eval, 'distance': utils.byte_decode})

        if method == 'RFR':
            kerneled_df = aggregator.get_kernel_retrieved_df(retrieved_df)
            args = (kerneled_df, gold_df, int(k), int(beta), fil, p_thres, at, True)
            get_score = cal_score.get_RFR_result
        else:
            raise ValueError(f"invalid aggregator tuning method")
    
        score_df, ans, TP, TN, FP, FN, FA = get_score(args)
        score_df.to_csv(score_output_dir, sep='\t', index=False)
        ans.to_csv(ans_output_dir, sep='\t', index=False)
        TP.to_csv(TP_outout_dir, sep='\t', index=False) 
        TN.to_csv(TN_outout_dir, sep='\t', index=False) 
        FP.to_csv(FP_outout_dir, sep='\t', index=False) 
        FN.to_csv(FN_outout_dir, sep='\t', index=False) 
        FA.to_csv(FA_outout_dir, sep='\t', index=False)
    else:
        scores_df = pd.read_csv(score_output_dir, sep='\t')