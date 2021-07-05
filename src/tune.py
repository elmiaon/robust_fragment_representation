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
from scipy import ndimage
from scipy.stats import entropy
import sys
from googletrans import Translator
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
    output_corpus_dir = f"data/tuned/{corpus}/{sub_corpus}/{'s'.join(aggregate_method)}" # define output dir
    utils.make_dir(f"{output_corpus_dir}") # create output dir
    output_dir = f"{output_corpus_dir}/{tokenize_method}.{'s'.join(represent_method)}.{'k'.join(retrieve_method)}.{s}-{t}.csv"

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
### tune aggregator                            ###
##################################################
def tune_aggregator(setting_code:int, corpus:str, sub_corpus:str, s:str, t:str):
    '''
    tuning the parameter for aggregator

    parameters
    ----------
    setting_code: int. setting coder to get the experiment parameters
    corpus: string. corpus to be tuned
    sub_corpus: string. sub corpus to be tuned
    s: string. source langugae to be tuned
    t: string. target language to be tuned

    returns
    -------
    tuned dataset files: csv. saved in data/tuned/ directory
    '''

    logger = log.get_logger(__name__)

    _, tokenize_method, represent_method, retrieve_method, aggregate_method = utils.get_experiment_setting(setting_code) # get settings

    skip, output_dir = check_output(corpus, sub_corpus, s, t, tokenize_method, represent_method, retrieve_method, aggregate_method) # check output existance

    if not skip:

        input_dir, gold_dir = check_input(corpus, sub_corpus, s, t, tokenize_method, represent_method, retrieve_method) # check and get exist input dir

        method, aggregate_setting_code = aggregate_method # unpack aggregate_method

        args = get_aggregate_setting(method, aggregate_setting_code)

        gold_df = pd.read_csv(gold_dir, sep='\t') # load gold
        gold_df = gold_df[[s, t]] # reorder columns
        gold_df.columns = ['id', 'pair'] # rename gold_df columns
        # print(f"gold_df:\n{gold_df}")

        retrieved_df = pd.read_csv(input_dir, sep='\t', converters={'candidates': literal_eval, 'distance': utils.byte_decode})

        if method == 'RFR':
            tune_aggregate = tune_RFR_aggregator
        else:
            raise ValueError(f"invalid aggregator tuning method")
        scores_df = tune_aggregate(retrieved_df, gold_df, *args)
        scores_df.to_csv(output_dir, sep='\t', index=False)
    
    else:
        scores_df = pd.read_csv(output_dir, sep='\t')
    
    at_1 = scores_df.loc[scores_df['n']==1]
    params_set = at_1.loc[at_1['align_f1']==at_1['align_f1'].max()]
    print(params_set)
    params = at_1.loc[at_1['align_f1']==at_1['align_f1'].max()].tail(1)
    print(params)


##################################################
### tuning methods                             ###
##################################################
def tune_RFR_aggregator(retrieved_df, gold_df, k_list, beta_list, filter_thres, p_thres, at):
    '''
    parameters
    ----------
    k_list: np array. list of k in k-NN to be included in aggration
    beta_list: np array. list of spiking coefficient
    filter_thre: np array. list of minimum entropy portion to be keep
    p_thres: np array. list of probability thresold
    at: np array. list of p@n_top

    returns
    -------
    tuned_score: csv. saved in data/tuned/ directory
    best_params: set of best tuning score params
    '''

    kerneled_df = aggregator.get_kernel_retrieved_df(retrieved_df)

    params_set = [[kerneled_df], [gold_df], k_list, beta_list, filter_thres, p_thres, [at], [False]] # generate parameter set to save

    mp_input = itertools.product(*params_set) # do permutation
    with mp.Pool(processes=PROCESS_NUM) as p:
        scores = p.map(aggregator.get_RFR_result, mp_input)
    
    scores_df = pd.concat(scores)
    return scores_df

##################################################
### get aggregate setting                      ###
##################################################
def get_aggregate_setting(method:str, setting_code:int):
    ''' 
    get aggregate setting from setting_code

    parameters
    ----------
    method: string. method to identify aggregate parset
    setting_code: int. setting code to get the setting_dict

    returns
    -------
    base_encoder_name: string. name to get base encoder model
    args: list. list of parameters for represent method
    '''

    with open('config/run.json') as f: # load RFR-CLSR json setting
        setting_dict = json.load(f)
    setting_code = str(setting_code)

    if not setting_code in setting_dict['aggregate_setting'][method].keys(): # check setting code existance
        raise ValueError(f"invalid {method} aggregate setting_code")
    
    setting = setting_dict['aggregate_setting'][method][setting_code]
    if method == 'RFR':
        aggregate_parser = RFR_parser
    else:
        return ValueError("invalid aggregate method")

    return aggregate_parser(setting)

def RFR_parser(setting):
    
    keys = setting.keys()
    if 'description' in keys: # get description
        DESCRIPTION = setting['description']
    else:
        raise ValueError('missing experiment description')
    
    if 'kNN' in keys: # get k for kNN
        start, stop, step = setting['kNN']
    else:
        start, stop, step = 5, 51, 5
    kNN = np.arange(start, stop, step)
    
    if 'beta' in keys: # get spiking coefficient
        start, stop, step = setting['beta']
    else:
        start, stop, step = 50, 101, 5
    beta = np.arange(start, stop, step)

    if 'p_min_entropy' in keys: # get pecentage of min entropy
        start, stop, step = setting['p_min_entropy']
    else:
        start, stop, step = 0.2, 1, 9
    p_min_entropy = np.linspace(start, stop, step)
    
    if 'p_thres' in keys: # get 
        start, stop, step = setting['p_thres']
    else:
        start, stop, step = 0.3, 1, 8
    p_thres = np.linspace(start, stop, step)

    if 'at' in keys:
        at = setting['at']
    else:
        at = np.array([1, 5,10])
    
    return (kNN, beta, p_min_entropy, p_thres, at)