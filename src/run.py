##################################################
### import                                     ###
##################################################
# basic lib
from ast import literal_eval
import json
import numpy as np
import os
import pandas as pd
# logging lib
import logging
import src.log as log
# timing lib
from time import time
# custom lib
import src.utils as utils
import src.run as run

##################################################
### run experiment                             ###
##################################################
def run(filename):
    logger = log.get_logger(__name__)
    pipeline = get_experiment_pipeline(filename)

    for method, args in pipeline:
        logger.info(f"run experiment: {method}, {args}")
        if method == 'RFR-CLSR':
            process = RFR_CLSR
        else:
            raise ValueError(f"invalid method")
        process(args)

##################################################
### get setting                                ###
##################################################
def get_experiment_pipeline(filename):
    '''
    get experiment pipeline from filename
    input: filename
    output: experiment pipeline ((task), (args))
    '''
    setting_code, s, t = get_filename_setting(filename) # get setting_code from filename

    with open('config/run.json') as f: # load setting json
        setting_dict = json.load(f)
    if not setting_code in setting_dict['pipeline'].keys(): # check settint existance
        raise ValueError('there is no pipeline for this pipeline_name')
    setting_code = str(setting_code)
    method, setting_list, corpus = setting_parser(setting_dict['pipeline'][setting_code]) # get experiment setting from setting name
    corpus_list = setting_dict['corpus'][corpus]
    if method == 'RFR-CLSR':
        pipeline = get_RFR_CLSR_pipeline
    else:
        raise ValueError(f"invalid pipeline for {method}")
    return pipeline(setting_list, corpus_list, s, t)

def get_filename_setting(filename:str):
    '''
    get setting name from filename with checking filename format
    input: filename(str) with format exp.[pipeline_name].[language_pair].py"
    output: (pipeline_name, s, t)
    '''
    exp, pipeline_name, language_pair, py = filename.split('.')
    if exp != 'exp' and py != 'py':
        raise ValueError('wrong filename format')
    s, t = language_pair.split('-')
    if s=='' or t=='':
        raise ValueError('missing language(s)')
    return pipeline_name, s, t


def setting_parser(setting:dict):
    '''
    parsing and cheking the setting_dict 
    input : setting_dict
    output: experiment settings
    '''
    keys = setting.keys()
    if 'method' in keys:
        method = setting['method']
    else:
        raise ValueError('missing experiment method')

    if 'setting_list' in keys:
        setting_list = setting['setting_list']
    else:
        raise ValueError('missing experiment setting_list')

    if 'corpus' in keys:
        corpus = setting['corpus']
    else:
        raise ValueError('missing corpus')

    return (method, setting_list, corpus)

##################################################
### pipeline                                   ###
##################################################
def get_RFR_CLSR_pipeline(setting_list, corpus_list, s, t):
    pipeline = []
    for setting_code in setting_list:
        for tr_corpus, tr_sub_corpus, te_corpus, te_sub_corpus in corpus_list:
            pipeline.append(  ( 'RFR-CLSR', ( setting_code, tr_corpus, tr_sub_corpus, te_corpus, te_sub_corpus, s, t ) )  )
    return pipeline

##################################################
### RFR_CLSR                                   ###
##################################################
def RFR_CLSR(args):
    logger = log.get_logger(__name__)
    SETTING_CODE, TRAIN_CORPUS, TRAIN_SUB_CORPUS, TEST_CORPUS, TEST_SUB_CORPUS, S, T = args
    DESCRIPTION, TOKENIZE_METHOD, REPRESENT_METHOD, RETRIEVE_METHOD, AGGREGATE_METHOD = utils.get_RFR_CLSR_setting(SETTING_CODE)
    n_steps = 8

    # 0.5) create log session
    logger = log.get_logger(__name__)

    world_tic = time()
    step = 1

    setting = f'''
    {'='*50}
    DESCRIPTION: {DESCRIPTION}
    SETTING_CODE: {SETTING_CODE}
    TRAIN CORPUS: {TRAIN_CORPUS} - {TRAIN_SUB_CORPUS}
    TEST_CORPUS: {TEST_CORPUS} - {TEST_SUB_CORPUS}
    S: {S}
    T: {T}
    TOKENIZE_METHOD: {TOKENIZE_METHOD}
    REPRESENT_METHOD: {REPRESENT_METHOD}
    RETRIEVE_METHOD: {RETRIEVE_METHOD}
    AGGREGATE_METHOD: {AGGREGATE_METHOD}
    {'='*50}
    '''

    logger.info(f"{setting}")

    # #####
    # # 1.) Preprocess training dataset
    # #####
    # tic = time()
    # preprocess.preprocess(EXP_CODE, TRAIN_CORPUS, TRAIN_SUB_CORPUS, S, T)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - preprocess training data in {toc-tic:.2f} second(s)")
    # step+=1

    # #####
    # # 2.) Retrieve the candidates
    # #####
    # tic = time()
    # retrieve.retrieve(EXP_CODE, TRAIN_CORPUS, TRAIN_SUB_CORPUS, S, T)
    # # training.retrieve_analysis()
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1


    # #####
    # # 3.) Parameters tuning
    # #####
    # tic = time()
    # params = tune_retrieve_params.tune_retrieve_params(EXP_CODE, TRAIN_CORPUS, TRAIN_SUB_CORPUS, S, T)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - tuning parameter in {toc-tic:.2f} second(s)")
    # step+=1
    
    # #####
    # # 4.) Looking around
    # #####
    # tic = time()
    # tune_retrieve_params.vary_around_best(EXP_CODE, TRAIN_CORPUS, TRAIN_SUB_CORPUS, S, T, params)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - tuning parameter in {toc-tic:.2f} second(s)")
    # step+=1

    # #####
    # # 5.) Preprocess training dataset
    # #####
    # tic = time()
    # preprocess.preprocess(EXP_CODE, TEST_CORPUS, TEST_SUB_CORPUS, S, T)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - preprocess testing data in {toc-tic:.2f} second(s)")
    # step+=1

    # #####
    # # 6.) retrieve the candidate
    # #####
    # tic = time()
    # retrieve.retrieve(EXP_CODE, TEST_CORPUS, TEST_SUB_CORPUS, S, T)
    # # training.retrieve_analysis()
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1

    # #####
    # # 7.) Get test score
    # #####
    # tic = time()
    # tune_retrieve_params.get_test_score(EXP_CODE, TEST_CORPUS, TEST_SUB_CORPUS, TRAIN_CORPUS, TRAIN_SUB_CORPUS, S, T, params)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1
    
    # #####
    # # 8.) Error analysis
    # #####
    # tic = time()
    # tune_retrieve_params.false_analysis(EXP_CODE, TEST_CORPUS, TEST_SUB_CORPUS, TRAIN_CORPUS, TRAIN_SUB_CORPUS, S, T, params)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1