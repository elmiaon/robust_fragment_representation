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
import src.tokenize as tokenize
import src.represent as represent
import src.retrieve as retrieve
import src.tune as tune
import src.analysis as analysis
import src.get_test_score as get_test_score
# import src.aggregator as aggregator

##################################################
### run experiment                             ###
##################################################
def run(filename:str):
    '''
    run experiment(s) from setting in filename
    input : filename(str) - dataset filename in exp.[dataset].[source_language]-[target_language].py format
    output: result for each steps(csv) - the result from each step saved in data/[step_name]/ directory
    '''
    logger = log.get_logger(__name__) # get logger instance
    pipeline = get_experiment_pipeline(filename) # get experiment pipeline from filename

    for method, args in pipeline: # run each experiment in the pipeline
        logger.info(f"run experiment: {method}, {args}") # log the running status
        if method == 'RFR-CLSR': # define running method and check the whether it is supported
            process = RFR_CLSR
        else:
            raise ValueError(f"invalid method")
        process(args) # run the experiment

    results = {}
    result_root_dir = f"data/results/{filename[4:-3]}"
    utils.make_dir(result_root_dir)
    for method, args in pipeline:
        logger.info(f"collecting result from {method}, {args}")
        collected_result = results.get(method, [])
        if method == 'RFR-CLSR':
            result_collect = RFR_result_collector
        else:
            raise ValueError(f"invalid method")
        collected_result.append(result_collect(args))
        results[method] = collected_result
    for key in results:
        result_df = pd.concat(results[key], ignore_index=True)
        print(result_df)
        result_df.to_csv(f"{result_root_dir}/{key}.csv", sep='\t', index=False)

##################################################
### get setting                                ###
##################################################
def get_experiment_pipeline(filename:str):
    '''
    get experiment pipeline
    input : filename(str) - dataset filename in exp.[dataset].[source_language]-[target_language].py format
    output: experiment pipeline(tuple) - pipeline tuple for create dataset with ((method), (args)) format
    '''
    setting_code, s, t = get_filename_setting(filename) # get setting_code from filename

    with open('config/run.json') as f: # load setting json
        setting_dict = json.load(f)
    if not setting_code in setting_dict['pipeline'].keys(): # check setting existance
        raise ValueError('there is no pipeline for this pipeline_name')
    setting_code = str(setting_code) # convert setting_code to string
    method, setting_list, corpus = setting_parser(setting_dict['pipeline'][setting_code]) # get setting params from setting_code to create pipeline
    corpus_list = setting_dict['corpus'][corpus] # get corpus list from corpus name
    if method == 'RFR-CLSR': # define get pipeline method
        pipeline = get_RFR_CLSR_pipeline
    else:
        raise ValueError(f"invalid pipeline for {method}")
    return pipeline(setting_list, corpus_list, s, t) # return pipeline with respect to the setting

def get_filename_setting(filename:str):
    '''
    get setting name from filename with checking filename format
    input: filename(str) - dataset filename in create.[dataset].py format
    output: 
        pipeline_name(str) - pipeline name for get the experiment pipeline
        s(str) - source language
        t(str) - target language
    '''
    exp, pipeline_name, language_pair, py = filename.split('.') # split filename using .(period)
    if exp != 'exp' and py != 'py': # check the start and end of the filename
        raise ValueError('wrong filename format') # throw an error if get the wrong format
    s, t = language_pair.split('-') # get source and target language
    if s=='' or t=='':
        raise ValueError('missing language(s)') # throw an error if get the wrong format
    return pipeline_name, s, t


def setting_parser(setting:dict):
    '''
    parsing and cheking the setting_dict to get experiment setting
    input : setting_dict(dict) - setting_dict from setting_name
    output: create dataset parameters(tuple) - parameters to get a create dataset pipeline composed with
        method(str) - method to get create pipeline
        setting_list(list) - list of setting_code to be run
        corpus(str) - corpus name to get corpus_list
    '''
    keys = setting.keys()
    if 'method' in keys: # get method
        method = setting['method']
    else:
        raise ValueError('missing experiment method')

    if 'setting_list' in keys: # get setting_list
        setting_list = setting['setting_list']
    else:
        raise ValueError('missing experiment setting_list')

    if 'corpus' in keys: # get corpus
        corpus = setting['corpus']
    else:
        raise ValueError('missing corpus')

    return (method, setting_list, corpus)

##################################################
### pipeline                                   ###
##################################################
def get_RFR_CLSR_pipeline(setting_list:list, corpus_list:list, s:str, t:str):
    '''
    get RFR-CLSR pipeline
    input : 
        setting_list(list) - list of setting_code to be run
        corpus_list(list) - list of corpus to be run in [[corpus, sub_corpus]] format
        s(str) - source language
        t(str) - target language
    output: RFR-CLSR pipeline(list) - pipeline for run the experiments in [ ( method, (arg) ) ] format
    '''
    pipeline = []
    for setting_code in setting_list: # for in setting list
        for tr_corpus, tr_sub_corpus, te_corpus, te_sub_corpus in corpus_list: # for each train and test corpus in corpus_list
            pipeline.append(  ( 'RFR-CLSR', ( setting_code, tr_corpus, tr_sub_corpus, te_corpus, te_sub_corpus, s, t ) )  )
    return pipeline

##################################################
### RFR_CLSR                                   ###
##################################################
def RFR_CLSR(args):
    '''
    tokenize sentences in the corpus
    input : args composed of
        SETTING_CODE(int) - setting_code to get the experiment parameter
        TUNE_CORPUS(str), TUNE_SUB_CORPUS(str) - corpus and sub corpus as tuning set
        TEST_CORPUS(str), TEST_SUB_CORPUS(str) - corpus and sub corpus as testing set
        S(str), T(str) - source and target language, respectively
    output: result for each steps(csv) - the result from each step saved in data/[step_name]/ directory
    '''
    logger = log.get_logger(__name__)
    SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, TEST_CORPUS, TEST_SUB_CORPUS, S, T = args # unpack args
    DESCRIPTION, TOKENIZE_METHOD, REPRESENT_METHOD, RETRIEVE_METHOD, AGGREGATE_METHOD = utils.get_experiment_setting(SETTING_CODE) # get experiment setting from setting_code
    n_steps = 8 # total number of steps to track the progress

    logger = log.get_logger(__name__) # get logger instance

    world_tic = time() # start time of the experiment
    step = 1 # the current step

    # descirption of the experiment
    setting = f'''
    {'='*50}
    DESCRIPTION: {DESCRIPTION}
    SETTING_CODE: {SETTING_CODE}
    TRAIN CORPUS: {TUNE_CORPUS} - {TUNE_SUB_CORPUS}
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

    #####
    # 1.) tokenize the training dataset
    #####
    tic = time()
    tokenize.CLSR(SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T)
    toc = time()
    logger.info(f"step {step}/{n_steps} - preprocess training data in {toc-tic:.2f} second(s)")
    step+=1

    #####
    # 2.) Represent the training dataset
    #####
    tic = time()
    represent.CLSR(SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T)
    toc = time()
    logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    step+=1

    #####
    # 3.) Retrieve the training dataset
    #####
    tic = time()
    retrieve.CLSR(SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T)
    toc = time()
    logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    step+=1


    #####
    # 4.) Parameters tuning
    #####
    tic = time()
    params = tune.tune_aggregator(SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T)
    toc = time()
    logger.info(f"step {step}/{n_steps} - tuning parameter in {toc-tic:.2f} second(s)")
    step+=1
    
    #####
    # 4.5.) Looking around
    #####
    tic = time()
    analysis.vary_around_params(SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T, params)
    toc = time()
    logger.info(f"step {step}/{n_steps} - tuning parameter in {toc-tic:.2f} second(s)")
    # step+=1

    #####
    # 5.) tokenize the testing dataset
    #####
    tic = time()
    tokenize.CLSR(SETTING_CODE, TEST_CORPUS, TEST_SUB_CORPUS, S, T)
    toc = time()
    logger.info(f"step {step}/{n_steps} - preprocess training data in {toc-tic:.2f} second(s)")
    step+=1

    #####
    # 6.) Represent the testing dataset
    #####
    tic = time()
    represent.CLSR(SETTING_CODE, TEST_CORPUS, TEST_SUB_CORPUS, S, T)
    toc = time()
    logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    step+=1

    #####
    # 7.) Retrieve the training dataset
    #####
    tic = time()
    retrieve.CLSR(SETTING_CODE, TEST_CORPUS, TEST_SUB_CORPUS, S, T)
    toc = time()
    logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    step+=1

    #####
    # 8.) Get test score
    #####
    tic = time()
    get_test_score.get_score(SETTING_CODE, TEST_CORPUS, TEST_SUB_CORPUS, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T, params)
    toc = time()
    logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    step+=1
    
    # #####
    # # 8.) Error analysis
    # #####
    # tic = time()
    # tune_retrieve_params.false_analysis(EXP_CODE, TEST_CORPUS, TEST_SUB_CORPUS, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T, params)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1

##################################################
### result_collector                           ###
##################################################
def RFR_result_collector(args):
    SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, TEST_CORPUS, TEST_SUB_CORPUS, S, T = args # unpack args
    DESCRIPTION, TOKENIZE_METHOD, REPRESENT_METHOD, RETRIEVE_METHOD, AGGREGATE_METHOD = utils.get_experiment_setting(SETTING_CODE) # get experiment setting from setting_code

    result_df = pd.read_csv(f"data/tested/{TOKENIZE_METHOD}.{'s'.join(REPRESENT_METHOD)}.{'k'.join(RETRIEVE_METHOD)}.{'s'.join(AGGREGATE_METHOD)}/test_{TEST_CORPUS}_{TEST_SUB_CORPUS}.tune_{TUNE_CORPUS}_{TUNE_SUB_CORPUS}.{S}-{T}.csv", sep='\t')
    len_result = len(result_df)
    corpus_df = pd.DataFrame({
        'te': [TEST_CORPUS]*len_result,
        'te_sub': [TEST_SUB_CORPUS]*len_result,
        'tu': [TUNE_CORPUS]*len_result,
        'tu_sub': [TUNE_SUB_CORPUS]*len_result,
        'lang': [f"{S}-{T}"]*len_result,
        'setting': [SETTING_CODE]*len_result
    })
    return pd.concat([corpus_df, result_df], axis=1)