##################################################
### import                                     ###
##################################################
# basic lib
import json
import numpy as np
import os
import pandas as pd
# logging lib
import logging
import src.log as log
# time lib
from time import time
# custom lib
import src.utils as utils
import src.reformat as reformat_CLSR
import src.sample as sample_CLSR

##################################################
### create dataset                             ###
##################################################
def create(filename:str):
    '''
    create dataset from setting in filename
    input : filename(str) - dataset filename in create.[dataset].py format
    output: reformatted dataset files(csv) - saved in data/reformatted/ directory
    '''
    logger = log.get_logger(__name__)
    pipeline = get_create_pipeline(filename)

    for method, args in pipeline:
        logger.info(f"{method}: {args}")
        if method == 'reformat_CLSR':
            reformat_CLSR.reformat_CLSR(args)
        elif method == 'sample_CLSR':
            sample_CLSR.sample_CLSR(args)
        else:
            raise ValueError('invalid create method')

##################################################
### get setting                                ###
##################################################
def get_create_pipeline(filename:str):
    '''
    get create dataset pipeline
    input : filename(str) - dataset filename in create.[dataset].py format
    output: create dataset pipeline(tuple) - pipeline tuple for create dataset with ((method), (args)) format
    '''
    setting_name = get_filename_setting(filename) # get setting_name from filename
    with open('config/create.json') as f: # load setting json
        setting_dict = json.load(f)
    if not setting_name in setting_dict['pipeline'].keys(): # check setting existance
        raise ValueError('there is no setting for this setting_name')
    method, corpus, source_language_list, target_language_list = setting_parser(setting_dict['pipeline'][setting_name]) # get create setting from setting name
    corpus_list = setting_dict['corpus'][corpus] # get corpus list
    if method == 'CLSRs0':
        pipeline = get_CLSRs0_pipeline
    else:
        raise ValueError('method is not supported')
    return pipeline(corpus_list, source_language_list, target_language_list)

def get_filename_setting(filename:str):
    '''
    get setting name from filename with checking filename format
    input: filename(str) - dataset filename in create.[dataset].py format
    output: setting_name(str) - setting_name to get a setting dict
    '''
    create, setting_name, py = filename.split('.')
    if create != 'create' or py != 'py':
        raise ValueError('wrong filename format. the correct format should be create.[setting_alias].py')
    return setting_name

def setting_parser(setting:dict):
    '''
    parsing and cheking the setting_dict to get create dataset parameters
    input : setting_dict(dict) - setting_dict from setting_name
    output: create dataset parameters(tuple) - parameters to get a create dataset pipeline composed of
        method(str) - method to get create pipeline
        corpus(str) - corpus_name to get corpus_list
        source_list(list) - list of source language to be create
        target_list(list) - list of target language to be create
    '''
    keys = setting.keys()
    if 'method' in keys:
        method = setting['method']
    else:
        raise ValueError('missing create method')

    if 'corpus' in keys:
        corpus = setting['corpus']
    else:
        raise ValueError('missing corpus_list')
    
    if 'source_list' in keys:
        source_list = setting['source_list']
    else:
        raise ValueError('missing source_list')

    if 'target_list' in keys:
        target_list = setting['target_list']
    else:
        raise ValueError('missing target_list')

    return (method, corpus, source_list, target_list)

##################################################
### pipeline                                   ###
##################################################
def get_CLSRs0_pipeline(corpus_list:list, source_language_list:list, target_language_list:list):
    '''
    get create CLSRs0 pipeline
    input : 
        corpus_list(list) - list of corpus to create in [[corpus, sub_corpus]] format
        source_language_list(list) - list of source language
        target_language_list(list) - list of target language
    output: CLSRs0 pipeline(list) - pipeline for create the desire dataset in [ ( method, (arg) ) ] format
    '''
    pipeline = []
    for corpus, sub_corpus in corpus_list: # for in corpus_list
        for s in source_language_list: # for in source_language_list
            for t in target_language_list: # for in target_language_list
                pipeline.append(  ( 'reformat_CLSR', (corpus, sub_corpus, s, t) )  ) # reformat the original corpus

                out_corpus = f"{corpus}-{sub_corpus}-CLSRs0"
                te = f"teParent" # main test set to be sampled
                teRemain = f"{te}Remain" # the remain for create the tuning set
                pipeline.append(  ( 'sample_CLSR', (out_corpus, te, corpus, sub_corpus, s, t, (2000, 2000, 1000), True) )  )

                tu = f"tuParent" # main tuning set to be sampled
                pipeline.append(  ( 'sample_CLSR', (out_corpus, tu, out_corpus, teRemain, s, t, (200, 200, 100), True) )  )

                #create variation
                #te
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "te-c3",   out_corpus, te, s, t, (1000, 1000, 30),   False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "te-c10",  out_corpus, te, s, t, (1000, 1000, 100),  False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "te-c30",  out_corpus, te, s, t, (1000, 1000, 300),  False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "te-c50",  out_corpus, te, s, t, (1000, 1000, 500),  False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "te-c75",  out_corpus, te, s, t, (1000, 1000, 750),  False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "te-c100", out_corpus, te, s, t, (1000, 1000, 1000), False) )  )
                #tr
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "tu-c3",   out_corpus, tu, s, t, (100, 100, 3),   False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "tu-c10",  out_corpus, tu, s, t, (100, 100, 10),  False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "tu-c30",  out_corpus, tu, s, t, (100, 100, 30),  False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "tu-c50",  out_corpus, tu, s, t, (100, 100, 50),  False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "tu-c75",  out_corpus, tu, s, t, (100, 100, 75),  False) )  )
                pipeline.append(  ( 'sample_CLSR', (out_corpus, "tu-c100", out_corpus, tu, s, t, (100, 100, 100), False) )  )

    return pipeline