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
import src.preprocess as preprocess
# import src.retrieve as retrieve
# import src.tune_retrieve_params as tune_retrieve_params
import src.utils as utils

##################################################
### run experiment                             ###
##################################################
def get_filename_setting(filename:str):
    '''
    get part 1/2 of setting from filename
    input: filename(str) with format "{alias}.{test_corpus}.{trian_corpus}.{language_pair}.py"
    output: (alias, test_corpus, train_corpus, source_language, target_language)
    '''
    exp, pipeline_name, language_pair, py = filename.split('.')
    if exp != 'exp' and py != 'py':
        raise ValueError('wrong filename format')
    s, t = language_pair.split('-')
    if s=='' or t=='':
        raise ValueError('missing language(s)')
    return pipeline_name, s, t