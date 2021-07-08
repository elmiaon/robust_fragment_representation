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

def vary_around_params(setting_code, corpus, sub_corpus, s, t, params):

    logger = log.get_logger(__name__)

    _, tokenize_method, represent_method, retrieve_method, aggregate_method = utils.get_experiment_setting(setting_code) # get settings


    input_dir = f"data/tuned/{corpus}/{sub_corpus}/{'s'.join(aggregate_method)}/{tokenize_method}.{'s'.join(represent_method)}.{'k'.join(retrieve_method)}.{s}-{t}.csv"

    result = pd.read_csv(input_dir, sep='\t')
    params_values = list([*params, 1])
    params_names = ['k', 'beta', 'fil', 'p_thres', 'n']

    for i, param in enumerate(params_names):
        temp_vals = params_values.copy()
        temp_names = params_names.copy()
        val = temp_vals.pop(i)
        name = temp_names.pop(i)
        logger.info(f"\neffect of {name}:\n{result.loc[  (result[temp_names[0]]==temp_vals[0]) & (result[temp_names[1]]==temp_vals[1]) & (result[temp_names[2]]==temp_vals[2]) & (result[temp_names[3]]==temp_vals[3])  ]}")
