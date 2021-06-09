import os
import base64
import itertools
import json
import numpy as np
import os
import pandas as pd
import pickle

import src.log as log

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def byte_encode(obj):
    b64_bytes = base64.b64encode(pickle.dumps(obj))
    return b64_bytes.decode('utf8')

def byte_decode(b64_str):
    return pickle.loads(base64.b64decode(b64_str))

def get_RFR_CLSR_setting(setting_code):
    '''
    get setting from setting_name
    input: setting_name(str)
    output: (DESCRIPTION, PREPROCESS_METHOD, W2V, NG, FS, SL, S2V, S2V_ARGS_LIST, DISTANCE_CALCULATION, N_MIN_DISTANCE, BETA, P_MIN_ENTROPY, P_TOP, N_RETRIEVE)
    '''
    logger = log.get_logger(__name__) # get logger instance
    with open('config/RFR-CLSR.json') as f: # load RFR-CLSR json setting
        setting_dict = json.load(f)
    setting_code = str(setting_code)
    if not setting_code in setting_dict['setting'].keys(): # check setting code existance
        raise ValueError('invalid experiment setting_name')

    setting = setting_dict['setting'][setting_code]
    keys = setting.keys()
    if 'description' in keys:
        DESCRIPTION = setting['description']
    else:
        raise ValueError('missing experiment description')
    
    if 'tokenized_method' in keys:
        TOKENIZED_METHOD = setting['tokenized_method']
    else:
        TOKENIZED_METHOD = 'RFR'
    
    if 'represent_method' in keys:
        REPRESENT_METHOD = setting['representation_method']
    else:
        REPRESENT_METHOD = ['RFR',0] # representation method, setting code

    if 'retrieve_method' in keys:
        RETRIEVE_METHOD = setting['retrieve_method']
    else:
        RETRIEVE_METHOD = ['cosine', 50] # retrieve method, k-NN
    
    if 'aggregate_method' in keys:
        AGGREGATE_METHOD = setting['aggregate_method']
    else:
        AGGREGATE_METHOD = ['RFR', 0]

    return (DESCRIPTION, TOKENIZED_METHOD, REPRESENT_METHOD, RETRIEVE_METHOD, AGGREGATE_METHOD)