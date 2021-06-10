import os
import base64
import itertools
import json
import numpy as np
import os
import pandas as pd
import pickle

import src.log as log

def make_dir(path:str):
    '''
    create directory along the path if the path is not exist
    input: path(str) - path of directory to create
    output: None
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def byte_encode(obj):
    '''
    encode object as byte string
    input: obj(object) - object to be encode
    output: byte string(str): encoded object as a bytestring
    '''
    b64_bytes = base64.b64encode(pickle.dumps(obj))
    return b64_bytes.decode('utf8')

def byte_decode(b64_str:str):
    '''
    decode bytestring to object
    input: b64_str(str) - encoded object as a bytestring
    output: obj(object) - object after decoding
    '''
    return pickle.loads(base64.b64decode(b64_str))

def get_RFR_CLSR_setting(setting_code:int):
    '''
    get setting from setting_name
    input: setting_code(int) - setting code to get the setting_dict
    output: RFR-CLSR experiment setting(tuple) - parameters to run an experiment composed with
        DESCRIPTION(str) - experiment description
        TOKENIZED_METHOD(str) - tokenize method name
        REPRESENT_METHOD(list) - [represent method, represent_setting]
        RETRIEVE_METHOD(list) - [distance function, k-NN]
        AGGREGATE_METHOD(list) - [aggregate method, aggregate_setting]
    '''
    logger = log.get_logger(__name__) # get logger instance
    with open('config/RFR-CLSR.json') as f: # load RFR-CLSR json setting
        setting_dict = json.load(f)
    setting_code = str(setting_code)
    if not setting_code in setting_dict['setting'].keys(): # check setting code existance
        raise ValueError('invalid experiment setting_name')

    setting = setting_dict['setting'][setting_code]
    keys = setting.keys()
    if 'description' in keys: # get description
        DESCRIPTION = setting['description']
    else:
        raise ValueError('missing experiment description')
    
    if 'tokenized_method' in keys: # get tokenize method
        TOKENIZE_METHOD = setting['tokenized_method']
    else:
        TOKENIZE_METHOD = 'RFR'
    
    if 'represent_method' in keys: # get represent method
        REPRESENT_METHOD = setting['representation_method']
    else:
        REPRESENT_METHOD = ['RFR',0] # representation method, setting code

    if 'retrieve_method' in keys: # get retrieve method
        RETRIEVE_METHOD = setting['retrieve_method']
    else:
        RETRIEVE_METHOD = ['cosine', 50] # retrieve method, k-NN
    
    if 'aggregate_method' in keys: # get aggregate method
        AGGREGATE_METHOD = setting['aggregate_method']
    else:
        AGGREGATE_METHOD = ['RFR', 0]

    return (DESCRIPTION, TOKENIZE_METHOD, REPRESENT_METHOD, RETRIEVE_METHOD, AGGREGATE_METHOD)