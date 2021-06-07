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

def get_experiment_setting(exp_code:int):
    '''
    get part 2/2 of corpus creation setting from json file
    input: corpus alias
    output: (corpus, sub_corpus, parent_corpus, parent_sub_corpus, method, condition_tuple)
    '''
    logger = log.get_logger(__name__)
    with open('config/experiment_setting.json') as f:
        setting_dict = json.load(f)
    exp_code = str(exp_code)
    if not exp_code in setting_dict.keys():
        logger.info(f"available experiment code: {setting_dict.keys()}")
        logger.info(f"input experiment code: {exp_code}")
        raise ValueError('there is no setting for this alias')
    return setting_parser(setting_dict[exp_code])

def setting_parser(setting:dict):
    logger = log.get_logger(__name__)
    keys = setting.keys()
    if 'description' in keys:
        DESCRIPTION = setting['description']
    else:
        raise ValueError('missing experiment description')
    
    if 'preprocess_method' in keys:
        PREPROCESS_METHOD = setting['preprocess_method']
    else:
        PREPROCESS_METHOD = 'CON'
        # logger.warn(f"using default preprocess_method as {PREPROCESS_METHOD}")
    if not PREPROCESS_METHOD in ['EMD', 'PSA', 'USE', 'CON']:
        raise ValueError(f"preprocess method:{PREPROCESS_METHOD} is not supproted")

    # preprocess config
    if 'w2v' in keys:
        W2V = setting['w2v']
    else:
        W2V = None
        # logger.warn(f"using default word2vec as {W2V}")
    
    if 'n_grams' in keys:
        NG = setting['n_grams']
    else:
        NG = 6
        # logger.warn(f"using default n_grams as {NG}")

    if 'fragment_size' in keys:
        FS = setting['fragment_size']
    else:
        FS = 1
        # logger.warn(f"using default fragment_size as {FS}")


    if 'sentence_len' in keys:
        SL = setting['sentence_len']
    else:
        SL = 1
        # logger.warn(f"using default sentence_len as {SL}")

    if 'sentence2vec' in keys:
        S2V = setting['sentence2vec']
    else:
        S2V = 'USE'
        # logger.warn(f"using default sentence2vec as {S2V}")

    if S2V == 'ploy':
        transformer_num = setting['transformer_num']
        head_num = setting['head_num']
        feed_forward_dim = setting['feed_forward_dim']
        dropout_rate = setting['dropout_rate']
        load_model = setting['load_model']
        S2V_ARGS_LIST = [transformer_num, head_num, feed_forward_dim, dropout_rate, FS, load_model]
    else:
        S2V_ARGS_LIST = None
        # logger.warn(f"using default s2v argument list as {S2V_ARGS_LIST}")

    # distance calculation config
    if 'distance_calculation' in keys:
        DISTANCE_CALCULATION = setting['distance_calculation']
    else:
        DISTANCE_CALCULATION = 'cosine'
        # logger.warn(f"using default distance_calculation as {DISTANCE_CALCULATION}")

    # training config
    if 'n_min_distance' in keys:
        start, stop, step = setting['n_min_distance']
    else:
        start, stop, step = 5, 51, 5
        # logger.warn(f"using default n_min_distance as {start}, {stop}, {step}")
    N_MIN_DISTANCE = np.arange(start, stop, step)

    # tuning parameters
    if 'beta' in keys:
        start, stop, step = setting['beta']
    else:
        start, stop, step = 50, 101, 5
        # logger.warn(f"using default beta as {start}, {stop}, {step}")
    BETA = np.arange(start, stop, step)   
    
    if 'p_min_entropy' in keys:
        start, stop, step = setting['p_min_entropy']
    else:
        start, stop, step = 0.2, 1, 9
        # logger.warn(f"using default p_min_entropy as {start}, {stop}, {step}")
    P_MIN_ENTROPY = np.linspace(start, stop, step)

    if 'p_top' in keys:
        start, stop, step = setting['p_top']
    else:
        start, stop, step = 0.3, 1, 8
        # logger.warn(f"using default p_top as {start}, {stop}, {step}")
    P_TOP = np.linspace(start, stop, step)

    if 'n_retrieve' in keys:
        start, stop, step = setting['n_retrieve']
    else:
        start, stop, step = 1, 10, 2
        # logger.warn(f"using default n_retrieve as {start}, {stop}, {step}")
    N_RETRIEVE = np.arange(start, stop, step)

    return (DESCRIPTION, PREPROCESS_METHOD, W2V, NG, FS, SL, S2V, S2V_ARGS_LIST, DISTANCE_CALCULATION, N_MIN_DISTANCE, BETA, P_MIN_ENTROPY, P_TOP, N_RETRIEVE)