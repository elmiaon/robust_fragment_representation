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
import src.cal_score as cal_score

##################################################
### RFR aggregator                             ###
##################################################

def get_RFR_aggregated(row, k, beta, fil, p_thres):
    '''
    get aggregated answer using FRF method

    parameters
    ----------
    row: df row. row(sentence id) to get the aggregated answer    
    k: int. k of k-NN to be included in aggration
    beta: int. spiking coefficient
    fil: float. minimum entropy portion to be keep
    p_thres: float. probability thresold

    returns
    -------
    row: df row. aggreted row composed of
        id - sentence id
        candidates - list of candidates
        prob - probability of each candidates
        ans - if true this sentence id will be include in the answer set, filted out otherwise.
    '''
    candidates = row.pop('candidates')
    similarity = row.pop('similarity')

    # get n_min_similarity candidates
    candidates = candidates[:,:k]
    similarity = similarity[:,:k]
    # convert similarity to probability
    prob = np.exp(beta*similarity)
    prob = prob/np.sum(prob, axis=1).reshape(-1,1)
    del similarity
    # calculate entropy and filter only p% min of entropy 
    ent = entropy(prob, axis=1)
    sorted_ent_idx = np.argsort(ent)[:int(np.ceil(len(ent)*fil))]
    candidates = candidates[sorted_ent_idx]
    prob = prob[sorted_ent_idx]
    # aggregate
    unique_candidates = np.unique(candidates)
    unique_prob = ndimage.sum(prob, candidates, unique_candidates)/len(prob)
    sorted_unique_idx = np.argsort(-unique_prob)
    # sort aggregated
    unique_candidates = unique_candidates[sorted_unique_idx]
    unique_prob = unique_prob[sorted_unique_idx]
    row['candidates'] = unique_candidates
    row['prob'] = unique_prob
    if unique_prob[0] < p_thres:
        row['ans'] = False
    else:
        row['ans'] = True
    return row


##################################################
### common functions                           ###
##################################################

def kernel_dist(dist):
    return(  np.exp( -(np.power(dist,2))/2 ) / np.sqrt(2*np.pi)  )

def group_by_sid(df):
    df = df[['id', 'candidates', 'distance']]
    df['s_candidates'] = df.parallel_apply(lambda x: [i.split('|')[0] for i in x['candidates']], axis=1)
    gp = df.groupby('id')
    sid = list(gp.groups.keys())
    candidates, distance = zip(  *gp.apply(lambda x: (np.vstack(x['s_candidates']), np.vstack(x['distance'])))  )
    return pd.DataFrame({'id': sid, 'candidates': candidates, 'distance': distance})

def get_kernel_retrieved_df(retrieved_df):
    kerneled_df = retrieved_df.copy()
    kerneled_df['distance'] = kerneled_df['distance'].parallel_apply(kernel_dist) # convert to kernel similarity
    kerneled_df = group_by_sid(kerneled_df) # group retrived fragments by sentence id
    kerneled_df.columns = ['id', 'candidates', 'similarity']
    return kerneled_df

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