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

def get_RFR_result(args):
    kerneled_df, gold_df, k, beta, fil, p_thres, at, analyse = args
    aggregated_df = kerneled_df.apply(get_RFR_aggregated, args=(k, beta, fil, p_thres), axis=1)
    scores = []
    for n in at:
        ans_df = aggregated_df.copy()
        ans_df['candidates'] = ans_df['candidates'].apply(lambda x: x[:n])
        score, analysis_component_ids = cal_score.cal_score(ans_df, gold_df)
        if n == 1 and analyse:
            TP, TN, FP, FN, FA = get_analysis_components(aggregated_df, analysis_component_ids)
        scores.append(np.concatenate([[k, beta, fil, p_thres, n], score], axis=None))
    score_df = pd.DataFrame(scores, columns=['k', 'beta', 'fil', 'p_thres', 'n', 'acc', 'fil_p', 'fil_r', 'fil_f1', 'align_p', 'align_r', 'align_f1']).convert_dtypes()
    
    if beta==100 and fil==1 and p_thres==1:
        print(f"finish {(k, beta, fil, p_thres, at, analyse)}")

    if analyse:
        return score_df, TP, TN, FP, FN, FA
    else:
        return score_df

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