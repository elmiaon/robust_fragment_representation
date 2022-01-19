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
import tqdm
# multiprocess lib
import multiprocessing as mp
PROCESS_NUM = mp.cpu_count()-2
# custom lib
import src.utils as utils
import src.cal_score as cal_score


##################################################
### tune and test                              ###
##################################################
def tune(setting_code:str, input_dir:str, gold_dir:str, output_dir:str):

    filename = input_dir.split('/')[-1]
    lang_pair = filename.split('.')[-2]
    s, t = lang_pair.split('-')

    gold_df = pd.read_csv(gold_dir, sep='\t') # load gold
    gold_df = gold_df[[s, t]] # reorder columns
    gold_df.columns = ['id', 'pair'] # rename gold_df columns

    retrieved_df = pd.read_csv(input_dir, sep='\t', converters={'candidates': literal_eval, 'distance': utils.byte_decode})
    kerneled_df = get_kernel_retrieved_df(retrieved_df)

    k_list, beta_list, filter_thres, p_thres, at = RFR_parser(setting_code)

    params = itertools.product(k_list, beta_list, filter_thres, p_thres)
    params_set = [[kerneled_df], [gold_df], params, [at], [False]] # generate parameter set to save

    mp_input = itertools.product(*params_set) # do permutation

    scores = []
    with mp.Pool(processes=PROCESS_NUM) as p:
        for score in tqdm.tqdm(p.imap_unordered(get_RFRa_result, mp_input), total=len(k_list)*len(beta_list)*len(filter_thres)*len(p_thres)):
            scores.append(score)

    scores_df = pd.concat(scores)
    scores_df.to_csv(output_dir, sep='\t', index=False)

    return scores_df

def random_tune(setting_code:str, input_dir:str, gold_dir:str, output_dir:str, n_points=2000, seed=42):
    filename = input_dir.split('/')[-1]
    lang_pair = filename.split('.')[-2]
    s, t = lang_pair.split('-')

    np.random.seed(seed)

    gold_df = pd.read_csv(gold_dir, sep='\t') # load gold
    gold_df = gold_df[[s, t]] # reorder columns
    gold_df.columns = ['id', 'pair'] # rename gold_df columns

    retrieved_df = pd.read_csv(input_dir, sep='\t', converters={'candidates': literal_eval, 'distance': utils.byte_decode})
    kerneled_df = get_kernel_retrieved_df(retrieved_df)

    K, B, F, P, at = RFR_parser(setting_code)

    k_min, k_max = K[0], K[-1]
    beta_min, beta_max = B[0], B[-1]
    filt_min, filt_max = F[0], F[-1]
    p_min, p_max = P[0], P[-1]

    k_list = np.random.randint(low=k_min, high=k_max+1, size=n_points)
    beta_list = convert_to_range(np.random.rand(n_points), 0, 1, beta_min, beta_max)
    filter_thres = convert_to_range(np.random.rand(n_points), 0, 1, filt_min, filt_max)
    p_thres = convert_to_range(np.random.rand(n_points), 0, 1, p_min, p_max)

    params = zip(k_list, beta_list, filter_thres, p_thres)
    params_set = [[kerneled_df], [gold_df], params, [at], [False]] # generate parameter set to save
    mp_input = itertools.product(*params_set) # do permutation

    scores = []
    with mp.Pool(processes=PROCESS_NUM) as p:
        for score in tqdm.tqdm(p.imap_unordered(get_RFRa_result, mp_input), total=n_points):
            scores.append(score)
        # scores = p.map(get_RFRa_result, mp_input)
    
    scores_df = pd.concat(scores)
    scores_df.to_csv(output_dir, sep='\t', index=False)

    return scores_df

def test(setting_code:str, input_dir:str, gold_dir:str, 
         score_output_dir:str, ans_output_dir:str, 
         TP_output_dir:str, TN_output_dir:str, 
         FP_output_dir:str, FN_output_dir:str, 
         FA_output_dir:str, 
         params):

    filename = input_dir.split('/')[-1]
    lang_pair = filename.split('.')[-2]
    s, t = lang_pair.split('-')

    gold_df = pd.read_csv(gold_dir, sep='\t') # load gold
    gold_df = gold_df[[s, t]] # reorder columns
    gold_df.columns = ['id', 'pair'] # rename gold_df columns

    retrieved_df = pd.read_csv(input_dir, sep='\t', converters={'candidates': literal_eval, 'distance': utils.byte_decode})
    kerneled_df = get_kernel_retrieved_df(retrieved_df)

    k, beta, fil, p_thres, at = RFR_parser(setting_code)

    k, beta, fil, p_thres = params
    k = int(k)

    args = (kerneled_df, gold_df, (k, beta, fil, p_thres), at, True)

    score_df, ans, TP, TN, FP, FN, FA = get_RFRa_result(args)

    score_df.to_csv(score_output_dir, sep='\t', index=False)
    ans.to_csv(ans_output_dir, sep='\t', index=False)
    TP.to_csv(TP_output_dir, sep='\t', index=False) 
    TN.to_csv(TN_output_dir, sep='\t', index=False) 
    FP.to_csv(FP_output_dir, sep='\t', index=False) 
    FN.to_csv(FN_output_dir, sep='\t', index=False) 
    FA.to_csv(FA_output_dir, sep='\t', index=False)

    return score_df, ans, TP, TN, FP, FN, FA


##################################################
### get RFRa result                            ###
##################################################
def get_RFRa_result(args):
    kerneled_df, gold_df, (k, beta, fil, p_thres), at, analyse = args
    aggregated_df = kerneled_df.apply(get_RFRa_aggregated, args=(k, beta, fil, p_thres), axis=1)
    scores = []
    for n in at:
        ans_df = aggregated_df.copy()
        ans_df['candidates'] = ans_df['candidates'].apply(lambda x: x[:n])
        score, analysis_component_ids = cal_score.cal_score(ans_df, gold_df)
        if n == 1 and analyse:
            ans, TP, TN, FP, FN, FA = cal_score.get_analysis_components(aggregated_df, analysis_component_ids)
        scores.append(np.concatenate([[k, beta, fil, p_thres, n], score], axis=None))
    score_df = pd.DataFrame(scores, columns=['k', 'beta', 'fil', 'p_thres', 'n', 'acc', 'fil_p', 'fil_r', 'fil_f1', 'align_p', 'align_r', 'align_f1']).convert_dtypes()

    if analyse:
        return score_df, ans, TP, TN, FP, FN, FA
    else:
        return score_df

##################################################
### RFR aggregator                             ###
##################################################
def get_RFRa_aggregated(row, k, beta, fil, p_thres):
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
    row['candidates'] = tuple(unique_candidates)
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

def RFR_parser(setting_code):

    with open('config/aggregate.json') as f: # load RFR-CLSR json setting
        setting_dict = json.load(f)
    setting_code = str(setting_code)
    
    if not setting_code in setting_dict['RFRa'].keys(): # check setting code existance
        raise ValueError(f"invalid {setting_code} aggregate setting_code")
    
    setting = setting_dict['RFRa'][setting_code]

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

def convert_to_range(a, amin, amax, vmin, vmax):
    return vmin+((vmax-vmin)*a)/(amax-amin)