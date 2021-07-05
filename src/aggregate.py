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
# from src.prepare_corpus import load_prepared, create_temp_raw

##################################################
### define global                              ###
##################################################
retrieved_df = None
gold_df = None
translator = None

##################################################
### check input and output                     ###
##################################################
def check_tuned_output(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str, represent_method:list, retrieve_method:list, aggregate_method:list):
    '''
    checking output for skip
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    s: source language
    t: target language
    tokenize_method: string.
    represent_method: list. [represent method name, setting code]
    retrieve_method: list. [retrieve method name, k]
    aggregate_method: list. [aggregate method name, setting code]

    returns
    -------
    skip: bool. skip boolean to skip the tokenized process
    output_dir: list. output dict for save the tokenized sentences
    '''
    output_corpus_dir = f"data/tuned/{corpus}/{sub_corpus}/{'s'.join(aggregate_method)}" # define output dir
    utils.make_dir(f"{output_corpus_dir}") # create output dir
    output_dir = f"{output_corpus_dir}/{tokenize_method}.{'s'.join(represent_method)}.{'k'.join(retrieve_method)}.{s}-{t}.csv"

    # check output to skip
    if os.path.isfile(output_dir):
        return True, output_dir
    else:
        return False, output_dir

def check_tuned_input(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str, represent_method:list, retrieve_method:list):
    '''
    checking input
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    s: source language
    t: target language
    tokenize_method: string.
    represent_method: list. [represent method name, setting code]

    returns
    -------
    input_dir: list. output dict for save the tokenized sentences
    '''
    # check the tokenized dataset to be represented
    input_corpus_dir = f"data/retrieved/{corpus}/{sub_corpus}/{'k'.join(retrieve_method)}" # define input dir

    input_dir = f"{input_corpus_dir}/{tokenize_method}.{'s'.join(represent_method)}.{s}-{t}.csv"
    
    if not os.path.isfile(input_dir):
        raise ValueError(f"There is no retrieve {corpus}-{sub_corpus}.{s}-{t}")

    if os.path.isfile(f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.gold.csv"):
        gold_dir = f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.gold.csv"
    elif os.path.isfile(f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.gold.csv"):
        gold_dir = f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.gold.csv"
    else:
        raise ValueError(f"There is no reformatted gold file")
    
    return input_dir, gold_dir

##################################################
### tune aggregator                            ###
##################################################
def tune_aggregator(setting_code:int, corpus:str, sub_corpus:str, s:str, t:str):
    '''
    tuning the parameter for aggregator

    parameters
    ----------
    setting_code: int. setting coder to get the experiment parameters
    corpus: string. corpus to be tuned
    sub_corpus: string. sub corpus to be tuned
    s: string. source langugae to be tuned
    t: string. target language to be tuned

    returns
    -------
    tuned dataset files: csv. saved in data/tuned/ directory
    '''

    global retrieved_df, gold_df
    logger = log.get_logger(__name__)

    _, tokenize_method, represent_method, retrieve_method, aggregate_method = utils.get_experiment_setting(setting_code) # get settings

    skip, output_dir = check_tuned_output(corpus, sub_corpus, s, t, tokenize_method, represent_method, retrieve_method, aggregate_method) # check output existance

    if not skip:

        input_dir, gold_dir = check_tuned_input(corpus, sub_corpus, s, t, tokenize_method, represent_method, retrieve_method) # check and get exist input dir

        method, aggregate_setting_code = aggregate_method # unpack aggregate_method

        args = get_aggregate_setting(method, aggregate_setting_code)

        gold_df = pd.read_csv(gold_dir, sep='\t') # load gold
        gold_df = gold_df[[s, t]] # reorder columns
        gold_df.columns = ['id', 'pair'] # rename gold_df columns
        # print(f"gold_df:\n{gold_df}")

        retrieved_df = pd.read_csv(input_dir, sep='\t', converters={'candidates': literal_eval, 'distance': utils.byte_decode})

        if method == 'RFR':
            retrieved_df['distance'] = retrieved_df['distance'].parallel_apply(kernel_dist) # convert to kernel similarity
            retrieved_df = group_by_sid(retrieved_df) # group retrived fragments by sentence id
            retrieved_df.columns = ['id', 'candidates', 'similarity']
            aggregate = RFR_aggregator
        else:
            raise ValueError(f"invalid aggregator tuning method")
        scores_df = aggregate(*args)
        scores_df.to_csv(output_dir, sep='\t', index=False)
    
    else:
        scores_df = pd.read_csv(output_dir, sep='\t')
    
    at_1 = scores_df.loc[scores_df['n']==1]
    params_set = at_1.loc[at_1['align_f1']==at_1['align_f1'].max()]
    print(params_set)
    params = at_1.loc[at_1['align_f1']==at_1['align_f1'].max()].tail(1)
    print(params)


##################################################
### tuning methods                             ###
##################################################
def RFR_aggregator(k_list, beta_list, filter_thres, p_thres, at):
    '''
    parameters
    ----------
    k_list: np array. list of k in k-NN to be included in aggration
    beta_list: np array. list of spiking coefficient
    filter_thre: np array. list of minimum entropy portion to be keep
    p_thres: np array. list of probability thresold
    at: np array. list of p@n_top

    returns
    -------
    tuned_score: csv. saved in data/tuned/ directory
    best_params: set of best tuning score params
    '''

    params_set = [k_list, beta_list, filter_thres, p_thres, [at], [False]] # generate parameter set to save

    # params_df = pd.DataFrame(itertools.product(*params_set), columns=['k', 'beta', 'fil', 'pthres', 'at']) # create parameter dataframe
    
    # # test single
    # score, TP, TN, FP, FN, FA = get_RFR_result((5, 50, 0.2, 0.3, [1, 5, 10], True))
    # print(f"score: {score}")
    # print(f"TP:{len(TP)}\n{TP}")
    # print(f"TN:{len(TN)}\n{TN}")
    # print(f"FP:{len(FP)}\n{FP}")
    # print(f"FN:{len(FN)}\n{FN}")
    # print(f"FA:{len(FA)}\n{FA}")
    
    # test batch
    # k_list = np.array([5, 10])
    # beta_list = np.array([50, 100])
    # filter_thres = np.array([0.1, 0.5, 1])
    # p_thres = np.array([0, 0.5, 1])
    # params_set = [k_list, beta_list, filter_thres, p_thres, [at], [False]] # create parameter set to tune 

    mp_input = itertools.product(*params_set) # do permutation
    with mp.Pool(processes=PROCESS_NUM) as p:
        scores = p.map(get_RFR_result, mp_input)
    
    scores_df = pd.concat(scores)
    return scores_df

    
def get_RFR_result(args):
    global retrieved_df
    k, beta, fil, p_thre, at, analyse = args
    aggregated_df = retrieved_df.apply(get_RFR_aggregated, args=(k, beta, fil, p_thre), axis=1)
    scores = []
    for n in at:
        ans_df = aggregated_df.copy()
        ans_df['candidates'] = ans_df['candidates'].apply(lambda x: x[:n])
        score, analysis_component_ids = cal_score(ans_df, gold_df)
        if n == 1 and analyse:
            TP, TN, FP, FN, FA = get_analysis_components(aggregated_df, analysis_component_ids)
        scores.append(np.concatenate([[k, beta, fil, p_thre, n], score], axis=None))
    score_df = pd.DataFrame(scores, columns=['k', 'beta', 'fil', 'p_thres', 'n', 'acc', 'fil_p', 'fil_r', 'fil_f1', 'align_p', 'align_r', 'align_f1']).convert_dtypes()
    
    if beta==100 and fil==1 and p_thre==1:
        print(f"finish {args}")

    if analyse:
        return score_df, TP, TN, FP, FN, FA
    else:
        return score_df

def get_RFR_aggregated(row, k, beta, fil, p_thre):
    '''
    get aggregated answer using FRF method

    parameters
    ----------
    row: df row. row(sentence id) to get the aggregated answer    
    k: int. k of k-NN to be included in aggration
    beta: int. spiking coefficient
    fil: float. minimum entropy portion to be keep
    p_thre: float. probability thresold

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
    if unique_prob[0] < p_thre:
        row['ans'] = False
    else:
        row['ans'] = True
    return row

def cal_score(aggregated_df, gold_df):


    ans_df = aggregated_df.loc[aggregated_df['ans'] == True][['id', 'candidates', 'prob']] # answered candidates
    fil_df = aggregated_df.loc[aggregated_df['ans'] == False][['id', 'candidates', 'prob']] # filtered candidates

    n_ans = len(ans_df)
    n_aggregated = len(aggregated_df)
    n_gold = len(gold_df)

    if fil_df.empty:
        FN_df = pd.DataFrame(columns=aggregated_df.columns)
        TN_df = pd.DataFrame(columns=aggregated_df.columns)
        n_TN = 0
    else:
        FN_df = fil_df.loc[fil_df['id'].isin(gold_df['id'])] # false negative (filtered out answers)
        TN_df = fil_df.loc[~fil_df['id'].isin(gold_df['id'])] # true negative (correctly filtered)
        n_TN = len(TN_df)
    
    if ans_df.empty:
        FP_df = pd.DataFrame(columns=ans_df.columns)
        TP_df = pd.DataFrame(columns=ans_df.columns)
        FA_df = pd.DataFrame(columns=ans_df.columns)
        n_TP = 0
        fil_p, fil_r, fil_f1, align_p, align_r, align_f1 = 0, 0, 0, 0, 0, 0
    else:
        FP_df = ans_df.loc[~ans_df['id'].isin(gold_df['id'])] # false positive (answers which are not in gold)
        hit_df = ans_df.loc[ans_df['id'].isin(gold_df['id'])] # answers which are included in gold

        merge_df = pd.merge(gold_df, hit_df, left_on='id', right_on='id')
        merge_df = merge_df.apply(validate_ans, axis=1)

        TP_df = merge_df.loc[merge_df['correct'] == True][['id', 'candidates', 'prob']]
        FA_df = merge_df.loc[merge_df['correct'] == False][['id', 'candidates', 'prob']]

        n_hit = len(hit_df)
        n_TP = len(TP_df)

        fil_p = n_hit/n_ans
        fil_r = n_hit/n_gold
        fil_f1 = f1(fil_p, fil_r)

        align_p = n_TP/n_ans
        align_r = n_TP/n_gold
        align_f1 = f1(align_p, align_r)

    acc = (n_TN + n_TP)/n_aggregated

    return np.array([acc, fil_p, fil_r, fil_f1, align_p, align_r, align_f1]), [TP_df['id'], TN_df['id'], FP_df['id'], FN_df['id'], FA_df['id']]

def get_analysis_components(df, analysis_component_ids):
    return [df.loc[df['id'].isin(ids)][['id', 'candidates', 'prob']] for ids in analysis_component_ids]
    
def validate_ans(row):
    row['correct'] = row['pair'] in row['candidates']
    return row

def f1(precision, recall):
    return (2*precision*recall)/(precision+recall)
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