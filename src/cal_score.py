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

        n_hit = len(hit_df)

        if n_hit == 0:
            fil_p = 0
            fil_r = 0
            fil_f1 = 0
        else:
            fil_p = n_hit/n_ans
            fil_r = n_hit/n_gold
            fil_f1 = f1(fil_p, fil_r)

        merge_df = pd.merge(gold_df, hit_df, left_on='id', right_on='id')
        if merge_df.empty:
            TP_df = pd.DataFrame(columns=ans_df.columns)
            FA_df = pd.DataFrame(columns=ans_df.columns)
            n_TP = 0
        else:
            merge_df = merge_df.apply(validate_ans, axis=1)

            TP_df = merge_df.loc[merge_df['correct'] == True][['id', 'candidates', 'prob']]
            FA_df = merge_df.loc[merge_df['correct'] == False][['id', 'candidates', 'prob']]
            n_TP = len(TP_df)

        if n_TP == 0:
            align_p = 0
            align_r = 0
            align_f1 = 0
        else:
            align_p = n_TP/n_ans
            align_r = n_TP/n_gold
            align_f1 = f1(align_p, align_r)

    acc = (n_TN + n_TP)/n_aggregated

    return np.array([acc, fil_p, fil_r, fil_f1, align_p, align_r, align_f1]), [TP_df['id'], TN_df['id'], FP_df['id'], FN_df['id'], FA_df['id']]

def get_analysis_components(df, analysis_component_ids):
    
    df['prob'] = df['prob'].parallel_apply(utils.byte_encode)
    analysis_df = [df]
    for ids in analysis_component_ids:
        if ids.empty:
            temp = pd.DataFrame(columns=df.columns)
            temp = temp[['id', 'candidates', 'prob']]
        else:
            temp = df.loc[df['id'].isin(ids)][['id', 'candidates', 'prob']]
        analysis_df.append(temp)
    return analysis_df
    
def validate_ans(row):
    row['correct'] = row['pair'] in row['candidates']
    return row

def f1(precision, recall):
    return (2*precision*recall)/(precision+recall)