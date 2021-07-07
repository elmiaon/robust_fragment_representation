##################################################
### import                                     ###
##################################################
# basic lib
import numpy as np
import os
import pandas as pd
# logging lib
import logging
import src.log as log
# time lib
from time import time
# custom lib
import src.utils as utils

##################################################
### sample CLSR                                ###
##################################################

def sample_CLSR(args:tuple):
    '''
    sample CLSR dataset from the reformatted original corpus
    input: args(tuple) composed of
        corpus(str), sub_corpus(str) - output corpus name
        parent_corpus, parent_sub_corpus - corpus and sub corpus to be sampled
        s(str), t(str) - source and target language to be sampled, respectively
        condition_tuple(tuple) - setting for sampling depend on each method
        return_remain(bool) - flag to save remainer from sampled
    output: sampled dataset saved in data/reformatted/ directory
    '''
    corpus, sub_corpus, parent_corpus, parent_sub_corpus, s, t, condition_tuple, return_remain = args # unpack args
    np.random.seed(42) # set random seed
    logger = log.get_logger(__name__)
    n_out_s, n_out_t, n_out_close = condition_tuple # unpack condition tuple

    # check the reformatted dataset to be sampled
    input_corpus_dir = f"data/reformatted/{parent_corpus}/{parent_sub_corpus}" # define input dir
    if os.path.isfile(f"{input_corpus_dir}/{s}-{t}.{s}.csv") and \
       os.path.isfile(f"{input_corpus_dir}/{s}-{t}.{t}.csv") and \
       os.path.isfile(f"{input_corpus_dir}/{s}-{t}.gold.csv"):
        pass
    elif os.path.isfile(f"{input_corpus_dir}/{t}-{s}.{s}.csv") and \
         os.path.isfile(f"{input_corpus_dir}/{t}-{s}.{t}.csv") and \
         os.path.isfile(f"{input_corpus_dir}/{t}-{s}.gold.csv"):
        pass
    else: # error if there is no reformatted file
        raise FileExistsError(f"There is no reformatted {parent_corpus}-{parent_sub_corpus}")

    corpus_dir = f"data/reformatted/{corpus}/{sub_corpus}" # define output dir
    utils.make_dir(corpus_dir) # create output dir

    # fwd output dir to store the sampled and check existance
    s_fwd_output_dir = f"{corpus_dir}/{s}-{t}.{s}.csv"
    t_fwd_output_dir = f"{corpus_dir}/{s}-{t}.{t}.csv"
    g_fwd_output_dir = f"{corpus_dir}/{s}-{t}.gold.csv"

    # bwd output dir to check existance of sampled
    s_bwd_output_dir = f"{corpus_dir}/{t}-{s}.{s}.csv"
    t_bwd_output_dir = f"{corpus_dir}/{t}-{s}.{t}.csv"
    g_bwd_output_dir = f"{corpus_dir}/{t}-{s}.gold.csv"
    
    # check to skip
    fwd_check = os.path.isfile(s_fwd_output_dir) and os.path.isfile(t_fwd_output_dir) and os.path.isfile(g_fwd_output_dir)
    bwd_check = os.path.isfile(s_bwd_output_dir) and os.path.isfile(t_bwd_output_dir) and os.path.isfile(g_bwd_output_dir)
    skip = (fwd_check or bwd_check)
    
    if return_remain:
        utils.make_dir(f"{corpus_dir}Remain") # create remain output dir

        # fwd remain output dir to store the sampled and check existance
        s_fwd_remain_dir = f"{corpus_dir}Remain/{s}-{t}.{s}.csv"
        t_fwd_remain_dir = f"{corpus_dir}Remain/{s}-{t}.{t}.csv"
        g_fwd_remain_dir = f"{corpus_dir}Remain/{s}-{t}.gold.csv"

        # bwd remain output dir to check existance of sampled
        s_bwd_remain_dir = f"{corpus_dir}Remain/{t}-{s}.{s}.csv"
        t_bwd_remain_dir = f"{corpus_dir}Remain/{t}-{s}.{t}.csv"
        g_bwd_remain_dir = f"{corpus_dir}Remain/{t}-{s}.gold.csv"

        #check to skip
        fwd_remain_check = os.path.isfile(s_fwd_remain_dir) and os.path.isfile(t_fwd_remain_dir) and os.path.isfile(g_fwd_remain_dir)
        bwd_remain_check = os.path.isfile(s_bwd_output_dir) and os.path.isfile(t_bwd_output_dir) and os.path.isfile(g_bwd_output_dir)
        skip = (fwd_check and fwd_remain_check) or (bwd_check and bwd_remain_check)

    if not skip:
        tic = time()
        sdf, tdf, gdf = load_reformatted(parent_corpus, parent_sub_corpus, s, t) # load reformatted
        toc = time()
        logger.info(f"loading reformatted data from {corpus} - {sub_corpus} in {toc-tic:.2f} second(s)")
        gdf = gdf.loc[gdf[s].isin(sdf['id']) & gdf[t].isin(tdf['id'])] # make gdf contain only valid source and target

        if n_out_close > len(gdf): # close-sampled cannot more than available sentence in gdf
            raise ValueError(f"cannot create corpus due to n_close > len(gdf)")
        if n_out_s > len(sdf): # number of source sample cannot more than original
            raise ValueError(f"cannot create corpus due to n_s > len(sdf)")
        if n_out_t > len(tdf): # number of target sample cannot more than original
            raise ValueError(f"cannot create corpus due to n_t > len(tdf)")
        if n_out_close > n_out_s or n_out_close > n_out_t: # number of close-sampled cannot more than total sentence
            raise ValueError(f"cannot create corpus due to n_close > n_s or n_t")
        
        n_out_open_s = n_out_s-n_out_close # calculate number of open source sample
        n_out_open_t = n_out_t-n_out_close # calculate number of open target sample
        
        out_g, remain_g = np.split(gdf.sample(frac=1), [n_out_close]) # random close sentence id and remain close sentence id from gold
        out_g = out_g.sort_index() # sort close sample by id
        remain_g = remain_g.sort_index() # sort close remain by id

        out_s, remain_s, remain_g = get_out_df(sdf, gdf, out_g, remain_g, n_out_open_s) # get sampled and remain source, and gold
        out_t, remain_t, _ = get_out_df(tdf, gdf, out_g, remain_g, n_out_open_t) # get sampled and remain target
        remain_g = remain_g.loc[remain_g[s].isin(remain_s['id']) & remain_g[t].isin(remain_t['id'])] # get remain gold

        # save sampled to csv
        out_s.to_csv(s_fwd_output_dir, index=False, sep='\t')
        out_t.to_csv(t_fwd_output_dir, index=False, sep='\t')
        out_g.to_csv(g_fwd_output_dir, index=False, sep='\t')
        
        if return_remain: # if return remain
            # save remain to csv
            remain_s.to_csv(s_fwd_remain_dir, index=False, sep='\t')
            remain_t.to_csv(t_fwd_remain_dir, index=False, sep='\t')
            remain_g.to_csv(g_fwd_remain_dir, index=False, sep='\t')

def get_out_df(in_df, gdf, out_g, remain_g, n_out_open_df):
    '''
    get sampled and remain dataframe
    input:
        in_df(dataframe) - original dataframe (can be either source or target)
        gdf(dataframe) - gold pair dataframe
        out_g(dataframe) - sample gold pair dataframe
        remain_g(dataframe) - remain gold pair dataframe
        n_out_open_df - number of open sample 
    output:
        out_df - sample dataframe (can be either source or target)
        remain_df - remain dataframe (can be either source or target)
        remain_g - remain gold pair dataframe
    '''
    in_close_df, in_open_df = get_df_from_id(in_df, gdf) # get close and open sentence from original gold pair
    out_close_df, remain_close_df = get_df_from_id(in_close_df, out_g) # get close and remain close from sampled gold pair
    out_open_df, remain_close_df, remain_open_df, remain_g = get_out_open_df(in_open_df, n_out_open_df, remain_close_df, remain_g) # get open sample, and remain (close, open, gold)
    out_df = pd.concat([out_close_df, out_open_df]).sort_index() # get sampled dataframe by concat close and open sample dataframe then sort by sentence id
    remain_df = in_df.loc[~in_df['id'].isin(out_df['id'])] # get remain dataframe
    return out_df, remain_df, remain_g # return sampled, and remain dataframe

def get_df_from_id(df, gdf):
    '''
    get sentences which are and are not the member of given gold dataframe
    input:
        df(dataframe) - input dataframe
        gdf(dataframe) - gold dataframe to get the sentences
    output:
        sentences in gdf(dataframe)
        sentences not in gdf(dataframe)
    '''
    sid, lang = df.columns # get lang from column
    return df.loc[df['id'].isin(gdf[lang])], df.loc[~df['id'].isin(gdf[lang])]

def get_out_open_df(in_open, n_out_open:int, remain_close, remain_g):
    '''
    get open sample dataframe along with remain source, target, gold dataframe
    input:
        in_open(dataframe) - original open sentences dataframe
        n_out_open(int) - number of open sample
        remain_close(dataframe) - remain close sentences after sample
        remain_g(dataframe) - remain gold pair after sample
    output:
        out_open(dataframe) - open sampled dataframe
        remain_close(dataframe) - remain close sentences after converts somes to open sentences (if have to)
        remain_open(dataframe) - remain open sentences after sampled
        remain_g(dataframe) - remain gold pair after converts somes to open sentences (if have to)
    '''
    n_in_open = len(in_open) # get the number of original open sentences
    if n_out_open > n_in_open+len(remain_g): # if number of open sample sentences is more than original close + original open sentences
        raise ValueError("cannot create corpus due to n_out_open > n_in_open+n_remain_close") # raise an error
    if n_in_open >= n_out_open: # if number of original open sentences is more than the number of open sample sentences
        out_open, remain_open = np.split(in_open.sample(frac=1), [n_out_open]) # just sample from original open sentences
    else: # number of original open sentences is less than the number of open sample sentences, convert some close to open sentences
        diff = n_out_open-n_in_open # calculate number of convert
        additional_open_g, remain_g = np.split(remain_g.sample(frac=1), [diff]) # sample some (diff) close to be open sample sentences from gold
        additional_open, remain_close = get_df_from_id(remain_close, additional_open_g) # get convert and remain sentences from converted gold
        out_open = pd.concat([in_open, additional_open]) # get open sample by concat original and converted from close open setnences
        remain_open = in_open.loc[~in_open['id'].isin(out_open['id'])] # get the remain open (empty dataframe actually)
    return out_open, remain_close, remain_open, remain_g

def load_reformatted(corpus, sub_corpus, s, t):
    '''
    load reformatted dataset
    input:
        corpus, sub_corpus - reformatted corpus and sub corpus to be loaded
        s(str), t(str) - source and target language to be loaded, respectively
    output:
        sdf(dataframe), tdf(dataframe), gdf(dataframe) - reformatted source, target, and gold, respectively
    '''
    corpus_dir = f"data/reformatted/{corpus}/{sub_corpus}"
    try:
        sdf = pd.read_csv(f"{corpus_dir}/{s}-{t}.{s}.csv", sep='\t')
        tdf = pd.read_csv(f"{corpus_dir}/{s}-{t}.{t}.csv", sep='\t')
        gdf = pd.read_csv(f"{corpus_dir}/{s}-{t}.gold.csv", sep='\t')
    except(FileNotFoundError):
        sdf = pd.read_csv(f"{corpus_dir}/{t}-{s}.{s}.csv", sep='\t')
        tdf = pd.read_csv(f"{corpus_dir}/{t}-{s}.{t}.csv", sep='\t')
        gdf = pd.read_csv(f"{corpus_dir}/{t}-{s}.gold.csv", sep='\t')
        gdf = gdf[[s, t]]
    return sdf, tdf, gdf