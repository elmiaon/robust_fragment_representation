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

def sample_CLSR(args):
    '''
    sample CLSR dataset from the original corpus
    input: (corpus, sub_corpus, parent_corpus, parent_sub_corpus, s, t, condition_tuple, return_remain)
    output: sampled dataset saved in data/reformatted/ directory
    '''
    corpus, sub_corpus, parent_corpus, parent_sub_corpus, s, t, condition_tuple, return_remain = args
    np.random.seed(42)
    logger = log.get_logger(__name__)
    n_out_s, n_out_t, n_out_close = condition_tuple

    corpus_dir = f"data/reformatted/{corpus}/{sub_corpus}"
    utils.make_dir(corpus_dir)
    s_fwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{s}.csv"
    t_fwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{t}.csv"
    g_fwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.gold.csv"

    s_bwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{s}.csv"
    t_bwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{t}.csv"
    g_bwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.gold.csv"
    
    if return_remain:
        utils.make_dir(f"{corpus_dir}Remain")
        s_fwd_remain_dir = f"data/reformatted/{corpus}/{sub_corpus}Remain/{s}-{t}.{s}.csv"
        t_fwd_remain_dir = f"data/reformatted/{corpus}/{sub_corpus}Remain/{s}-{t}.{t}.csv"
        g_fwd_remain_dir = f"data/reformatted/{corpus}/{sub_corpus}Remain/{s}-{t}.gold.csv"

        s_bwd_remain_dir = f"data/reformatted/{corpus}/{sub_corpus}Remain/{t}-{s}.{s}.csv"
        t_bwd_remain_dir = f"data/reformatted/{corpus}/{sub_corpus}Remain/{t}-{s}.{t}.csv"
        g_bwd_remain_dir = f"data/reformatted/{corpus}/{sub_corpus}Remain/{t}-{s}.gold.csv"

        s_check = os.path.isfile(s_fwd_output_dir) or os.path.isfile(s_bwd_output_dir)
        t_check = os.path.isfile(t_fwd_output_dir) or os.path.isfile(t_bwd_output_dir)
        g_check = os.path.isfile(g_fwd_output_dir) or os.path.isfile(g_bwd_output_dir)

        skip = (s_check and t_check and g_check) and (s_remain_check and t_remain_check and g_remain_check)

    else:
        s_remain_check = os.path.isfile(s_fwd_remain_dir) or os.path.isfile(s_bwd_output_dir)
        t_remain_check = os.path.isfile(t_fwd_remain_dir) or os.path.isfile(t_bwd_output_dir)
        g_remain_check = os.path.isfile(g_fwd_remain_dir) or os.path.isfile(g_bwd_output_dir)

        skip = (s_check and t_check and g_check)

    if not skip:
        tic = time()
        sdf, tdf, gdf = load_reformatted(parent_corpus, parent_sub_corpus, s, t)
        toc = time()
        logger.info(f"loading reformatted data from {corpus} - {sub_corpus} in {toc-tic:.2f} second(s)")
        gdf = gdf.loc[gdf[s].isin(sdf['id']) & gdf[t].isin(tdf['id'])]

        if n_out_close > len(gdf):
            raise ValueError(f"cannot create corpus due to n_close > len(gdf)")
        if n_out_s > len(sdf):
            raise ValueError(f"cannot create corpus due to n_s > len(sdf)")
        if n_out_t > len(tdf):
            raise ValueError(f"cannot create corpus due to n_t > len(tdf)")
        if n_out_close > n_out_s or n_out_close > n_out_t:
            raise ValueError(f"cannot create corpus due to n_close > n_s or n_t")
            
        n_out_open_s = n_out_s-n_out_close
        n_out_open_t = n_out_t-n_out_close
        
        out_g, remain_g = np.split(gdf.sample(frac=1), [n_out_close])
        out_g = out_g.sort_index()
        remain_g = remain_g.sort_index()

        out_s, remain_s, remain_g = get_out_df(sdf, gdf, out_g, remain_g, n_out_open_s)
        out_t, remain_t, _ = get_out_df(tdf, gdf, out_g, remain_g, n_out_open_t)
        remain_g = remain_g.loc[remain_g[s].isin(remain_s['id']) & remain_g[t].isin(remain_t['id'])]

        out_s.to_csv(s_fwd_output_dir, index=False, sep='\t')
        out_t.to_csv(t_fwd_output_dir, index=False, sep='\t')
        out_g.to_csv(g_fwd_output_dir, index=False, sep='\t')
        
        if return_remain:
            remain_s.to_csv(s_fwd_remain_dir, index=False, sep='\t')
            remain_t.to_csv(t_fwd_remain_dir, index=False, sep='\t')
            remain_g.to_csv(g_fwd_remain_dir, index=False, sep='\t')

def get_out_df(in_df, gdf, out_g, remain_g, n_out_open_df):
    in_close_df, in_open_df = get_df_from_id(in_df, gdf)
    out_close_df, remain_close_df = get_df_from_id(in_close_df, out_g)
    out_open_df, remain_close_df, remain_open_df, remain_g = get_out_open_df(in_open_df, n_out_open_df, remain_close_df, remain_g)
    out_df = pd.concat([out_close_df, out_open_df]).sort_index()
    remain_df = in_df.loc[~in_df['id'].isin(out_df['id'])]
    return out_df, remain_df, remain_g

def get_df_from_id(df, gdf):
    sid, lang = df.columns
    return df.loc[df['id'].isin(gdf[lang])], df.loc[~df['id'].isin(gdf[lang])]

def get_out_open_df(in_open, n_out_open, remain_close, remain_g):
    n_in_open = len(in_open)
    if n_out_open > n_in_open+len(remain_g):
        raise ValueError("cannot create corpus due to n_out_open > n_in_open+n_remain_close")
    if n_in_open >= n_out_open:
        out_open, remain_open = np.split(in_open.sample(frac=1), [n_out_open])
    else:
        diff = n_out_open-n_in_open
        additional_open_g, remain_g = np.split(remain_g.sample(frac=1), [diff])
        additional_open, remain_close = get_df_from_id(remain_close, additional_open_g)
        out_open = pd.concat([in_open, additional_open])
        remain_open = in_open.loc[~in_open['id'].isin(out_open['id'])]
    return out_open, remain_close, remain_open, remain_g

def load_reformatted(corpus, sub_corpus, s, t):
    try:
        sdf = pd.read_csv(f"corpus/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{s}.csv", sep='\t', converters={f"{s}": literal_eval})
        tdf = pd.read_csv(f"corpus/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{t}.csv", sep='\t', converters={f"{t}": literal_eval})
        gdf = pd.read_csv(f"corpus/reformatted/{corpus}/{sub_corpus}/{s}-{t}.gold.csv", sep='\t')
    except(FileNotFoundError):
        sdf = pd.read_csv(f"corpus/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{s}.csv", sep='\t', converters={f"{s}": literal_eval})
        tdf = pd.read_csv(f"corpus/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{t}.csv", sep='\t', converters={f"{t}": literal_eval})
        gdf = pd.read_csv(f"corpus/reformatted/{corpus}/{sub_corpus}/{t}-{s}.gold.csv", sep='\t')
        gdf = gdf[[s, t]]
    return sdf, tdf, gdf