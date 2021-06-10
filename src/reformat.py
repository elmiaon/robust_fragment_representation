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
### reformat CLSR                              ###
##################################################

def reformat_CLSR(args):
    corpus, sub_corpus, s, t = args

    logger = log.get_logger(__name__)
    corpus_dir = f"data/reformatted/{corpus}/{sub_corpus}"
    utils.make_dir(corpus_dir)
    s_fwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{s}.csv"
    t_fwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.{t}.csv"
    g_fwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{s}-{t}.gold.csv"

    s_bwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{s}.csv"
    t_bwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.{t}.csv"
    g_bwd_output_dir = f"data/reformatted/{corpus}/{sub_corpus}/{t}-{s}.gold.csv"

    s_check = os.path.isfile(s_fwd_output_dir) or os.path.isfile(s_bwd_output_dir)
    t_check = os.path.isfile(t_fwd_output_dir) or os.path.isfile(t_bwd_output_dir)
    g_check = os.path.isfile(g_fwd_output_dir) or os.path.isfile(g_bwd_output_dir)

    if not (s_check and t_check and g_check):
        tic = time()
        sdf, tdf, gdf = load_raw_CLSR(corpus, sub_corpus, s, t)
        toc = time()
        logger.info(f"loading raw data from {corpus} - {sub_corpus} in {toc-tic:.2f} second(s)")

        if not s_check:
            tic = time()
            sdf[s].replace('', np.nan, inplace=True)
            sdf = sdf.dropna()
            sdf.to_csv(s_fwd_output_dir, index=False, sep='\t')
            toc = time()
            logger.info(f"prepare {s} dataset in {toc-tic:.2f} second(s)")
            logger.info(f"sample: {sdf}")
        
        if not t_check:
            tic = time()
            tdf[t].replace('', np.nan, inplace=True)
            tdf = tdf.dropna()
            tdf.to_csv(t_fwd_output_dir, index=False, sep = '\t')
            toc = time()
            logger.info(f"prepare {t} dataset in {toc-tic:.2f} second(s)")
            logger.info(f"sample: {tdf}")
        
        if not g_check:
            tic = time()
            gdf = gdf.loc[gdf[s].isin(sdf['id'].values) & gdf[t].isin(tdf['id'].values)]
            gdf.to_csv(g_fwd_output_dir, index=False, sep='\t')
            toc = time()
            logger.info(f"prepare gold in {toc-tic:.2f} second(s)")
            logger.info(f"sample: {gdf}")
        return True
    else:
        return False

##################################################
### common function for CLSR                   ###
##################################################

def load_raw_CLSR(corpus, sub_corpus, s, t):
    if corpus == "UN":
        if not sub_corpus in ['6way', 'devset', 'testset']:
            raise ValueError(f"{sub_corpus} is not sub-corpus of {corpus}")
        sdf, tdf, gdf = load_UN(sub_corpus, s, t)
    elif corpus == "BUCC":
        if not sub_corpus in ['sample', 'training']:
            raise ValueError(f"{sub_corpus} is not sub-corpus of {corpus}")            
        sdf, tdf, gdf = load_BUCC(sub_corpus, s, t)
    elif corpus == "europarl":
        if not sub_corpus in ['training']:
            raise ValueError(f"{sub_corpus} is not sub-corpus of {corpus}") 
        sdf, tdf, gdf = load_europarl(sub_corpus, s, t)
    elif corpus == 'opus':
        if not sub_corpus in ['JW300', 'TED2020', 'QED']:
            raise ValueError(f"{sub_corpus} is not sub-corpus of {corpus}") 
        sdf, tdf, gdf = load_opus(sub_corpus, s, t)
    else:
        raise ValueError(f"{corpus} is not supported")
    return sdf, tdf, gdf

def read_raw(filedir):
    with open(filedir, "r") as f:
        lines = f.read()
    data = lines.split('\n')
    return data

def create_id(lang, amount):
    sen_id = [f"{lang}-{i:0>8d}" for i in range(amount)]
    return sen_id

##################################################
### load CLSR dataset                          ###
##################################################
### UN ###
def create_UN_df(part, lang):
    data = read_raw(f"data/raw/UN/{part}/UNv1.0.{part}.{lang}")
    sid = create_id(lang, len(data))
    df = pd.DataFrame({"id": sid,
                        lang: data})
    return df

def load_UN(part, s, t):
    sdf = create_UN_df(part, s)
    tdf = create_UN_df(part, t)
    gdf = pd.DataFrame({s: sdf["id"],
                         t: tdf["id"]})
    return [sdf, tdf, gdf]

### BUCC ###
def load_BUCC(part, s, t):
    try:
        sdf = pd.read_csv(f"data/raw/BUCC/{part}/{s}-{t}.{part}.{s}", sep="\t", names=["id", s])
        tdf = pd.read_csv(f"data/raw/BUCC/{part}/{s}-{t}.{part}.{t}", sep="\t", names=["id", t])
        gdf = pd.read_csv(f"data/raw/BUCC/{part}/{s}-{t}.{part}.gold", sep="\t", names=[s, t])
    except(FileNotFoundError):
        sdf = pd.read_csv(f"data/raw/BUCC/{part}/{t}-{s}.{part}.{s}", sep="\t", names=["id", s])
        tdf = pd.read_csv(f"data/raw/BUCC/{part}/{t}-{s}.{part}.{t}", sep="\t", names=["id", t])
        gdf = pd.read_csv(f"data/raw/BUCC/{part}/{t}-{s}.{part}.gold", sep="\t", names=[t, s])
        gdf = gdf[[s, t]]
    finally:
        return sdf, tdf, gdf

### europarl ###
def create_europarl_df(part, s, t, lang):
    try:
        data = read_raw(f"data/raw/europarl/{part}/europarl-v7.{s}-{t}.{lang}")
    except FileNotFoundError:
        data = read_raw(f"data/raw/europarl/{part}/europarl-v7.{t}-{s}.{lang}")
    sid = create_id(lang, len(data))
    df = pd.DataFrame({"id": sid,
                        lang: data})
    return df

def load_europarl(part, s, t):
    sdf = create_europarl_df(part, s, t, s)
    tdf = create_europarl_df(part, s, t, t)
    gdf = pd.DataFrame({s: sdf["id"],
                         t: tdf["id"]})
    return [sdf, tdf, gdf]

### opus ###
def create_opus_df(part, s, t, lang):
    try:
        data = read_raw(f"data/raw/opus/{part}/{s}-{t}.{lang}")
    except FileNotFoundError:
        data = read_raw(f"data/raw/opus/{part}/{t}-{s}.{lang}")
    sid = create_id(lang, len(data))
    df = pd.DataFrame({"id": sid,
                        lang: data})
    return df

def load_opus(part, s, t):
    sdf = create_opus_df(part, s, t, s)
    tdf = create_opus_df(part, s, t, t)
    gdf = pd.DataFrame({s: sdf["id"],
                         t: tdf["id"]})
    return [sdf, tdf, gdf]