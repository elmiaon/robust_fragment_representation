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

def reformat_CLSR(args:tuple):
    '''
    reformat for CLSR dataset
    input : args(tuple) composed of
        corpus(str), sub_corpus(str) - corpus and sub corpus to be reformatted
        s(str), t(str) - source and target language to be reformatted, respectively
    output: reformatted dataset files(csv) - saved in data/reformatted/ directory
    '''
    corpus, sub_corpus, s, t = args # unpack args

    logger = log.get_logger(__name__) # get logger instance
    output_dir = f"data/reformatted/{corpus}/{sub_corpus}" # define output dir
    utils.make_dir(output_dir) # create output dir

    # fwd output dir to store the reformatted and check existance
    s_fwd_output_dir = f"{output_dir}/{s}-{t}.{s}.csv"
    t_fwd_output_dir = f"{output_dir}/{s}-{t}.{t}.csv"
    g_fwd_output_dir = f"{output_dir}/{s}-{t}.gold.csv"

    # bwd output dir to check existance of reformatted
    s_bwd_output_dir = f"{output_dir}/{t}-{s}.{s}.csv"
    t_bwd_output_dir = f"{output_dir}/{t}-{s}.{t}.csv"
    g_bwd_output_dir = f"{output_dir}/{t}-{s}.gold.csv"

    # check to skip
    fwd_check = os.path.isfile(s_fwd_output_dir) and os.path.isfile(t_fwd_output_dir) and os.path.isfile(g_fwd_output_dir)
    bwd_check = os.path.isfile(s_bwd_output_dir) and os.path.isfile(t_bwd_output_dir) and os.path.isfile(g_bwd_output_dir)

    s_check = os.path.isfile(s_fwd_output_dir) or os.path.isfile(s_bwd_output_dir)
    t_check = os.path.isfile(t_fwd_output_dir) or os.path.isfile(t_bwd_output_dir)
    g_check = os.path.isfile(g_fwd_output_dir) or os.path.isfile(g_bwd_output_dir)

    if not (fwd_check or bwd_check):
        tic = time()
        sdf, tdf, gdf = reformat_raw_CLSR(corpus, sub_corpus, s, t) # reformat raw CLSR data
        toc = time()
        logger.info(f"loading raw data from {corpus} - {sub_corpus} in {toc-tic:.2f} second(s)")

        if not s_check:
            tic = time()
            sdf[s].replace('', np.nan, inplace=True) # replace empty sentence with Nan
            sdf = sdf.dropna() # remove empty sentence
            sdf.to_csv(s_fwd_output_dir, index=False, sep='\t') #save
            toc = time()
            logger.info(f"reformatted {corpus}.{sub_corpus}.{s}-{t}.{s} dataset in {toc-tic:.2f} second(s)")
            logger.info(f"sample: {sdf}")
        
        if not t_check:
            tic = time()
            tdf[t].replace('', np.nan, inplace=True) # replace empty sentence with Nan
            tdf = tdf.dropna() # remove empty sentence
            tdf.to_csv(t_fwd_output_dir, index=False, sep = '\t') #save
            toc = time()
            logger.info(f"reformatted {corpus}.{sub_corpus}.{s}-{t}.{t} dataset in {toc-tic:.2f} second(s)")
            logger.info(f"sample: {tdf}")
        
        if not g_check:
            tic = time()
            gdf = gdf.loc[gdf[s].isin(sdf['id'].values) & gdf[t].isin(tdf['id'].values)] # remove empty pairs
            gdf.to_csv(g_fwd_output_dir, index=False, sep='\t') # save
            toc = time()
            logger.info(f"prepare gold in {toc-tic:.2f} second(s)")
            logger.info(f"sample: {gdf}")
        return True
    else:
        return False

##################################################
### common function for CLSR                   ###
##################################################

def reformat_raw_CLSR(corpus:str, sub_corpus:str, s:str, t:str):
    '''
    reformat for CLSR dataset from raw
    input :
        corpus(str), sub_corpus(str) - corpus and sub corpus to be reformatted
        s(str), t(str) - source and target language to be reformatted, respectively
    output: 
        sdf(dataframe) - reformatted source dataframe with [id, s] as columns
        tdf(dataframe) - reformatted target dataframe with [id, t] as columns
        gdf(dataframe) - gold pair dataframe with [s, t] as columns
    '''
    if corpus == "UN":
        if not sub_corpus in ['6way', 'devset', 'testset']:
            raise ValueError(f"{sub_corpus} is not sub-corpus of {corpus}")
        sdf, tdf, gdf = reformat_UN(sub_corpus, s, t)
    elif corpus == "BUCC":
        if not sub_corpus in ['sample', 'training']:
            raise ValueError(f"{sub_corpus} is not sub-corpus of {corpus}")            
        sdf, tdf, gdf = reformat_BUCC(sub_corpus, s, t)
    elif corpus == "europarl":
        if not sub_corpus in ['training']:
            raise ValueError(f"{sub_corpus} is not sub-corpus of {corpus}") 
        sdf, tdf, gdf = reformat_europarl(sub_corpus, s, t)
    elif corpus == 'opus':
        if not sub_corpus in ['JW300', 'TED2020', 'QED']:
            raise ValueError(f"{sub_corpus} is not sub-corpus of {corpus}") 
        sdf, tdf, gdf = reformat_opus(sub_corpus, s, t)
    else:
        raise ValueError(f"{corpus} is not supported")
    return sdf, tdf, gdf

def read_raw(filedir):
    '''
    reformat for CLSR dataset from raw
    input : filedir(str) - file dir to read
    output: data(list) - list of sentences splitted using newline
    '''
    with open(filedir, "r") as f:
        lines = f.read()
    data = lines.split('\n')
    return data

def create_id(lang, amount):
    '''
    create sentence id for specific language and amount
    input :
        lang(str) - language to create sentence id
        amount(int) - number of sentences
    output: sen_id(list) - list of sentence ids for specific language
    '''
    sen_id = [f"{lang}-{i:0>8d}" for i in range(amount)]
    return sen_id

##################################################
### load CLSR dataset                          ###
##################################################
### UN ###
def create_UN_df(part, lang):
    '''
    create UN dataframe for specific sub_corpus, and language
    input :
        part(str) - sub corpus
        lang(str) - specific language to be a column name
    output: df(dataframe) - with [id, lang] as columns
    '''
    data = read_raw(f"data/raw/UN/{part}/UNv1.0.{part}.{lang}")
    sid = create_id(lang, len(data))
    df = pd.DataFrame({"id": sid,
                        lang: data})
    return df

def reformat_UN(part, s, t):
    '''
    create UN dataframe for specific sub_corpus, and language
    input :
        part(str) - sub corpus
        s(str), t(str) - source and target language to be reformatted, respectively
    output: 
        sdf(dataframe) - reformatted source dataframe with [id, s] as columns
        tdf(dataframe) - reformatted target dataframe with [id, t] as columns
        gdf(dataframe) - gold pair dataframe with [s, t] as columns
    '''
    sdf = create_UN_df(part, s)
    tdf = create_UN_df(part, t)
    gdf = pd.DataFrame({s: sdf["id"],
                         t: tdf["id"]})
    return [sdf, tdf, gdf]

### BUCC ###
def reformat_BUCC(part, s, t):
    '''
    create BUCC dataframe for specific sub_corpus, and language
    input :
        part(str) - sub corpus
        lang(str) - specific language to be a column name
    output: 
        sdf(dataframe) - reformatted source dataframe with [id, s] as columns
        tdf(dataframe) - reformatted target dataframe with [id, t] as columns
        gdf(dataframe) - gold pair dataframe with [s, t] as columns
    '''
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
    '''
    create europarl dataframe for specific sub_corpus, and language
    input :
        part(str) - sub corpus
        s(str), t(str) - source and target language to be reformatted, respectively
        lang(str) - specific language to be a column name
    output: df(dataframe) - with [id, lang] as columns
    '''
    try:
        data = read_raw(f"data/raw/europarl/{part}/europarl-v7.{s}-{t}.{lang}")
    except FileNotFoundError:
        data = read_raw(f"data/raw/europarl/{part}/europarl-v7.{t}-{s}.{lang}")
    sid = create_id(lang, len(data))
    df = pd.DataFrame({"id": sid,
                        lang: data})
    return df

def reformat_europarl(part, s, t):
    '''
    create europarl dataframe for specific sub_corpus, and language
    input :
        part(str) - sub corpus
        s(str), t(str) - source and target language to be reformatted, respectively
    output: 
        sdf(dataframe) - reformatted source dataframe with [id, s] as columns
        tdf(dataframe) - reformatted target dataframe with [id, t] as columns
        gdf(dataframe) - gold pair dataframe with [s, t] as columns
    '''
    sdf = create_europarl_df(part, s, t, s)
    tdf = create_europarl_df(part, s, t, t)
    gdf = pd.DataFrame({s: sdf["id"],
                         t: tdf["id"]})
    return [sdf, tdf, gdf]

### opus ###
def create_opus_df(part, s, t, lang):
    '''
    create europarl dataframe for specific sub_corpus, and language
    input :
        part(str) - sub corpus
        s(str), t(str) - source and target language to be reformatted, respectively
        lang(str) - specific language to be a column name
    output: df(dataframe) - with [id, lang] as columns
    '''
    try:
        data = read_raw(f"data/raw/opus/{part}/{s}-{t}.{lang}")
    except FileNotFoundError:
        data = read_raw(f"data/raw/opus/{part}/{t}-{s}.{lang}")
    sid = create_id(lang, len(data))
    df = pd.DataFrame({"id": sid,
                        lang: data})
    return df

def reformat_opus(part, s, t):
    '''
    create opus dataframe for specific sub_corpus, and language
    input :
        part(str) - sub corpus
        s(str), t(str) - source and target language to be reformatted, respectively
    output: 
        sdf(dataframe) - reformatted source dataframe with [id, s] as columns
        tdf(dataframe) - reformatted target dataframe with [id, t] as columns
        gdf(dataframe) - gold pair dataframe with [s, t] as columns
    '''
    sdf = create_opus_df(part, s, t, s)
    tdf = create_opus_df(part, s, t, t)
    gdf = pd.DataFrame({s: sdf["id"],
                         t: tdf["id"]})
    return [sdf, tdf, gdf]