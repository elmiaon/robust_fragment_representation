##################################################
### import                                     ###
##################################################
# basic lib
from ast import literal_eval
from emd import emd
import numpy as np
import os
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(use_memory_fs=False)
from scipy.spatial.distance import cdist
import sys
# logging lib
import logging
import src.log as log
# time lib
from time import time
# custom lib
import src.utils as utils

##################################################
### define global                              ###
##################################################
t_vec = None # store target vector for many source to look at
t_label = None # store target label for many source

##################################################
### check input and output                     ###
##################################################
def check_output(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str, represent_method:list, retrieve_method:list):
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
    retrieve_method: list. [retriene method name, k]

    returns
    -------
    skip: bool. skip boolean to skip the tokenized process
    output_dir: list. output dict for save the tokenized sentences
    '''
    output_corpus_dir = f"data/retrieved/{corpus}/{sub_corpus}/{''.join(retrieve_method)}" # define output dir
    utils.make_dir(f"{output_corpus_dir}") # create output dir
    output_dir = f"{output_corpus_dir}/{tokenize_method}.{''.join(represent_method)}.{s}-{t}.csv"

    # check output to skip
    if os.path.isfile(output_dir):
        return True, output_dir
    else:
        return False, output_dir

def check_input(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str, represent_method:list):
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
    input_corpus_dir = f"data/represented/{corpus}/{sub_corpus}/{''.join(represent_method)}" # define input dir

    input_dir_fwd = {
        s: f"{input_corpus_dir}/{tokenize_method}.{s}-{t}.{s}.csv",
        t: f"{input_corpus_dir}/{tokenize_method}.{s}-{t}.{t}.csv"
    }
    
    input_dir_bwd = {
        s: f"{input_corpus_dir}/{tokenize_method}.{t}-{s}.{s}.csv",
        t: f"{input_corpus_dir}/{tokenize_method}.{t}-{s}.{t}.csv"
    }

    if os.path.isfile(input_dir_fwd[s]) and os.path.isfile(input_dir_fwd[t]):
        return input_dir_fwd
    elif os.path.isfile(input_dir_bwd[s]) and os.path.isfile(input_dir_bwd[t]):
        return input_dir_bwd
    else: # error if there is no represented file
        raise FileExistsError(f"There is no represented {corpus}-{sub_corpus}")

##################################################
### Interface                                  ###
##################################################
def retrieve(tokenize_method:str, represent_method:list, retrieve_method:list, corpus:str, sub_corpus:str, s:str, t:str, chunksize:int=1000):
    '''
    retrieve similar fragments from given similarity function

    parameters
    ----------
    tokenize_method: str. method to tokenize the reformatted corpus
    represent_method: list. list of [method, setting_code] to represent the tokenized corpus
    retrieve_method: list. list of [similarity_function, top_k] to retrieve k-NN similar target fragments
    corpus: str. corpus name
    sub_corpus: str. sub corpus name
    s: str. source language
    t: str. target language

    returns
    -------
    None

    * Note: there is not return but the function save result in data/retrieved/ directory
    '''

    global t_vec, t_label
    logger = log.get_logger(__name__)

    skip, output_dir = check_output(corpus, sub_corpus, s, t, tokenize_method, represent_method, retrieve_method)
    
    if not skip:

        input_dir = check_input(corpus, sub_corpus, s, t, tokenize_method, represent_method)

        method, k = retrieve_method # unpack retrieve_method

        k = int(k) # convert k to int

        tdf = pd.read_csv(input_dir[t], sep='\t', converters={'vec': utils.byte_decode}) # load tdf
        t_vec = np.vstack(tdf['vec'].values) # create target vector
        t_label = tdf[['id', 'fid']].parallel_apply(lambda x: f"{x[0]}|{x[1]}", axis=1).to_numpy() # create target label
        del tdf

        # select retrive method
        if method == 'cosine':
            calculate = cosine
        else:
            raise ValueError(f"invalid calculate method")
        
        retrieved_list = []
        for idx_chunk, chunk in enumerate(pd.read_csv(input_dir[s], sep='\t', chunksize=chunksize, converters={'vec':utils.byte_decode})):
            chunk['candidates'], chunk['distance'] = zip(*chunk.pop('vec').parallel_apply(calculate, args=(k,))) # retrieve for chunk
            retrieved_list.append(chunk) # append the chunk for cancat
            print(f"finish {corpus}-{sub_corpus}.{s}-{t} part {idx_chunk+1} retrieval") # print to tell the status of each chunk
        
        df = pd.concat(retrieved_list, ignore_index=True) # concat all chunk
        df['distance'] = df['distance'].parallel_apply(utils.byte_encode)
        df.to_csv(output_dir, sep='\t', index=False) # save the retrieved

        logger.info(f"finish {corpus}-{sub_corpus}.{s}-{t} retrieval")
        logger.info(f"sample:\n{df}") # show some samples

    else:
        logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t} retrieval")

##################################################
### distance function                          ###
##################################################

# def approx_emd(vec, n_min_distance):
#     global t_vec, t_label
#     distance = cdist([vec], t_vec, 'euclidean')
#     distance = distance.reshape(-1)
#     ind = np.argsort(distance)[:n_min_distance]
#     distance = distance[ind]
#     distance = kernel_dist(distance)
#     return t_label[ind], distance

# def approx_emd_square(vec, n_min_distance):
#     global t_vec, t_label
#     distance = cdist([vec], t_vec, 'euclidean')
#     distance = distance.reshape(-1)
#     ind = np.argsort(distance)[:n_min_distance]
#     distance = distance[ind]
#     distance = np.power(distance, 2)
#     distance = kernel_dist(distance)
#     return t_label[ind], distance

# def exact_emd(vec, n_min_distance):
#     global t_vec, t_label
#     tic = time()
#     distance = np.array(  [wmd(vec, t) for t in t_vec]  )
#     # distance = distance.reshape(-1)
#     ind = np.argsort(distance)[:n_min_distance]
#     distance = distance[ind]
#     distance = kernel_dist(distance)
#     toc = time()
#     print(f"finish calculate in {toc-tic:.2f} sencond(s)")
#     return t_label[ind], distance

# def wmd(a, b):
#     p_a, w_a = a
#     p_b, w_b = b
#     D = cdist(p_a, p_b, 'euclidean').tolist()
#     return (emd(w_a.tolist(), w_b.tolist(), D))

# def kernel_dist(dist):
#     return(  np.exp( -(np.power(dist,2))/2 ) / np.sqrt(2*np.pi)  )

def cosine(vec, k:int):
    '''
    retrieve k-NN of given source vectors from global target_vector

    parameters
    ----------
    vec: numpy array. [chunksize, embedding dim]. source vector array 
    k: int. k for k-NN

    returns
    -------
    labels: numpy array. [chunksize, k].
        sorted k-NN target label for source vector each element composed of 
        "fragment_id|sentence_id" of target fragment
    distance: numpy array. [chunksize, k]. distance between each source and k-NN target fragmemt
    '''
    global t_vec, t_label
    distance = np.inner(vec, t_vec)
    distance = 1-distance
    ind = np.argsort(distance)[:k]
    distance = distance[ind]
    return t_label[ind], distance

# def arccos(vec, n_min_distance):
#     global t_vec, t_label
#     tic = time()
#     sim = np.inner(vec, t_vec)
#     distance = 1-np.arccos(sim)/np.pi
#     ind = np.argsort(distance)[:n_min_distance]
#     distance = distance[ind]
#     distance = kernel_dist(distance)
#     toc = time()
#     print(f"finish calculate in {toc-tic:.2f} sencond(s)")
#     return t_label[ind], distance





