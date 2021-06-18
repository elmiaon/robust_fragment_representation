##################################################
### import                                     ###
##################################################
# basic lib
from ast import literal_eval
import json
import numpy as np
import os
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(use_memory_fs=False)
# logging lib
import logging
import src.log as log
# time lib
from time import time
# nlp lib
import nltk
nltk.download('stopwords')
from nltk import ngrams
# custom lib
import src.utils as utils
# model stuff lib
from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from laserembeddings import Laser

##################################################
### define global                              ###
##################################################
current_encoder_name = None
encoder = None
encoder_model = None
preprocessor = None

model_dir = 'data/model'
utils.make_dir(model_dir)
os.environ['TFHUB_CACHE_DIR'] = model_dir

##################################################
### check input and output                     ###
##################################################
def check_output(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str, represent_method:list):
    '''
    checking output for skip
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    tokenize_method: string.
    represent_method: list. [represent method name, setting code]
    s: source language
    t: target language

    returns
    -------
    skip: bool. skip boolean to skip the tokenized process
    output_dir: list. output dict for save the tokenized sentences
    '''
    output_corpus_dir = f"data/represented/{corpus}/{sub_corpus}/{'s'.join(represent_method)}" # define output dir
    utils.make_dir(f"{output_corpus_dir}") # create output dir
    output_dir_fwd = { # output from {s}->{t}
        s: f"{output_corpus_dir}/{tokenize_method}.{s}-{t}.{s}.csv",
        t: f"{output_corpus_dir}/{tokenize_method}.{s}-{t}.{t}.csv"
    }

    output_dir_bwd = { # output from {t}->{s}
        s: f"{output_corpus_dir}/{tokenize_method}.{t}-{s}.{s}.csv",
        t: f"{output_corpus_dir}/{tokenize_method}.{t}-{s}.{t}.csv"
    }

    # check output to skip
    if os.path.isfile(output_dir_fwd[s]) and os.path.isfile(output_dir_fwd[t]):
        return True, output_dir_fwd
    elif os.path.isfile(output_dir_bwd[s]) and os.path.isfile(output_dir_bwd[t]):
        return True, output_dir_bwd
    else:
        return False, output_dir_fwd

def check_input(corpus:str, sub_corpus:str, s:str, t:str, tokenize_method:str):
    '''
    checking input
    parameters
    ----------
    corpus: string.
    sub_corpus: string.
    s: source language
    t: target language
    tokenize_method: string.

    returns
    -------
    input_dir: list. output dict for save the tokenized sentences
    '''
    # check the tokenized dataset to be represented
    input_corpus_dir = f"data/tokenized/{corpus}/{sub_corpus}/{tokenize_method}" # define input dir

    input_dir_fwd = {
        s: f"{input_corpus_dir}/{s}-{t}.{s}.csv",
        t: f"{input_corpus_dir}/{s}-{t}.{t}.csv"
    }
    
    input_dir_bwd = {
        s: f"{input_corpus_dir}/{t}-{s}.{s}.csv",
        t: f"{input_corpus_dir}/{t}-{s}.{t}.csv"
    }

    if os.path.isfile(input_dir_fwd[s]) and os.path.isfile(input_dir_fwd[t]):
        return input_dir_fwd
    elif os.path.isfile(input_dir_bwd[s]) and os.path.isfile(input_dir_bwd[t]):
        return input_dir_bwd
    else: # error if there is no reformatted file
        raise FileExistsError(f"There is no reformatted {corpus}-{sub_corpus}")

##################################################
### CLSR representation                        ###
##################################################

def CLSR(setting_code:int, corpus:str, sub_corpus:str, s:str, t:str, chunksize:int=10000):
    '''
    create representation for CLSR task
    input : 
        setting_code(int) - setting_code to get the experiment parameter
        corpus(str), sub_corpus(str) - corpus and sub corpus to be tokenzied
        s(str), t(str) - source and target language to be represented, respectively
    output: represented dataset files(csv) - saved in data/represented/ directory
    '''
    logger = log.get_logger(__name__)
    
    _, tokenize_method, represent_method, _, _ = utils.get_experiment_setting(setting_code)

    skip, output_dir = check_output(corpus, sub_corpus, s, t, tokenize_method, represent_method)
    
    if not skip:

        input_dir = check_input(corpus, sub_corpus, s, t, tokenize_method)

        method, represent_setting_code = represent_method # unpack represent_method

        base_encoder_name, args = get_representation_setting(method, represent_setting_code) # get base encoder and args for represent method

        set_encoder(base_encoder_name) # set encoder to global

        for lang in [s,t]:
            in_dir = input_dir[lang]
            out_dir = output_dir[lang]

            if not os.path.isfile(out_dir):
                
                preprocessed_list = []

                for idx_chunk, chunk in enumerate(pd.read_csv(in_dir, sep='\t', chunksize=chunksize, converters={f"{lang}": literal_eval})): # read csv chunk-by-chunk
                    preprocessed_list.append(CLSR_represent(chunk, method, args)) # preprocess each chunk
                    print(f"finish {corpus}-{sub_corpus}.{s}-{t}.{lang} part {idx_chunk+1} representation") # print to tell the status of each chunk

                df = pd.concat(preprocessed_list, ignore_index=True) # concat all chunk
                df.to_csv(out_dir, sep='\t', index=False) # save the represented

                logger.info(f"finish {corpus}-{sub_corpus}.{s}-{t}.{lang} tokenization")
                logger.info(f"sample:\n{df}") # show some samples
            
            else:
                logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t}.{lang} representation") # the represent step for specific language is skipped
            
    else:
        logger.info(f"skip {corpus}-{sub_corpus}.{s}-{t} representation") # the represent step is skipped

##################################################
### CLSR_represent                             ###
##################################################
### CLSR ###
def CLSR_represent(df, represent_method:str, args:tuple):
    '''
    CLSR represent for input dataframe

    parameters
    ----------
    df: dataframe. dataframe to be represented with [id, {language}] as columns
    represent_method: string

    returns
    -------
    represented_dataframe: dataframe. represented dataframe
    '''

    sid, lang = df.columns # get each column to indicate the input language

    # select representer
    if represent_method == 'RFR':
        represent = RFR_represent
    else:
        raise ValueError(f"invalid represent_method ({represent_method})")

    df = represent(df, *args)

    return df

def create_fragment(df):
    '''
    convert sentence to fragment dataframe

    parameters
    ----------
    df: dataframe. sentence dataframe to be covert with [id, {language}] as columns

    returns
    -------
    fdf: dataframe. fragment dataframe
    '''
    sid, lang = df.columns
    f_sid = np.concatenate(df.apply(lambda x: [x[sid]]*len(x[lang]),axis=1).values)
    fid = np.concatenate(df.apply(lambda x: np.arange(len(x[lang])), axis=1).values)
    content = np.concatenate(df[lang].values)
    fdf = pd.DataFrame({sid:f_sid,
                        'fid':fid,
                        lang: content})
    
    return fdf

##################################################
### represent method                           ###
##################################################
### RFR ###
def RFR_represent(df, n_grams:int, min_sentence_length:int):
    '''
    create fragment dataframe from tokenized dataframe

    parameters
    ----------
    df: dataframe. tokenized dataframe to be represented with [id, {language}] as columns
    n_grams: int. number of grams to create n_grams pharse
    min_sentence_length: int. minimum sentence length to allow n_grams pharse creation

    returns
    -------
    fdf: dataframe. represented fragment dataframe
    '''

    sid, lang = df.columns

    df = df.parallel_apply(RFR_sentence_represent, args=(n_grams, min_sentence_length), axis=1)
    df = df.dropna()
    sen_df = pd.DataFrame({'id': df['id']})
    sen_df['vec'] = encoder(df.pop('sen'), lang).tolist()
    fdf = create_fragment(df)
    fdf['vec'] = encoder(fdf[lang], lang).tolist()
    fdf = pd.merge(left=fdf, right=sen_df, left_on=['id'], right_on=['id'])
    vec = np.hstack(  [ np.vstack(fdf.pop('vec_x')), np.vstack(fdf.pop('vec_y')) ]  )
    vec = vec/np.linalg.norm(vec,axis=1).reshape(-1,1)
    fdf['vec'] = vec.tolist()
    fdf['vec'] = fdf['vec'].parallel_apply(utils.byte_encode)
    return fdf

def RFR_sentence_represent(row, n_grams, min_sentence_length):
    '''
    create n_grams pharse and join the sentence

    parameters
    ----------
    row: dataframe row. row to be create n_grams pharse and join the sentence
    n_grams: int. number of grams to create n_grams pharse
    min_sentence_length: int. minimum sentence length to allow n_grams pharse creation

    returns
    -------
    row: dataframe row. row with joined sentence and created fragment pharse
    '''
    sid, lang = row.index
    
    sentence = row[lang]
    sentence_len = len(sentence)
    
    # do n-grams on sentence that longer than minimum sentence length
    if sentence_len >= min_sentence_length:
        n_grams_phrases = list(ngrams(sentence, n_grams))
    else:
        n_grams_phrases = [sentence]

    # concat sentence
    if lang in ['th']:
        row[lang] = [''.join(i) for i in n_grams_phrases]
        row['sen'] = ''.join(sentence)
    else:
        row[lang] = [' '.join(i) for i in n_grams_phrases]
        row['sen'] = ' '.join(sentence) 
    
    return row



##################################################
### get representation setting                 ###
##################################################
def get_representation_setting(method:str, setting_code:int):
    '''
    get representation setting from setting_code

    parameters
    ----------
    method: string. method to identify representation parset
    setting_code: int. setting code to get the setting_dict

    returns
    -------
    base_encoder_name: string. name to get base encoder model
    args: list. list of parameters for represent method
    '''

    with open('config/run.json') as f: # load RFR-CLSR json setting
        setting_dict = json.load(f)
    setting_code = str(setting_code)

    if not setting_code in setting_dict['representation_setting'][method].keys(): # check setting code existance
        raise ValueError(f"invalid {method} representation setting_code")
    
    setting = setting_dict['representation_setting'][method][setting_code]
    if method == 'RFR':
        representation_parser = RFR_parser
    else:
        return ValueError("invalid representation method")

    return representation_parser(setting)

def RFR_parser(setting):
    
    keys = setting.keys()
    if 'description' in keys: # get description
        DESCRIPTION = setting['description']
    else:
        raise ValueError('missing experiment description')
    
    if 'base_encoder' in keys: # get tokenize method
        base_encoder = setting['base_encoder']
    else:
        base_encoder = 'USE'
    
    if 'n_grams' in keys:
        n_grams = int(setting['n_grams'])
    else:
        n_grams = 6
    
    if 'sentence_length' in keys:
        sentence_length = int(setting[sentence_length])
    else:
        sentence_length = 6
    
    if sentence_length < n_grams:
        raise ValueError(f"sentence_length must greater or equal to n_grams ({sentence_length} >= {n_grams})")
    
    return base_encoder, (n_grams, sentence_length)

##################################################
### set encoder                                ###
##################################################
def set_encoder(base_encoder_name:str):
    '''
    set encoder from encoder name
    input: base_encoder_name
    output: None
    '''
    global current_encoder_name, encoder, encoder_model
    if base_encoder_name != current_encoder_name:
        if base_encoder_name == 'USE':
            encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
            encoder = USE_encoder
        elif base_encoder_name == 'USE_Large':
            encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
            encoder = USE_Large_encoder
        elif  base_encoder_name == 'USE_QA':
            encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3")
            encoder = [USE_QA_encoder_question, USE_QA_encoder_answer]
        elif base_encoder_name == 'LASER':
            encoder_model = Laser()
            encoder = LASER_encoder
        elif base_encoder_name == 'LaBSE':
            preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
            encoder_model = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")
            encoder = LaBSE_encoder
        else:
            raise ValueError(f"invalid base_encoder_name: {base_encoder_name}")
        current_encoder_name = base_encoder_name
    
def USE_encoder(sentence_array, lang, batch_size=10):
    
    n_batch = get_n_batch(len(sentence_array), batch_size)

    sentence_vec = []
    for i in range(n_batch):
        sentence_batch = sentence_array[(i*batch_size):((i+1)*batch_size)]
        sentence_vec.append(encoder_model(sentence_batch).numpy())
    return np.vstack(sentence_vec)

def USE_Large_encoder(sentence_array, lang, batch_size=10):
    n_batch = get_n_batch(len(sentence_array), batch_size)

    sentence_vec = []
    for i in range(n_batch):
        sentence_batch = sentence_array[(i*batch_size):((i+1)*batch_size)]
        sentence_vec.append(encoder_model(sentence_batch).numpy())
    return np.vstack(sentence_vec)

def USE_QA_encoder_question(sentence_array, lang, batch_size=10):
    n_batch = get_n_batch(len(sentence_array), batch_size)

    sentence_vec = []
    for i in range(n_batch):
        sentence_batch = sentence_array[(i*batch_size):((i+1)*batch_size)]
        sentence_vec.append(encoder_model.signatures['question_encoder'](tf.constant(sentence_batch))['outputs'].numpy())
    return np.vstack(sentence_vec)

def USE_QA_encoder_answer(context_array, sentence_array, lang, batch_size=10):
    n_batch = get_n_batch(len(sentence_array), batch_size)

    sentence_vec = []
    for i in range(n_batch):
        sentence_batch = sentence_array[(i*batch_size):((i+1)*batch_size)]
        context_batch = context_array[(i*batch_size):((i+1)*batch_size)]
        sentence_vec.append(encoder_model.signatures['response_encoder'](input=tf.constant(sentence_batch), context=tf.constant(context_batch))['outputs'].numpy())
    return np.vstack(sentence_vec)

def LASER_encoder(sentence_array, lang, batch_size=10):
    n_batch = get_n_batch(len(sentence_array), batch_size)

    sentence_vec = []
    for i in range(n_batch):
        sentence_batch = sentence_array[(i*batch_size):((i+1)*batch_size)]
        sentence_vec.append(encoder_model.embed_sentences(sentence_batch, lang))
    return np.vstack(sentence_vec)

def LaBSE_encoder(sentence_array, lang, batch_size=10):
    n_batch = get_n_batch(len(sentence_array), batch_size)

    sentence_vec = []
    for i in range(n_batch):
        sentence_batch = sentence_array[(i*batch_size):((i+1)*batch_size)]
        sentence_vec.append(encoder_model(preprocessor(sentence_batch))['default'])
    return normalization(np.vstack(sentence_vec))

def get_n_batch(len_array, batch_size):
    if len_array%batch_size == 0:
        return len_array//batch_size
    else:
        return len_array//batch_size+1

def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds/norms