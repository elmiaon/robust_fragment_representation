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
from src.representation.encoder import set_encoder
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


##################################################
### CLSR representation                        ###
##################################################
def RFRr(represent_setting_code:str,input_dir:dict, output_dir:dict, chunksize:int=10000):
    '''
    represent sentences in the corpus using method in RFR paper
    
    parameters
    ----------
    input_dir: dict. input directory with language as key
    output_dir: dict. output directory with language as key
    chunksize: int. default=10000. 

    returns
    -------
    None

    * Note: there is not return but the function save result in data/tokenized/ directory
    '''
    global current_encoder_name, encoder, encoder_model, prepreocessor

    logger = log.get_logger(__name__)

    base_encoder_name, n_grams, min_sentence_length = RFRr_parser(represent_setting_code) # get base encoder and args for represent method

    if base_encoder_name != current_encoder_name:
        if base_encoder_name == 'LaBSE':
            prepreocessor, encoder_model, encoder = set_encoder(base_encoder_name)
        else:
            encoder_model, encoder = set_encoder(base_encoder_name)
        current_encoder_name = base_encoder_name
    
    for lang in input_dir:
        in_dir = input_dir[lang]
        out_dir = output_dir[lang]

        if not os.path.isfile(out_dir):
            
            represented_list = []

            for idx_chunk, chunk in enumerate(pd.read_csv(in_dir, sep='\t', chunksize=chunksize, converters={f"{lang}": literal_eval})): # read csv chunk-by-chunk

                chunk = chunk.parallel_apply(RFR_sentence_represent, args=(n_grams, min_sentence_length), axis=1)
                chunk = chunk.dropna()
                sen_df = pd.DataFrame({'id': chunk['id']})
                sen_df['vec'] = encoder(chunk.pop('sen'), lang).tolist()
                fdf = create_fragment(chunk)
                fdf['vec'] = encoder(fdf[lang], lang).tolist()
                fdf = pd.merge(left=fdf, right=sen_df, left_on=['id'], right_on=['id'])
                vec = np.hstack(  [ np.vstack(fdf.pop('vec_x')), np.vstack(fdf.pop('vec_y')) ]  )
                vec = vec/np.linalg.norm(vec,axis=1).reshape(-1,1)
                fdf['vec'] = vec.tolist()
                fdf['vec'] = fdf['vec'].parallel_apply(utils.byte_encode)

                represented_list.append(fdf) # preprocess each chunk
                print(f"finish {lang} part {idx_chunk+1} representation") # print to tell the status of each chunk

            df = pd.concat(represented_list, ignore_index=True) # concat all chunk
            df.to_csv(out_dir, sep='\t', index=False) # save the represented

            logger.info(f"finish {lang} tokenization")
            logger.info(f"sample:\n{df}") # show some samples
        
        else:
            logger.info(f"skip {lang} representation") # the represent step for specific language is skipped

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

def RFRr_parser(setting_code:str):
    '''
    get RFRr parameters from setting_code
    
    parameters
    ----------
    setting_code: str. RFRr represent setting code

    returns
    -------
    base_encoder: str. base_encoder name
    args_tuple: tuple. tuple of (n_grams, sentence_length)
        n_grams: int. number of grams to create fragment
        sentence_length: int. minimum sentence length to create fragment
    '''

    with open('config/represent.json') as f: # load RFR-CLSR json setting
        setting_dict = json.load(f)

    if not setting_code in setting_dict['RFRr'].keys(): # check setting code existance
        raise ValueError(f"invalid RFRr representation setting_code")

    setting = setting_dict['RFRr'][setting_code]

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
    
    return base_encoder, n_grams, sentence_length