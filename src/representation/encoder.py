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
model_dir = 'data/model'
utils.make_dir(model_dir)
os.environ['TFHUB_CACHE_DIR'] = model_dir
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
        return preprocessor, encoder_model, encoder
    else:
        raise ValueError(f"invalid base_encoder_name: {base_encoder_name}")

    return encoder_model, encoder
    
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