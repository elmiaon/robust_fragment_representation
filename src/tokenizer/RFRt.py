##################################################
### import                                     ###
##################################################
# basic lib
import numpy as np
import os
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(use_memory_fs=False)
import string
# logging lib
import logging
import src.log as log
# nlp lib
import re
from pythainlp.tokenize import word_tokenize

##################################################
### define global                              ###
##################################################
punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.«»" # special punctuation especially in Japanese
punc2 = string.punctuation # common punctuation
double_space = " ​ " # double space in Thai

##################################################
### RFRt                                       ###
##################################################
def RFRt(input_dir:dict, output_dir:dict, chunksize:int=10000):
    '''
    tokenize sentences in the corpus using method in RFR paper
    
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

    logger = log.get_logger(__name__)
    # tokenize {s} and {t} corpus
    for lang in input_dir:
        in_dir = input_dir[lang] # define input dir for the specific language
        out_dir = output_dir[lang] # define output dir for the specific language

        if not os.path.isfile(out_dir):
            tokenized_list = [] # define to store tokenized chunk

            for idx_chunk, chunk in enumerate(pd.read_csv(in_dir, sep='\t', chunksize=chunksize)): # read csv chunk-by-chunk
                chunk[lang] = chunk[lang].parallel_apply(RFR_tokenize_sentence, args=[lang,])
                tokenized_list.append(chunk) # tokenize each chunk
                print(f"finish {lang} tokenization part {idx_chunk+1}") # print to tell the status of each chunk
                
            df = pd.concat(tokenized_list, ignore_index=True) # concatenate all chunk
            df.to_csv(out_dir, sep='\t', index=False) # save the tokenized
        
            logger.info(f"finish {lang} tokenzation")
            logger.info(f"sample:\n{df}") # show some sample
        
        else:
            logger.info(f"skip {lang} tokenzation") # the tokenize step for specific language is skipped

def RFR_tokenize_sentence(sentence:str, lang:str):
    '''
    RFR tokenize for input sentence
    input : 
        sentence(str) - sentence to be tokenized
        lang(str) - language indicator
    output: tokenized sentences(list) - tokenized dataframe with dropped Nan
    '''
    global punc, punc2, double_space # define global varible in the function
    sentence = sentence.strip('\n') # strip new line
    sentence = sentence.lower() # lower the characters in the setnence
    # remove punctuation
    sentence = re.sub(r"[%s%s]+"%(punc, punc2), '', sentence) # remove punc, punc2
    sentence = sentence.strip(' ') # remove puncs probably create some space at the start and end of sentence
    sentence = re.sub(r"[%s]+"%(double_space), ' ', sentence) # replace double space with single space

    if lang == 'th': # use special tokenizer for Thai
        sentence= word_tokenize(sentence, engine='newmm')
        sentence = [i for i in sentence if i != ' ']
        if len(sentence)==0:
            return None
    else: # for other languages tokenize using whitespace
        sentence = sentence.split(' ')
    
    if len(sentence) == 1 and sentence[0]=='': # remove the empty tokenized sentence
        return None
    else: # return the normal tokenized sentence
        return sentence