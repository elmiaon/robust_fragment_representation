##################################################
### import                                     ###
##################################################
# basic lib
from time import time
# logging lib
import logging
import src.log as log
# custom lib
import src.utils as utils
from src.tokenize import tokenize
import src.represent as represent
import src.retrieve as retrieve
import src.tune as tune
import src.analysis as analysis
import src.get_test_score as get_test_score
import src.cal_score as cal_score

def RFR(method_params, corpus):
    '''
    tokenize sentences in the corpus
    input : args composed of
        SETTING_CODE(int) - setting_code to get the experiment parameter
        TUNE_CORPUS(str), TUNE_SUB_CORPUS(str) - corpus and sub corpus as tuning set
        TEST_CORPUS(str), TEST_SUB_CORPUS(str) - corpus and sub corpus as testing set
        S(str), T(str) - source and target language, respectively
    output: result for each steps(csv) - the result from each step saved in data/[step_name]/ directory
    '''
    logger = log.get_logger(__name__)
    TOKENIZE_METHOD, REPRESENT_METHOD, RETRIEVE_METHOD, AGGREGATE_METHOD = method_params
    TUNE_CORPUS, TUNE_SUB_CORPUS, TEST_CORPUS, TEST_SUB_CORPUS, S, T = corpus
    n_steps = 8 # total number of steps to track the progress

    logger = log.get_logger(__name__) # get logger instance

    world_tic = time() # start time of the experiment
    step = 1 # the current step

    # descirption of the experiment
    setting = f'''
    {'='*50}
    TRAIN CORPUS: {TUNE_CORPUS} - {TUNE_SUB_CORPUS}
    TEST_CORPUS: {TEST_CORPUS} - {TEST_SUB_CORPUS}
    S: {S}
    T: {T}
    TOKENIZE_METHOD: {TOKENIZE_METHOD}
    REPRESENT_METHOD: {REPRESENT_METHOD}
    RETRIEVE_METHOD: {RETRIEVE_METHOD}
    AGGREGATE_METHOD: {AGGREGATE_METHOD}
    {'='*50} 
    '''
    logger.info(f"{setting}")

    #####
    # 1.) tokenize the training dataset
    #####
    tic = time()
    tokenize(TOKENIZE_METHOD, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T)
    toc = time()
    logger.info(f"step {step}/{n_steps} - preprocess training data in {toc-tic:.2f} second(s)")
    step+=1

    # #####
    # # 2.) Represent the training dataset
    # #####
    # tic = time()
    # represent.CLSR(SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1

    # #####
    # # 3.) Retrieve the training dataset
    # #####
    # tic = time()
    # retrieve.CLSR(SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1


    # #####
    # # 4.) Parameters tuning
    # #####
    # tic = time()
    # params = tune.tune_aggregator(SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - tuning parameter in {toc-tic:.2f} second(s)")
    # step+=1
    
    # #####
    # # 4.5.) Looking around
    # #####
    # tic = time()
    # analysis.vary_around_params(SETTING_CODE, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T, params)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - tuning parameter in {toc-tic:.2f} second(s)")
    # # step+=1

    # #####
    # # 5.) tokenize the testing dataset
    # #####
    # tic = time()
    # tokenize.CLSR(SETTING_CODE, TEST_CORPUS, TEST_SUB_CORPUS, S, T)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - preprocess training data in {toc-tic:.2f} second(s)")
    # step+=1

    # #####
    # # 6.) Represent the testing dataset
    # #####
    # tic = time()
    # represent.CLSR(SETTING_CODE, TEST_CORPUS, TEST_SUB_CORPUS, S, T)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1

    # #####
    # # 7.) Retrieve the training dataset
    # #####
    # tic = time()
    # retrieve.CLSR(SETTING_CODE, TEST_CORPUS, TEST_SUB_CORPUS, S, T)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1

    # #####
    # # 8.) Get test score
    # #####
    # tic = time()
    # get_test_score.get_test_score(SETTING_CODE, TEST_CORPUS, TEST_SUB_CORPUS, TUNE_CORPUS, TUNE_SUB_CORPUS, S, T, params)
    # toc = time()
    # logger.info(f"step {step}/{n_steps} - retrieve candidates in {toc-tic:.2f} second(s)")
    # step+=1