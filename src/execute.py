##################################################
### import                                     ###
##################################################
import json
# custom lib
from src.RFR import RFR

##################################################
### execute                                    ###
##################################################
def basic(corpus_config_name:str, pipeline_config_list:list, lang_config_name:str):
    '''
    execute the experiment(s) from given corpus, pipeline, and language pair config

    parameters
    ----------
    corpus_config_name: str. key of a corpus config file
    pipeline_config_list: list. list of the [pipeline name, pipeline setting] to execute
    lang_config_name: str. key of a language config file

    returns
    -------
    None

    * Note: there is no return but this function save intermediate results in csv file format depended on pipeline.
    '''

    execute_list = get_execute_list(corpus_config_name, pipeline_config_list, lang_config_name)
    
    for method, method_params, corpus in execute_list:
        if method == 'RFR':
            execute = RFR
        else:
            raise (f"Invalid execution method")
        
        execute(method_params, corpus)


##################################################
### create execute list                        ###
##################################################

def get_execute_list(corpus_config_name:str, pipeline_config_list:list, lang_config_name:str):
    '''
    create execute list from given corpus, pipeline, and language pair config.

    parameters
    ----------
    corpus_config_name: str. key of a corpus config file
    pipeline_config_list: list. list of [pipeline name, pipeline setting] to execute
    lang_config_name: str. key of a language config file

    returns
    -------
    execute_list: list. list of the [pipeline_method, [method_params], [corpus]] to execute
    '''

    corpus_list = get_corpus_list(corpus_config_name) # get corpus list
    pipeline_list = get_pipeline_list(pipeline_config_list) # get pipeline list
    lang_list = get_lang_list(lang_config_name) # get language pair list

    execute_list = [] # initiate empty list to store execution element
    for pipeline_method, method_params in pipeline_list:
        for corpus in corpus_list:
            for lang in lang_list:
                execute_list.append((pipeline_method, method_params, [*corpus, *lang])) # add execution

    return execute_list

##################################################
### get list from config                       ###
##################################################

def get_corpus_list(corpus_config_name:str):
    '''
    get corpus list from corpus config name

    parameters
    ----------
    corpus_config_name: str. key of a corpus config file

    returns
    -------
    corpus_list: list. list of corpus to execute
    '''
    with open('config/corpus.json') as f:
        corpus_dict = json.load(f)

    return corpus_dict[corpus_config_name]

def get_pipeline_list(pipeline_config_list:list):
    '''
    get pipeline list from pipeline config list

    parameters
    ----------
    pipeline_config_list: list. list of [pipeline name, pipeline setting] to execute

    returns
    -------
    pipeline_list: list. list of [pipeline name, parameters] to execute
    '''
    with open('config/pipeline.json') as f:
        pipeline_dict = json.load(f)
    
    pipeline_list = []

    for pipeline_name, pipeline_config in pipeline_config_list:

        pipeline_list.append([pipeline_name, pipeline_dict[pipeline_name][pipeline_config]["pipeline"]])
    return pipeline_list

def get_lang_list(lang_config_name:str):
    '''
    get language pair list from language config name

    parameters
    ----------
    lang_config_name: str. key of a language config file

    returns
    -------
    lang_list: list. list of language pairs [s, t] to execute
    '''
    with open('config/language.json') as f:
        lang_dict = json.load(f)

    return lang_dict[lang_config_name]
