##################################################
### import                                     ###
##################################################
# custom lib
import src.execute as execute
import src.log as log
import src.utils as utils

# define setting
description = 'ADD YOUR DESCRIPTION'
corpus_config_name = 'REPLACE_WITH_YOUR_DATASET_NAME'
pipeline_config_list = [['RFR', '0']]
lang_config_name = 'REPLACE_WITH_YOUR_LANGUAGE_CONFIG_NAME'

if __name__ == '__main__':
    logger = log.init_logger(__file__, __name__, "DEBUG")
    execute.basic(corpus_config_name, pipeline_config_list, lang_config_name)
    