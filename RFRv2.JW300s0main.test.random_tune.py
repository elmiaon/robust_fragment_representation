##################################################
### import                                     ###
##################################################
# custom lib
import src.execute as execute
import src.log as log
import src.utils as utils

# define setting
description = 'test new version of RFR'
corpus_config_name = 'JW300s0main'
pipeline_config_list = [['RFR', '1']]
lang_config_name = 'fr'

if __name__ == '__main__':
    logger = log.init_logger(__file__, __name__, "DEBUG")
    execute.basic(corpus_config_name, pipeline_config_list, lang_config_name)
    