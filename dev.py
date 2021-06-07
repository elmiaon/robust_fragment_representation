import src.load_corpus as load_corpus
import numpy as np
import src.log as log

def main():
    logger = log.get_logger(__name__)
    corpus = 'opus'
    sub_corpus = 'QED'
    src = 'th'
    tar = 'en'

    #load corpus
    if load_corpus.load_CLSR((corpus, sub_corpus, src, tar)):
        logger.info(f"now timed")
    else:
        logger.info(f"skipped")
        
    # tokenize
    
    # embed

    # retrieve

    # aggregate

    # answer

    # calculate score

    # analyse

    pass

if __name__ == '__main__':
    logger = log.init_logger(__file__, __name__, "DEBUG")
    main()