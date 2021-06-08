import src.run as run
import src.log as log

if __name__ == '__main__':
    logger = log.init_logger(__file__, __name__, "DEBUG")
    run.CLSR(__file__)