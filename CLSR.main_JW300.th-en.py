import src.experiment as experiment
import src.log as log

if __name__ == '__main__':
    logger = log.init_logger(__file__, __name__, "DEBUG")
    experiment.experiment(__file__)