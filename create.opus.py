import src.create as create
import src.log as log

if __name__ == '__main__':
    logger = log.init_logger(__file__, __name__, "DEBUG")
    create.create_dataset(__file__)