import logging
import sys

def get_logger(name) -> logging.Logger:
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger