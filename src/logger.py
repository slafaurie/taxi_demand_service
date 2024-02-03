import logging

def get_logger() -> logging.Logger:
    logger = logging.getLogger("DataFlow")
    logger.setLevel(logging.INFO)
    return logger