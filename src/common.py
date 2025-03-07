from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


### FOLDERS 

PARENT_DIR = Path(__file__).parent.resolve().parent

DATA_DIR = PARENT_DIR / "data"
MODEL_DIR = PARENT_DIR / "models"


import logging
import sys

def get_logger(name) -> logging.Logger:
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

# # Create directories if they don't exist
# for directory in [PARENT_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRANSFORMED_DATA_DIR, MODEL_DIR]:
#     if not directory.exists():
#         directory.mkdir()
#         print(f"Created directory: {directory}")