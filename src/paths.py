from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


### FOLDERS 

PARENT_DIR = Path(__file__).parent.resolve().parent

DATA_DIR = PARENT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"

MODEL_DIR = PARENT_DIR / "models"


# Create directories if they don't exist
for directory in [PARENT_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRANSFORMED_DATA_DIR, MODEL_DIR]:
    if not directory.exists():
        directory.mkdir()
        print(f"Created directory: {directory}")