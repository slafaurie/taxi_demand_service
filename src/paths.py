from pathlib import Path

#### URLS


FILE_PATTERN = "yellow_tripdata_{year}-{month:02d}.parquet"

BASE_URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{FILE_PATTERN}"


#### Base URLS


### FOLDERS 

PARENT_DIR = Path(__file__).parent.resolve().parent

DATA_DIR = PARENT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"



# Create directories if they don't exist
for directory in [PARENT_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    if not directory.exists():
        directory.mkdir()
        print(f"Created directory: {directory}")