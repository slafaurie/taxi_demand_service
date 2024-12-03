from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()

PARENT_DIR = Path(__file__).parent.resolve().parent
#### URL FILE PATTERN
FILE_PATTERN = "yellow_tripdata_{year}-{month:02d}.parquet"
BASE_URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{FILE_PATTERN}"

#### DATABASE

db_mode = os.getenv('DB_MODE', 'LOCAL')

if db_mode == 'CLOUD':
    db_token = os.getenv('MOTHERDUCK_TOKEN')
    if db_token is None:
        raise Exception("Token is not found")
    DATABASE_URL = "md:my_db"
else:
    DATABASE_URL = PARENT_DIR / "data" / "dwh.duckdb"

    
DATABASE_NAME = 'nyc_trips'