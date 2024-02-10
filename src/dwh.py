from pathlib import Path
import duckdb
from src.paths import DATA_DIR, PROCESSED_DATA_DIR, FILE_PATTERN
from src.logger import get_logger


logger = get_logger(__name__)

DATABASE_URL = DATA_DIR / "dwh.duckdb"

def create_pickup_table(db: duckdb.DuckDBPyConnection):
    db.execute("DROP TABLE IF EXISTS dwh.main.pickup_hourly;")
    db.execute(
        """
        CREATE TABLE dwh.main.pickup_hourly (
            key STRING PRIMARY KEY
            , pickup_datetime_hour TIMESTAMP
            , pickup_location_id SMALLINT
            , num_pickup SMALLINT
        );
        """
    )
    
    logger.info("Created dwh.main.pickup_hourly table")

def generate_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=str(DATABASE_URL))

def upsert_file_into_db(db: duckdb.DuckDBPyConnection, year:int, month:int):
    
    file = str(PROCESSED_DATA_DIR / Path(FILE_PATTERN.format(year=year, month=month)))
    
    statement = """
        CREATE OR REPLACE TEMP TABLE stg_pickup_hourly AS
        SELECT * 
        FROM read_parquet('{file}');
        
        INSERT INTO dwh.main.pickup_hourly  
        SELECT * FROM stg_pickup_hourly
        ON CONFLICT(key)
        DO UPDATE SET num_pickup = EXCLUDED.num_pickup;
        
        DROP TABLE stg_pickup_hourly;
    """.format(file=file)
    
    db.execute(statement)
    logger.info(f"Upserted {file} into dwh.main.pickup_hourly")
    

    

if __name__ == "__main__":
    db = duckdb.connect(database=str(DATABASE_URL))
    create_pickup_table(db)
    print("Tables created successfully!")

