from pathlib import Path
from datetime import datetime
import os
import duckdb
import polars as pl
from abc import ABC, abstractmethod

from src.paths import PROCESSED_DATA_DIR, PARENT_DIR
from src.logger import get_logger
from src.etl.constants import DATABASE_NAME, FILE_PATTERN

logger = get_logger(__name__)



class NYCRepository(ABC):

    
    @abstractmethod
    def upsert_data(self, data: pl.DataFrame):
        pass
    
    @abstractmethod
    def fetch_data(self, from_data: datetime, to_date: datetime, pickup_ids: list[int]) -> pl.DataFrame:
        pass
        



####### DDL

def generate_connection():
    # should I pass the mode as default argument?
    """Returns a connection to the DuckDB database.

    Returns:
        duckdb.DuckDBPyConnection: _description_
    """
    
    # should it default to local mode if db_mode is passed?
    db_mode = os.getenv('DB_MODE', None)
    if not db_mode:
        logger.info('DB_MODE env variable not found')
        db_mode='LOCAL'
    
    logger.info("Running DB in mode %s", db_mode)

    if db_mode == 'CLOUD':
        db_token = os.getenv('MOTHERDUCK_TOKEN')
        if db_token is None:
            logger.error("DB Token is not found")
            raise Exception("Token is not found")
        # TODO | 2025-02-01 | Avoid exposing DB URL
        # this should be loaded from the ENV variable?
        database_url = "md:my_db"
    else:
        database_url = PARENT_DIR / "data" / f"{DATABASE_NAME}.duckdb"
    # TODO | 2024-12-03 | Yield connection
    # I'm creating a new connection everytime I call this, probably there's a better way
    return duckdb.connect(database=str(database_url))

def run_database_operation(operation:str, *args, **kwargs):
    """ 
    Interface to call different operations
    """
    if operation not in ["upsert_pickup_data", "fetch_pickup_data"]:
        raise ValueError("Invalid operation name")
    
    with generate_connection() as db:
        if operation == "upsert_pickup_data":
            upsert_pickup_data(db, *args, **kwargs)
        elif operation == "fetch_pickup_data":
            df = fetch_pickup_data(db, *args, **kwargs)
            return df
    
def create_pickup_table(connection: duckdb.DuckDBPyConnection):
    """
    Creates the pickup_hourly table in the data warehouse.

    This function drops the existing dwh.main.pickup_hourly table if it exists and then creates a new one.
    The new table includes columns for a unique key, the hour of the pickup, the location ID of the pickup,
    and the number of pickups that occurred during that hour.

    Parameters:
    - db (duckdb.DuckDBPyConnection): The database connection object.

    Returns:
    None
    """
    with connection:
        # TODO | 2024-12-03 | Reduce duplication
        # Try to find a better practice to avoid having to call again the env
        # as this is called too when we create the connection
        if os.getenv('DB_MODE', 'LOCAL') == 'CLOUD':
            connection.execute(f"CREATE OR REPLACE DATABASE {DATABASE_NAME};")
            
        connection.execute(f"DROP TABLE IF EXISTS {DATABASE_NAME}.main.pickup_hourly;")
        connection.execute(
            f"""
            CREATE TABLE {DATABASE_NAME}.main.pickup_hourly (
                key STRING PRIMARY KEY
                , pickup_datetime_hour TIMESTAMP
                , pickup_location_id SMALLINT
                , num_pickup SMALLINT
            );
            """
        )    
    logger.info("Created %s.main.pickup_hourly table", DATABASE_NAME)

def upsert_pickup_data(connection: duckdb.DuckDBPyConnection, year:int, month:int):
    """
    Upserts data from a processed file into the dwh.main.pickup_hourly table.

    This function reads data from a parquet file corresponding to the specified year and month, 
    creates a temporary staging table, and then upserts the data into the dwh.main.pickup_hourly table.
    If a record with the same key already exists, it updates the num_pickup field with the new value.

    Parameters:
    - db (duckdb.DuckDBPyConnection): The database connection object.
    - year (int): The year of the data file to be upserted.
    - month (int): The month of the data file to be upserted.

    Returns:
    None
    """
    
    file = str(PROCESSED_DATA_DIR / Path(FILE_PATTERN.format(year=year, month=month)))
    
    statement = f"""
        CREATE OR REPLACE TEMP TABLE stg_pickup_hourly AS
        SELECT * 
        FROM read_parquet('{file}');
        
        INSERT INTO {DATABASE_NAME}.main.pickup_hourly  
        SELECT * FROM stg_pickup_hourly
        ON CONFLICT(key)
        DO UPDATE SET num_pickup = EXCLUDED.num_pickup;
        
        DROP TABLE stg_pickup_hourly;
    """    
    with connection:
        connection.execute(statement)
    logger.info("Upserted %s into dwh.main.pickup_hourly", file)
    
    
def fetch_pickup_data(connection: duckdb.DuckDBPyConnection, from_date: datetime, to_date: datetime, pickup_locations: list[int] = None) -> pl.DataFrame:
    """
    Fetches pickup data from the data warehouse for a given date range and optional list of pickup locations.

    Parameters:
    - from_date (datetime): The start date and time for the query range.
    - to_date (datetime): The end date and time for the query range.
    - pickup_locations (list[int] | None): Optional. A list of integers representing pickup location IDs to filter the query. If None, no location filter is applied.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing the query results.
    """
    
    if pickup_locations is None:
        pickup_locations = []
    
    if isinstance(pickup_locations, int):
        pickup_locations = [pickup_locations]
            
    if not isinstance(pickup_locations, list):
        raise ValueError("pickup_locations should be a list of integers")
    
    with connection:
        query = f"""
        SELECT 
            CAST(pickup_datetime_hour AS DATETIME) AS pickup_datetime_hour
            , CAST(pickup_location_id AS FLOAT) AS pickup_location_id
            , CAST(sum(num_pickup) AS FLOAT) AS num_pickup
        FROM 
            {DATABASE_NAME}.main.pickup_hourly
        WHERE 
            pickup_datetime_hour >= '{from_date}' 
            AND pickup_datetime_hour < '{to_date}'
            AND IF(LENGTH({pickup_locations}) > 0, list_contains({pickup_locations}, pickup_location_id), TRUE)
        GROUP BY 1,2
        ORDER BY 1,2
        """
        
        df = connection.sql(query).pl()  
    return df

