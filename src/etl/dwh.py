from pathlib import Path
from datetime import datetime
import duckdb
from src.paths import DATA_DIR, PROCESSED_DATA_DIR, FILE_PATTERN
from src.logger import get_logger


import polars as pl

logger = get_logger(__name__)

# TODO | 2024-12-02 | Save this into MotherDuck


DATABASE_URL = DATA_DIR / "dwh.duckdb"

####### DDL

def generate_connection() -> duckdb.DuckDBPyConnection:
    """Returns a connection to the DuckDB database.

    Returns:
        duckdb.DuckDBPyConnection: _description_
    """
    return duckdb.connect(database=str(DATABASE_URL))

def run_database_operation(operation:str, *args, **kwargs):
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
        connection.execute("DROP TABLE IF EXISTS dwh.main.pickup_hourly;")
        connection.execute(
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
    
    # pylint: disable=consider-using-f-string
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
    # pylint: enable=consider-using-f-string
    
    with connection:
        connection.execute(statement)
    logger.info("Upserted %s into dwh.main.pickup_hourly", file)
    
    
def fetch_pickup_data(connection: duckdb.DuckDBPyConnection, from_date: datetime, to_date: datetime, pickup_locations: list[int] = []) -> pl.DataFrame:
    """
    Fetches pickup data from the data warehouse for a given date range and optional list of pickup locations.

    This function queries the `dwh.main.pickup_hourly` table to retrieve all records between `from_date` and `to_date` for
    the pickup locations specified in `pickup_locations`. 
    
    If `pickup_locations` is None, no location filter is applied, hence the IF statement in the query.
    This approach will require to pass any additional filter explicitly in the query and have the query to handle
    the case when no filter is passed. I consider this the best practice to split the logic between the query and the
    function itself. If needed, the query can be tested outside.

    Parameters:
    - from_date (datetime): The start date and time for the query range.
    - to_date (datetime): The end date and time for the query range.
    - pickup_locations (list[int] | None): Optional. A list of integers representing pickup location IDs to filter the query. If None, no location filter is applied.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing the query results.
    """
    
    if isinstance(pickup_locations, int):
        pickup_locations = [pickup_locations]
            
    if not isinstance(pickup_locations, list):
        raise ValueError("pickup_locations should be a list of integers")
    
    with connection:
        query = """
        SELECT 
            CAST(pickup_datetime_hour AS DATE) as pickup_datetime_hour
            , pickup_location_id 
            , sum(num_pickup) AS num_pickup
        FROM 
            dwh.main.pickup_hourly
        WHERE 
            pickup_datetime_hour >= '{from_date}' 
            AND pickup_datetime_hour < '{to_date}'
            AND IF(LENGTH({pickup_locations}) > 0, list_contains({pickup_locations}, pickup_location_id), TRUE)
        GROUP BY 1,2
        ORDER BY 1,2
            
        """.format(from_date=from_date, to_date=to_date, pickup_locations=pickup_locations) 
        
        df = connection.sql(query).pl()  
    return df



if __name__ == "__main__":
    with generate_connection() as db:
        create_pickup_table(db)
    print("Tables created successfully!")

