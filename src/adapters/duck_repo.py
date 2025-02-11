import os 
import duckdb 
import polars as pl
from datetime import datetime



from src.adapters.base import NYCTaxiRepository, DATABASE_NAME, SCHEMA
from src.common import DATA_DIR, get_logger



logger = get_logger(__name__)



class DuckDBRepository(NYCTaxiRepository):
    
    def __init__(self, db_mode: str = os.getenv('DB_MODE', 'LOCAL')):
        self.db_mode = db_mode 
        self.db_url = self._get_db_url()
        
    def _get_db_url(self) -> str:
        # should I pass the mode as default argument?
        """Returns a connection to the DuckDB database.
        We separate the db_url and connection methods because the
        db_url method can be called at instatiation already to resolve
        the DB URL.

        Returns:
            duckdb.DuckDBPyConnection: _description_
        """
        logger.info("Running DB in mode %s", self.db_mode)

        if self.db_mode == 'CLOUD':
            db_url = os.getenv('DB_URL')
            if db_url is None:
                logger.error("DB Token is not found")
                raise Exception("Token is not found")
            return db_url
        return str(DATA_DIR / f"{DATABASE_NAME}.duckdb")

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Returns a DB connection.
        We separate the db_url and connection because connection
        method is meant to be called as part of a context manager
        when doing a transaction.

        Returns:
            duckdb.DuckDBPyConnection: _description_
        """
        return duckdb.connect(database=self.db_url)
    
    def create_tables(self):
        """
        
        # TODO | 2025-02-09 | Relate this to the etl.models
        # there should be an explicit dependency to the schema object in etl.models
        # the models defines the data type and this functino should setup the database
        Creates the pickup_hourly table in the data warehouse.

        This function drops the existing dwh.main.pickup_hourly table if it exists and then creates a new one.
        The new table includes columns for a unique key, the hour of the pickup, the location ID of the pickup,
        and the number of pickups that occurred during that hour.

        Parameters:
        - db (duckdb.DuckDBPyConnection): The database connection object.

        Returns:
        None
        """
        self._pickup_table = f"{DATABASE_NAME}.{SCHEMA}.pickup_hourly" # noqa

        with self._get_connection() as conn:
            if self.db_mode == 'CLOUD':
                conn.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME};")
            
            conn.execute(
                f"""
                CREATE SCHEMA IF NOT EXISTS {DATABASE_NAME}.{SCHEMA};
                """
            )
            
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._pickup_table} (
                    key STRING PRIMARY KEY
                    , pickup_datetime_hour TIMESTAMP
                    , pickup_location_id SMALLINT
                    , num_pickup SMALLINT
                );
                """
            )    
            logger.info("Created %s table", self._pickup_table)
            
    def upsert_pickup_data(self, data: pl.DataFrame):
        """
        Upserts data from a processed file into the pickup_hourly table.
        Duckdb and Polars have a strong interoperability, polars DF
        are part of the scope of a DuckDB connection therefore they can
        be reference as SQL tables
        
        https://duckdb.org/docs/guides/python/polars.html
        """
        
        with self._get_connection() as conn:
                
            statement = f"""
                CREATE OR REPLACE TEMP TABLE stg_pickup_hourly AS
                SELECT * 
                FROM data;
                
                INSERT INTO {DATABASE_NAME}.{SCHEMA}.pickup_hourly  
                SELECT * FROM stg_pickup_hourly
                ON CONFLICT(key)
                DO UPDATE SET num_pickup = EXCLUDED.num_pickup;
                
                DROP TABLE stg_pickup_hourly;
            """    
            conn.execute(statement)
            
            logger.info("Upserted into dwh.main.pickup_hourly")
            
            
    def fetch_pickup_data(self, from_date: datetime, to_date: datetime, pickup_locations: list[int] | None = None) -> pl.DataFrame:
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
        
        with self._get_connection() as conn:
            query = f"""
            SELECT 
                CAST(pickup_datetime_hour AS DATETIME) AS pickup_datetime_hour
                , CAST(pickup_location_id AS FLOAT) AS pickup_location_id
                , CAST(sum(num_pickup) AS FLOAT) AS num_pickup
            FROM 
                {DATABASE_NAME}.{SCHEMA}.pickup_hourly
            WHERE 
                pickup_datetime_hour >= '{from_date}' 
                AND pickup_datetime_hour < '{to_date}'
                AND IF(LENGTH({pickup_locations}) > 0, list_contains({pickup_locations}, pickup_location_id), TRUE)
            GROUP BY 1,2
            ORDER BY 1,2
            """
            df = conn.sql(query).pl()  
        return df

