

from abc import ABC, abstractmethod
from datetime import datetime
import polars as pl


#### DATABASE

DATABASE_NAME = 'nyc_trips'
SCHEMA = 'main'


""" 
TODO | 2025-02-10 | function to initialize the repo

create an entrypoint for the repositories. The function
should receive make sure to initialize the correct repository,
passing optional arguments if needed and validating only the repositories listed are used
"""
class NYCTaxiRepository(ABC):
        
    @abstractmethod
    def create_tables(self):
        """This method setup creates the schema & tables within the repository
        and guarantee their existence.
        """
        pass
        
    @abstractmethod
    def upsert_pickup_data(self, data: pl.DataFrame):
        pass
    
    @abstractmethod
    def fetch_pickup_data(self, from_date: datetime, to_date: datetime, pickup_locations: list[int] | None = None) -> pl.DataFrame:
        pass
    