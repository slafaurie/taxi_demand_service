

from abc import ABC, abstractmethod
from datetime import datetime
import polars as pl


#### DATABASE

DATABASE_NAME = 'nyc_trips'
SCHEMA = 'main'



class NYCTaxiRepository(ABC):
        
    @abstractmethod
    def _ddl(self):
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
    