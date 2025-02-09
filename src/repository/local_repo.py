import os 
import polars as pl
from datetime import datetime
from pathlib import Path



from src.etl.models import NYCPickupHourlySchema
from src.repository.base import NYCTaxiRepository, DATABASE_NAME, SCHEMA
from src.logger import get_logger
from src.paths import DATA_DIR




logger = get_logger(__name__)


class LocalRepository(NYCTaxiRepository):
    
    FORMAT = 'parquet'
    
    def __init__(self, custom_root_dir:str = None):
        self.root_dir = self._resolve_root(custom_root_dir)
        
    def _resolve_root(self, custom_root_dir:str) -> Path:
        if custom_root_dir:
            resolved_root = Path(custom_root_dir)
        else:
            resolved_root = DATA_DIR / "local"
        resolved_root.mkdir(exist_ok=True)
        return resolved_root

    def create_tables(self):
        """This method setup creates the schema & tables within the repository
        and guarantee their existence.
        """
        
        (self.root_dir / DATABASE_NAME).mkdir(exist_ok=True)
        (self.root_dir / DATABASE_NAME / SCHEMA).mkdir(exist_ok=True)
        (self.root_dir / DATABASE_NAME / SCHEMA / "pickup_hourly").mkdir(exist_ok=True)
        
        logger.info("Created %s.%s.pickup_hourly table", DATABASE_NAME, SCHEMA)
        self._pickup_table = self.root_dir / DATABASE_NAME / SCHEMA / "pickup_hourly" / "data.parquet"
    
    @staticmethod
    def _deduplicate_pickup_data(new_data: pl.DataFrame, current_data:pl.DataFrame) -> pl.DataFrame:
        
        
        
        return (
            pl.concat([
                current_data.with_columns(priority=1)
                , new_data.with_columns(priority=0)
            ])
            .sort(['priority'], descending=False)
            .filter(
                pl.col('priority').cum_count().over('key') == 1
            )
            .sort(by='key')
            .drop(['priority'])
        )


    def upsert_pickup_data(self, data: pl.DataFrame):   
             
        if not self._pickup_table.exists():
            data.write_parquet(self._pickup_table)
        
        current_data = NYCPickupHourlySchema.enforce_schema(pl.read_parquet(self._pickup_table))
        current_data = self._deduplicate_pickup_data(new_data=data, current_data=current_data)
        current_data.write_parquet(self._pickup_table)
        logger.info('data persisted')

    
    
    def fetch_pickup_data(self, from_date: datetime, to_date: datetime, pickup_locations: list[int] | None = None) -> pl.DataFrame:
        
        if from_date > to_date:
            raise ValueError(f"{from_date} can't be higher than {to_date}")
        
        data = (
            pl.scan_parquet(self._pickup_table)
            .filter(
                pl.col('pickup_datetime_hour').is_between(from_date, to_date)
            )
        )
        
        if pickup_locations:
            if isinstance(pickup_locations, int):
                pickup_locations = [pickup_locations]
            
            data = (data
                    .filter(
                        pl.col('pickup_location_id').is_in(pickup_locations)
                    )
            )
        
        return NYCPickupHourlySchema.enforce_schema(data.collect())
            