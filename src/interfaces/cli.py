""" 
TODO 
- Expose ETL methods
- Expose Repo DDL
"""

import typer
import datetime
from typing_extensions import Annotated
from datetime import datetime

from src.etl.pipeline import batch_etl
from src.adapters.base import initialize_repository
from src.model.train import train_model as train_model_pipeline
from src.model.config import (
    TRAIN_DATA_FROM,
    TRAIN_DATA_TO,
    TEST_DATA_FROM,
    MAX_HORIZON,
    PICKUPS_LOCATION,
    CROSS_VALIDATION_FREQUENCY
)

app = typer.Typer()
etl_app = typer.Typer()
model_app = typer.Typer()

app.add_typer(etl_app, name='etl')
app.add_typer(model_app, name='model')



@etl_app.command()
def download_taxi_data(
    from_date:  Annotated[datetime, typer.Argument()],
    to_date:  Annotated[datetime, typer.Argument()],
    repo: Annotated[str, typer.Option()] = "duckdb"
):
    """ 
    Download taxi data from source 
    """
    
    repo_obj = initialize_repository(repo)
     
    batch_etl(
        repo = repo_obj,
        from_date = from_date,
        to_date = to_date
    )
    
@etl_app.command()
def create_tables(
    
    repo: Annotated[str, typer.Option()] = "duckdb"
):
    """ 
    Create tables
    """
    
    repo_obj = initialize_repository(repo)
    repo_obj.create_tables()
    
    

@model_app.command()
def train_model(
    train_data_from: Annotated[datetime, typer.Option()] = TRAIN_DATA_FROM,
    train_data_to: Annotated[datetime, typer.Option()] = TRAIN_DATA_TO,
    test_data_from: Annotated[datetime, typer.Option()] = TEST_DATA_FROM,
    pickup_locations: Annotated[list[int], typer.Option()] = PICKUPS_LOCATION,
    max_horizon: Annotated[int, typer.Option()] = MAX_HORIZON,
    cross_validation_split: Annotated[str, typer.Option()] = CROSS_VALIDATION_FREQUENCY,
    repo: Annotated[str, typer.Option()] = "duckdb"
):
    """
    Train the forecasting model using historical taxi pickup data.
    
    Args:
        train_data_from: Start date for training data
        train_data_to: End date for training data
        test_data_from: Start date for test data
        pickup_locations: List of pickup location IDs to train on
        max_horizon: Maximum number of days to forecast
        cross_validation_split: Frequency for cross-validation splits (e.g. '3mo')
        repo: Repository type to use ('duckdb' or other supported types)
    """
    repo_obj = initialize_repository(repo)
    
    train_model_pipeline(
        repo=repo_obj,
        train_data_from=train_data_from,
        train_data_to=train_data_to,
        test_data_from=test_data_from,
        pickup_locations=pickup_locations,
        max_horizon=max_horizon,
        cross_validation_split_frequency=cross_validation_split
    )