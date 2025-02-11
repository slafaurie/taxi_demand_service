""" 
TODO 
- Expose ETL methods
- Expose Repo DDL
"""

import typer
import datetime
from typing_extensions import Annotated

from src.etl.pipeline import batch_etl
from src.adapters.base import initialize_repository

app = typer.Typer()
etl_app = typer.Typer()

app.add_typer(etl_app, name='etl')



@etl_app.command()
def download_taxi_data(
    from_date: str,
    to_date: str,
    repo: Annotated[str, typer.Option()] = "local"
):
    """ 
    Download taxi data from source 
    """
    
    repo_obj = initialize_repository(repo)
     
    batch_etl(
        repo = repo_obj,
        from_date = datetime.date.fromisoformat(from_date),
        to_date = datetime.date.fromisoformat(to_date)
    )
    
    


