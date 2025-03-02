

from abc import ABC, abstractmethod
from datetime import datetime
import polars as pl


#### DATABASE

DATABASE_NAME = 'nyc_trips'
SCHEMA = 'main'





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
    
    
def initialize_repository(repo_type: str = "duckdb", **kwargs) -> NYCTaxiRepository:
    """Initialize and return a repository instance based on the specified type.
    
    Args:
        repo_type (str): Type of repository to initialize.
        **kwargs: Additional arguments to pass to the repository constructor
        
    Returns:
        NYCTaxiRepository: An initialized repository instance
        
    Raises:
        ValueError: If an unsupported repository type is specified
    """
    valid_repos = [
        "duckdb",
        "local"
    ]
    
    if repo_type not in valid_repos:
        raise ValueError(f"Unsupported repository type: {repo_type}. Must be one of: {valid_repos}")
        
    match repo_type:
        case "duckdb":
            from src.adapters.duck_repo import DuckDBRepository
            repo = DuckDBRepository(**kwargs)
        case "local":
            from src.adapters.local_repo import LocalRepository
            repo = LocalRepository(**kwargs)
    return repo
