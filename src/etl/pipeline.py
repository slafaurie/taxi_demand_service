import polars as pl
from datetime import date
from tqdm import tqdm


from src.common import get_logger
from src.adapters.base import NYCTaxiRepository


from src.etl.transform import transform_raw_data
from src.etl.helpers import generate_list_of_months


logger = get_logger(__name__)

def fetch_raw_data(year:int, month:int) -> pl.DataFrame:
    """
    """
    FILE_PATTERN = "yellow_tripdata_{year}-{month:02d}.parquet"
    BASE_URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{FILE_PATTERN}"


    url = BASE_URL.format(year=year, month=month)
    
    try:
        df = pl.read_parquet(url)
        return df
    except Exception:
        logger.exception("Error downloading %s", url)
        
    
def file_etl(repo: NYCTaxiRepository, year:int, month:int) -> None:
    """
    Executes the ETL process for a single file corresponding to a given year and month.
    Parameters:
    - year (int): The year of the data file to process.
    - month (int): The month of the data file to process.

    Returns:
    None. The function performs operations that result in writing to disk and database but does not return any value.
    """
    
    
    # clean and load the file
    clean_data =  (
        fetch_raw_data(year, month)
        .pipe(transform_raw_data, year, month)
    )
    
    repo.upsert_pickup_data(clean_data)
    
    
def batch_etl(repo: NYCTaxiRepository, from_date:date, to_date:date) -> None:
    """
    Loads raw taxi trip data for a specified year and optional list of months, validates it, and saves the validated data.

    This function downloads raw taxi trip data for the specified year and months. If no months are provided, it defaults to all months in the year.
    After downloading, it validates the data by checking if the records fall within the specified year and month(s) and then saves the validated data
    into a processed data directory in parquet format.

    Parameters:
    - year (int): The year for which to download and validate the data.
    - months (Optional[list[int]]): An optional list of integers representing the months for which to download and validate the data.
      If None, data for all months in the specified year will be processed.

    Returns:
    None. The function saves the validated data into a processed data directory without returning any value.
    """
    
    logger.info("Downloading data from %s to %s", from_date, to_date)
    
    list_of_months = generate_list_of_months(from_date, to_date)
 
    for period in tqdm(list_of_months):
        try:
            file_etl(repo, period.year, period.month)
        except Exception:
            logger.exception("Error downloading data for %s", period)
            continue
    logger.info("Data from %s to %s has been downloaded and validated", from_date, to_date)

