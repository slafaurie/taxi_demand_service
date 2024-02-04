from pathlib import Path
import polars as pl
import requests
from tqdm import tqdm


from src.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, FILE_PATTERN, BASE_URL
from src.logger import get_logger

logger = get_logger()

##################################################################################### Data Loading and Validation 

def download_file_from_source_into_raw_folder(year:int, month:int) -> Path:
    """
    Downloads a file from a specified source URL into the raw data directory.

    This function constructs a file path using the year and month parameters, then downloads the file
    from a constructed URL. The file is saved into the raw data directory. If the download is successful,
    the function logs the download action and returns the path to the downloaded file.

    Parameters:
    - year (int): The year part of the file to be downloaded.
    - month (int): The month part of the file to be downloaded.

    Returns:
    - Path: The path to the downloaded file in the raw data directory.
    """
    file = RAW_DATA_DIR / Path(FILE_PATTERN.format(year=year, month=month))
    url = BASE_URL.format(year=year, month=month)
    response = requests.get(url)
    response.raise_for_status()
    if response.status_code == 200:
        logger.info("Downloading file from %s to %s", url, file)
        with open(file, "wb") as f:
            f.write(response.content)
    return file

def validate_file(file:Path, year:int, month:int) -> pl.DataFrame:
    """
    Validates the given parquet file to ensure all records are within the specified year and month.

    This function reads the parquet file, filters the records to only include those where the
    'tpep_pickup_datetime' falls within the specified year and month, and then returns a DataFrame
    containing only the 'pickup_datetime' and 'pickup_location_id' columns for the filtered records.

    Parameters:
    - file (Path): The path to the parquet file to be validated.
    - year (int): The year to filter the 'tpep_pickup_datetime' by.
    - month (int): The month to filter the 'tpep_pickup_datetime' by.

    Returns:
    - pl.DataFrame: A DataFrame containing the 'pickup_datetime' and 'pickup_location_id' columns
      for records that fall within the specified year and month.
    """
    
    df = pl.read_parquet(file)

    range_expresion = (
        ( (pl.col("tpep_pickup_datetime").dt.year() == year)
            & (pl.col("tpep_pickup_datetime").dt.month() == month))
    )


    aggregate_values = (
        df
        .select([
            pl.col("tpep_pickup_datetime").filter(range_expresion).count().alias("records_in_range")
            , pl.col("tpep_pickup_datetime").count().alias("total_records")
        ]
        )
    )

    records_in_range = aggregate_values["records_in_range"].item()
    total_records = aggregate_values["total_records"].item()

    logger.info("Validation for file: %s", file)
    logger.info("Total records: %s", total_records)
    logger.info("Records deleted: %s", total_records-records_in_range)
    logger.info("Percentage: %.2f%%", records_in_range / total_records * 100)

    clean_data = (
        df
        .filter(range_expresion)
        .select([
            pl.col("tpep_pickup_datetime").alias("pickup_datetime")
            , pl.col("PULocationID").alias("pickup_location_id")
        ])
    )

    return clean_data

def delete_raw_data_file(file: Path) -> None:
    """
    Deletes a file in the data/raw directory after it have been validated and processed.

    Parameters:
    - file (Path): file to delete
    
    Returns:
    None.
    """
    try:
        file.unlink()
        logger.info(f"Deleted file: %s", file)
    except Exception as e:
        logger.error(f"Error deleting file %s: %s", file, e)
        
def load_raw_data(year:int, months: list[int] | None = None) -> None:
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
    
    logger.info("Downloading data for year %s", year)
    
    if months is None: 
        months = range(1, 13)
    if isinstance(months, int):
        months = [months]
 
    for month in tqdm(months):
        try:
            file = download_file_from_source_into_raw_folder(year, month)
            validate_file(file, year, month).write_parquet(PROCESSED_DATA_DIR / Path(FILE_PATTERN.format(year=year, month=month)))
            delete_raw_data_file(file)
        except requests.exceptions.HTTPError as e:
            logger.error("Error downloading data for year %s and month %s: %s", year, month, e)
            continue
    logger.info("Data for year %s has been downloaded and validated", year)
    

##################################################################################### Data Transformation

