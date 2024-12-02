from pathlib import Path
from datetime import datetime 
from argparse import ArgumentParser

import polars as pl
import requests
from tqdm import tqdm


from src.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, FILE_PATTERN, BASE_URL
from src.logger import get_logger
from src.etl.dwh import run_database_operation

logger = get_logger(__name__)


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

def delete_file(file: Path) -> None:
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

def read_file(folder:Path, year:int, month:int) -> pl.DataFrame:
    """
    Reads a parquet file for a given year and month from a specified folder.

    This function constructs the file path using the provided folder, year, and month. It then reads the parquet file
    located at that path into a Polars DataFrame and returns it.

    Parameters:
    - folder (Path): The folder where the parquet file is located.
    - year (int): The year part of the file to be read.
    - month (int): The month part of the file to be read.

    Returns:
    - pl.DataFrame: The Polars DataFrame containing the data from the parquet file.
    """
    return pl.read_parquet(folder / FILE_PATTERN.format(year=year, month=month))

def generate_hourly_datetimes_with_ranges(year: int, month: int) -> pl.DataFrame:
    """
    Generates a Polars DataFrame with a single column containing datetimes for every hour in the specified month
    using the pl.datetime_ranges function.

    Parameters:
    - year (int): The year of the month for which to generate hourly datetimes.
    - month (int): The month for which to generate hourly datetimes.

    Returns:
    - pl.DataFrame: A DataFrame with a single column named 'datetime', containing hourly datetimes for the specified month.
    """
    # Calculate the start datetime of the month
    start_date = datetime(year, month, 1)
    # Handle December separately to avoid month overflow
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    # Create a DataFrame from the datetime range
    df = pl.DataFrame({
        "pickup_datetime_hour": pl.datetime_range(
            start=start_date, 
            end=end_date, 
            interval="1h", 
            eager=True, 
            time_unit="ns",
            closed="left")
    })
    
    return df

def aggregate_pickup_into_timeseries_data(df: pl.DataFrame, year: int, month: int) -> pl.DataFrame:
    """
    Aggregate the pickup data into hourly timeseries data for the specified year and month. Timeseries
    must contain all hours in the month, and the number of pickups for each pickup location ID at each hour.
    
    In case there are no pickups for a given hour and pickup location ID, the number of pickups should be 0.

    Parameters:
    - year (int): The year of the month for which to aggregate the pickup data.
    - month (int): The month for which to aggregate the pickup data.
    - df (pl.DataFrame): The DataFrame containing the pickup data to be aggregated.

    Returns:
    - pl.DataFrame: The DataFrame containing the aggregated pickup data.
    """
    # Truncate the pickup datetime to the nearest hour and group by the pickup location ID
    hourly_pickups = (
        df
        .group_by([
            pl.col("pickup_datetime").dt.truncate("1h").alias("pickup_datetime_hour"),
            pl.col("pickup_location_id")
        ])
        .agg(
            pl.col("pickup_location_id").count().alias("num_pickups")
        )
    )
    
    hourly_df = generate_hourly_datetimes_with_ranges(year, month)
    
    
    return ( 
            hourly_df
            .join(hourly_pickups.select(pl.col("pickup_location_id").unique()), how="cross")
            .join(hourly_pickups, on=["pickup_datetime_hour", "pickup_location_id"], how="left")
            .with_columns(
                pl.col("num_pickups").fill_null(pl.lit(0))
            )
    )
    
def generate_surrogate_key(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates a surrogate key for each record in the DataFrame by concatenating the pickup_datetime_hour and pickup_location_id.

    This function takes a DataFrame and adds a new column named 'key' which is a string concatenation of 'pickup_datetime_hour' and 'pickup_location_id', separated by a hyphen. It also ensures that 'pickup_location_id' and 'num_pickups' are cast to Int16 for consistency.

    Parameters:
    - df (pl.DataFrame): The DataFrame to process.

    Returns:
    - pl.DataFrame: The original DataFrame with an additional 'key' column and casted 'pickup_location_id' and 'num_pickups' columns.
    """
    return df.select([
        pl.concat_str(pl.col("pickup_datetime_hour"), pl.lit("-"), pl.col("pickup_location_id")).alias("key"),
        pl.col("pickup_datetime_hour"),
        pl.col("pickup_location_id").cast(pl.Int16),
        pl.col("num_pickups").cast(pl.Int16)
    ])
    
def file_etl(year:int, month:int) -> None:
    """
    Executes the ETL process for a single file corresponding to a given year and month.

    This function encompasses the entire ETL (Extract, Transform, Load) process for taxi trip data of a specific year and month. 
    It downloads the raw data file, validates and cleans it, aggregates it into timeseries data, generates a surrogate key, 
    writes the processed data to a parquet file, upserts the data into a database, and finally cleans up the intermediate files.

    Parameters:
    - year (int): The year of the data file to process.
    - month (int): The month of the data file to process.

    Returns:
    None. The function performs operations that result in writing to disk and database but does not return any value.
    """
    
    # Download the file
    raw_file = download_file_from_source_into_raw_folder(year, month)
    processed_file = PROCESSED_DATA_DIR / FILE_PATTERN.format(year=year, month=month)
    raw_df = validate_file(raw_file, year, month)
    delete_file(raw_file)
    
    # clean and load the file
    (
        raw_df
        .pipe(aggregate_pickup_into_timeseries_data, year, month)
        .pipe(generate_surrogate_key)
        .write_parquet(processed_file)
    )
    
    # upsert the file into the database
    run_database_operation("upsert_pickup_data", year, month)
    delete_file(processed_file)
    
def batch_etl(year:int, months: list[int] | None = None) -> None:
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
            file_etl(year, month)
        except requests.exceptions.HTTPError as e:
            logger.error("Error downloading data for year %s and month %s: %s", year, month, e)
            continue
    logger.info("Data for year %s has been downloaded and validated", year)

if __name__ == "__main__":
    parser = ArgumentParser(description="ETL process for NYC taxi trip data")
    parser.add_argument("year", type=int, help="The year for which to download and validate the data")
    parser.add_argument("--months", type=int, nargs="+", help="The months for which to download and validate the data")
    args = parser.parse_args()
    batch_etl(args.year, args.months)