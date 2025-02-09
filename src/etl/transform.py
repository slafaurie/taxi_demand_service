
from datetime import datetime
import polars as pl

from src.logger import get_logger
from src.etl.models import NYCPickupHourlySchema


logger = get_logger(__name__)




def standardize_raw_schema(df: pl.DataFrame):
    """

    Args:
        df (pl.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    
    RAW_SCHEMA = {
        "tpep_pickup_datetime": {"clean_name": "pickup_datetime", "type": pl.Datetime},
        "passenger_count": {"clean_name": "passenger_count", "type": pl.Int32}, 
        "PULocationID": {"clean_name": "pickup_location_id", "type": pl.Int32}
    }
    
    return (
        df 
        .select(RAW_SCHEMA.keys())
        .rename({key: val.get('clean_name') for key,val in RAW_SCHEMA.items()})
        .cast({val.get('clean_name'): val.get('type') for val in RAW_SCHEMA.values()})
    )
    
    
def filter_out_of_range_points(df: pl.DataFrame, year:int, month:int) -> pl.DataFrame:
    """
    """
    
    range_expresion = (
        ( (pl.col("pickup_datetime").dt.year() == year)
            & (pl.col("pickup_datetime").dt.month() == month))
    )


    aggregate_values = (
        df
        .select([
            pl.col("pickup_datetime").filter(range_expresion).count().alias("records_in_range")
            , pl.col("pickup_datetime").count().alias("total_records")
        ]
        )
    )

    records_in_range = aggregate_values["records_in_range"].item()
    total_records = aggregate_values["total_records"].item()

    logger.info("Validate data for year: %s and month: %s", year, month)
    logger.info("Total records: %s", total_records)
    logger.info("Records deleted: %s", total_records-records_in_range)
    logger.info("Percentage: %.2f%%", records_in_range / total_records * 100)

    clean_data = (
        df
        .filter(range_expresion)
    )

    return clean_data

def _generate_hourly_datetimes_with_ranges(year: int, month: int) -> pl.DataFrame:
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
            time_unit="us",
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
    logger.info("Aggregating data to hourly frequency")

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
    
    hourly_df = _generate_hourly_datetimes_with_ranges(year, month)
    
    
    return ( 
            hourly_df
            .join(hourly_pickups.select(pl.col("pickup_location_id").unique()), how="cross")
            .join(hourly_pickups, on=["pickup_datetime_hour", "pickup_location_id"], how="left")
            .with_columns(
                pl.col("num_pickups").fill_null(pl.lit(0))
            )
    )
    
def add_surrogate_key(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates a surrogate key for each record in the DataFrame by concatenating the pickup_datetime_hour and pickup_location_id.

    This function takes a DataFrame and adds a new column named 'key' which is a string concatenation of 'pickup_datetime_hour' and 'pickup_location_id', separated by a hyphen. It also ensures that 'pickup_location_id' and 'num_pickups' are cast to Int16 for consistency.

    Parameters:
    - df (pl.DataFrame): The DataFrame to process.

    Returns:
    - pl.DataFrame: The original DataFrame with an additional 'key' column and casted 'pickup_location_id' and 'num_pickups' columns.
    """
    return df.with_columns([
        pl.concat_str(pl.col("pickup_datetime_hour"), pl.lit("-"), pl.col("pickup_location_id")).alias("key")
    ])


def transform_raw_data(df: pl.DataFrame, year: int, month:int):
    clean_data = (
        df 
        .pipe(standardize_raw_schema)
        .pipe(filter_out_of_range_points, year, month)
        .pipe(aggregate_pickup_into_timeseries_data, year, month)
        .pipe(add_surrogate_key)
        
    )
    
    return NYCPickupHourlySchema.enforce_schema(clean_data)
