##################################################################################### Feature Engineering

def get_time_lags(df: pl.DataFrame, n_lags: int) -> pl.DataFrame:
    """
    Generates time-lagged features for the number of pickups.

    This function takes a DataFrame and an integer n_lags to generate n_lags new columns in the DataFrame. Each new column represents the number of pickups n hours ago, where n ranges from 1 to n_lags. The function sorts the DataFrame by 'pickup_location_id' and 'pickup_datetime_hour' before shifting to ensure that the lagged values are correctly aligned with the corresponding times and locations.

    Parameters:
    - df (pl.DataFrame): The DataFrame containing the pickup data.
    - n_lags (int): The number of lagged time periods to generate.

    Returns:
    - pl.DataFrame: The original DataFrame with n_lags new columns added, each representing the number of pickups n hours ago.
    """
    return (
        df
        .with_columns([
            pl.col("num_pickups").sort_by(["pickup_location_id", "pickup_datetime_hour"]).shift(i).over("pickup_location_id").alias(f"num_pickups_{i}h_ago") for i in range(1, n_lags+1)
        ])
        .drop_nulls()
    )
    
def generate_ts_features_for_file(year: int, month: int, n_lags: int) -> pl.DataFrame:
    """
    Generates time-lagged features for the number of pickups in a given month.

    This function reads the pickup data for the specified year and month, aggregates it into hourly data, and then generates time-lagged features using the get_time_lags function.

    Parameters:
    - year (int): The year of the month for which to generate time-lagged features.
    - month (int): The month for which to generate time-lagged features.
    - n_lags (int): The number of lagged time periods to generate.

    Returns:
    - pl.DataFrame: The DataFrame containing the time-lagged features.
    """
    return (
        read_file(PROCESSED_DATA_DIR, year, month)
        .pipe(get_time_lags, n_lags)
        .write_parquet(TRANSFORMED_DATA_DIR / FILE_PATTERN.format(year=year, month=month))
    )