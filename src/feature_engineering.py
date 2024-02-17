import polars as pl


##################################################################################### Feature Engineering

def get_time_lags(df: pl.DataFrame, n_lags: list[int]) -> pl.DataFrame:
    """
    
    Description
    
    Generates time-lagged features for the number of pickups. It receives a list with the lags, in days, to generate. For example:
    - 1 means 1 day ago
    - 24 -> 24 days ago

    Parameters:
    - df (pl.DataFrame): The DataFrame containing the pickup data.
    - n_lags (list[int]): The number of lagged time periods to generate.

    Returns:
    - pl.DataFrame: The original DataFrame with n_lags new columns added, each representing the number of pickups n hours ago.
    """
    return (
        df
        .with_columns([
            pl.col("num_pickup").sort_by(["pickup_location_id", "pickup_datetime_hour"]).shift(i).over("pickup_location_id").alias(f"num_pickup_{i}d_ago") for i in n_lags
        ])
        .drop_nulls()
    )
    
