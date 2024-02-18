import polars as pl
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import TS_INDEX


##################################################################################### Feature Engineering

"""
TODO
This should not be a python function. This should be abstracted to the database operation.
I should be able to load the data already daily frequency. Modify the SQL query to allow for aggregation.
"""
def aggregate_to_daily(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by_dynamic("pickup_datetime_hour", every="1d", by="pickup_location_id").agg(pl.col("num_pickup").sum())

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
            pl.col("num_pickup")
            .sort_by(["pickup_location_id", "pickup_datetime_hour"])
            .shift(i)
            .over("pickup_location_id")
            .alias(f"num_pickup_{i}d_ago") for i in n_lags
        ])
        .drop_nulls()
    )

##################################################################################### Scikit-learn transformers


class LagTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self, lags:list[int]):
        self.lags = lags
    
    def fit(self, X:pl.DataFrame, y=None):
        return self
    
    def transform(self, X: pl.DataFrame):
        df = get_time_lags(X, self.lags)
        return df[self.get_feature_names()]
        
    def get_feature_names(self) -> list[str]:
        return [f"num_pickup_{i}d_ago" for i in self.lags] + [TS_INDEX]
    
    

