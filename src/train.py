import polars as pl 
from datetime import datetime 
from src.config import TEST_DATA_FROM

def split_into_train_and_test(df:pl.DataFrame, cutoff_date:datetime = TEST_DATA_FROM):
    return (
        df.filter(pl.col("pickup_datetime_hour") < cutoff_date)
        ,  df.filter(pl.col("pickup_datetime_hour") >= cutoff_date)
    )
    
def split_target_and_features(df:pl.DataFrame, target:str):
    return df.drop(target), df[target]

