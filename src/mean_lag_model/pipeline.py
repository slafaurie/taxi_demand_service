import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline


from src.mean_lag_model.config import ModelConfig


##################################################################################### Feature Engineering
def get_time_lags(df: pl.DataFrame, 
                  n_lags: list[int],
                  target:str='y',
                  ts_column:str='ds',
                  unique_id:str='unique_id',
                  drop_nulls:bool=True
                ) -> pl.DataFrame:
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
    
    df = (
        df
        .select(
        [ts_column, unique_id, target]
        +
        [
            pl.col(target)
            .shift(i)
            .over(partition_by= unique_id,  order_by=ts_column)
            .alias(f"{target}__{i}__lag") for i in n_lags
        ])
    )
    
    if drop_nulls:
        return df.drop_nulls()
    return df


##################################################################################### Scikit-learn transformers
    
    

class MeanLagPredictor(BaseEstimator, RegressorMixin):
    """Model that predicts the number of pickups for a given time period by averaging the number of pickups in the past.
    # TODO - Add multi-step capacity by integrating lag predictor here
    """
    
    def __init__(self, lags: list[int], unique_id:str='unique_id', ts_index:str='ds', target:str='y', random_state:int = 25):
        self.ts_index = ts_index
        self.lags = lags
        self.unique_id=unique_id
        self.target=target
        self.random_state = random_state
        self.residuals = None
        
    def fit(self, X:pl.DataFrame, y:pl.DataFrame):
        
        yhat_sample = self.predict(X, return_prediction_interval=False)
        self.residuals = (
            X.join(yhat_sample, on=self.ts_index, how="inner")
            .select(
                (pl.col(self.target) - pl.col("prediction")).alias("residual")
            )
            ["residual"]
            .to_list()
        )   
        return self
    
    
    
    def predict(self, X: pl.DataFrame) -> pl.DataFrame:
        """The -1 is because we remove the
        index column from the average calculation.

        Args:
            X (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """
        X = (
            X
            .pipe(get_time_lags, self.lags, self.target, self.ts_index, self.unique_id, True)
        )
        
        
                   
        pred = (
            X
            .with_columns(
                (pl.sum_horizontal(cs.contains("lag").truediv(cs.contains))
            )
        )
        return pred



