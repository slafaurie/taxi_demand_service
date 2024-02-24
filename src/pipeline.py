import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline


from src.config import ModelConfig


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
    """A scikit-learn transformer that generates time-lagged features for the number of pickups.
    It's essentially a wrapper around the get_time_lags function to be able to use it on a Scikit-learn
    pipeline. 
    """
    def __init__(self, lags:list[int]):
        self.lags = lags
    
    def fit(self, X:pl.DataFrame, y=None):
        return self
    
    def transform(self, X: pl.DataFrame):
        df = get_time_lags(X, self.lags)
        return df[self.get_feature_names()]
        
    def get_feature_names(self) -> list[str]:
        """
        This method ensures the inclusion of the target column alongside the generated features.
        Retaining the target column is crucial for subsequent pipeline steps, as it is required for model training and evaluation.
        """
        return [f"num_pickup_{i}d_ago" for i in self.lags] + [ModelConfig.TS_INDEX, ModelConfig.TARGET]
    
    

class MeanLagPredictor(BaseEstimator, RegressorMixin):
    """Model that predicts the number of pickups for a given time period by averaging the number of pickups in the past.
    """
    
    def __init__(self, ts_index:str = ModelConfig.TS_INDEX):
        self.ts_index = ts_index
        self.residuals = None
        
    def fit(self, X:pl.DataFrame, y:pl.DataFrame):
        yhat_sample = self.predict(X)
        self.residuals = (
            X.join(yhat_sample, on=self.ts_index, how="inner")
            .select([
                pl.col(self.ts_index)
                , (pl.col("num_pickup") - pl.col("prediction")).alias("residual")
            ])
        )   
        return self
    
    def predict(self, X:pl.DataFrame) -> pl.DataFrame:
        """The -1 is because we remove the
        index column from the average calculation.

        Args:
            X (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """
        return (
            X
            .select(
                pl.col(self.ts_index)
                , (pl.sum_horizontal(pl.exclude([self.ts_index])) / (X.shape[1] - 1)).alias("prediction")
            )
        )


model = Pipeline([
    ("lag_transformer", LagTransformer(lags=ModelConfig.LAGS))
    , ("mean_predictor", MeanLagPredictor())
])

