import polars as pl
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline


from src.config import ModelConfig


##################################################################################### Feature Engineering
def get_time_lags(df: pl.DataFrame, n_lags: list[int], drop_nulls:bool=True) -> pl.DataFrame:
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
        .with_columns([
            pl.col("num_pickup")
            .sort_by(["pickup_location_id", "pickup_datetime_hour"])
            .shift(i)
            .over("pickup_location_id")
            .alias(f"num_pickup_{i}d_ago") for i in n_lags
        ])
        # .drop_nulls()
    )
    
    if drop_nulls:
        return df.drop_nulls()
    return df


##################################################################################### Scikit-learn transformers


# class LagTransformer(BaseEstimator, TransformerMixin): 
#     """A scikit-learn transformer that generates time-lagged features for the number of pickups.
#     It's essentially a wrapper around the get_time_lags function to be able to use it on a Scikit-learn
#     pipeline. 
#     # TODO - Deprecate this class and integrate its functionality into the Baseline model
#     """
#     def __init__(self, lags:list[int]):
#         self.lags = lags
    
#     def fit(self, X:pl.DataFrame, y=None):
#         return self
    
#     def transform(self, X: pl.DataFrame):
#         df = get_time_lags(X, self.lags)
#         return df[self.get_feature_names()]
        
#     def get_feature_names(self) -> list[str]:
#         """
#         This method ensures the inclusion of the target column alongside the generated features.
#         Retaining the target column is crucial for subsequent pipeline steps, as it is required for model training and evaluation.
#         """
#         return [f"num_pickup_{i}d_ago" for i in self.lags] + [ModelConfig.TS_INDEX, ModelConfig.TARGET]
    
    

class MeanLagPredictor(BaseEstimator, RegressorMixin):
    """Model that predicts the number of pickups for a given time period by averaging the number of pickups in the past.
    # TODO - Add multi-step capacity by integrating lag predictor here
    """
    
    def __init__(self, ts_index:str = ModelConfig.TS_INDEX, lags: list[int] = ModelConfig.LAGS, random_state:int = 25):
        self.ts_index = ts_index
        self.lags = lags
        self.random_state = random_state
        self.residuals = None
        
    def fit(self, X:pl.DataFrame, y:pl.DataFrame):
        
        yhat_sample = self.predict(X, return_prediction_interval=False)
        self.residuals = (
            X.join(yhat_sample, on=self.ts_index, how="inner")
            .select(
                (pl.col("num_pickup") - pl.col("prediction")).alias("residual")
            )
            ["residual"]
            .to_list()
        )   
        return self
    
    
    
    def predict(self, X:pl.DataFrame, return_prediction_interval:bool=True, calculate_lags:bool=True, **kwargs) -> pl.DataFrame:
        """The -1 is because we remove the
        index column from the average calculation.

        Args:
            X (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """
        if calculate_lags:
            X = (
                X
            .pipe(self.get_lagged_features, **kwargs)
            )   
            
        pred = (
            X
            .select(
                pl.col(self.ts_index)
                , (pl.sum_horizontal(pl.exclude([self.ts_index])) / (X.shape[1] - 1)).alias("prediction")
            )
        )
        
        if return_prediction_interval:
            return self.get_prediction_intervals(pred, **kwargs)
        
        return pred
    
            
    def get_lagged_features(self, X:pl.DataFrame, target:str = ModelConfig.TARGET, drop_nulls:bool=True, **kwargs) -> pl.DataFrame:
        """
        This method ensures the inclusion of the target column alongside the generated features.
        Retaining the target column is crucial for subsequent pipeline steps, as it is required for model training and evaluation.
        """
        return (
            X
            .pipe(get_time_lags, self.lags, drop_nulls)
            [[f"num_pickup_{i}d_ago" for i in self.lags] + [self.ts_index, target]]
        )
    
    def get_prediction_intervals(self, pred:pl.DataFrame, B:int = 500, CIs:list[int] = [2.5, 97.5], **kwargs) -> pl.DataFrame:
        """
        Generates prediction intervals for the predictions made by the model.

        This method simulates B bootstrap samples of the residuals to generate prediction intervals for each prediction.
        The prediction intervals are determined based on the specified confidence intervals (CIs).

        Args:
            X (pl.DataFrame): The input DataFrame containing the features and predictions.
            B (int, optional): The number of bootstrap samples to generate. Defaults to 500.
            CIs (list[int], optional): The confidence intervals for which to generate the lower and upper bounds. Defaults to [2.5, 97.5].

        Returns:
            pl.DataFrame: A DataFrame containing the original predictions along with the lower and upper bounds of the prediction intervals.
        """
        
        rng = np.random.default_rng(seed=self.random_state)
        bootstrap_errors = pl.DataFrame(
            rng.choice(self.residuals, (pred.shape[0], B))
            , schema=[f"prediction_{x}" for x in range(B)]
        )
        return (
            pred
            .hstack(bootstrap_errors)
            .with_columns(
                [(pl.col("prediction") + pl.col(f"prediction_{x}")).alias(f"prediction_{x}") for x in range(B)]
            )
            .melt(id_vars=[self.ts_index, "prediction"])
            .group_by([pl.col(self.ts_index), pl.col("prediction")])
            .agg(
                pl.col("value").quantile(CIs[0] / 100).alias("lower_bound")
                , pl.col("value").quantile(CIs[1] / 100).alias("upper_bound")
            )
            .select([pl.col(self.ts_index), pl.col("prediction"), pl.col("lower_bound"), pl.col("upper_bound")])
            .sort(self.ts_index)
        )




# model = Pipeline([
#     ("lag_transformer", LagTransformer(lags=ModelConfig.LAGS))
#     , ("mean_predictor", MeanLagPredictor())
# ])

model = Pipeline([
    ("mean_predictor", MeanLagPredictor())
])

