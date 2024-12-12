import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin


##################################################################################### Feature Engineering
def get_time_lags(df: pl.DataFrame, 
                  lags: list[int],
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
            .alias(f"{target}__{i}__lag") for i in lags
        ])
    )
    
    if drop_nulls:
        return df.drop_nulls()
    return df


##################################################################################### Scikit-learn transformers
    
    

class MeanLagPredictor(BaseEstimator, RegressorMixin):
    """Model that predicts the number of pickups for a given time period by averaging the number of pickups in the past.
    # TODO - Add multi-step capacity by integrating lag predictor here
    
    Column names are inspired by Nixtla StatsForecast 
    https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/statisticalneuralmethods.html
    """ 
    
    def __init__(self, lags: list[int], freq:str, unique_id:str='unique_id', ds:str='ds', target:str='y',  random_state:int = 25):
        # TODO | 2024-12-11 | freq validation
        # Frequency should be one of Polars compatible string
        # listed here https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by_dynamic.html
        self.ds = ds
        self.lags = lags
        self.unique_id=unique_id
        self.target=target
        self.freq = freq
        self.random_state = random_state
        self.residuals = None
        
    def fit(self, X:pl.DataFrame, y:pl.DataFrame=None):
        """
        Fitting function. 'y' is present to maintain consistency with
        ML libraries API but it's not required because the predict function
        returns the y as well due to TS data nature.

        Args:
            X (pl.DataFrame): _description_
            y (pl.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        self.residuals = (
            self.predict(X)
            .filter(pl.col(self.target).is_not_null())
            .select(
                (pl.col(self.target) - pl.col("y_hat")).alias("residual")
            )
            ["residual"]
            .to_list()
        )   
        return self
    
    
    
    def predict(self, X: pl.DataFrame) -> pl.DataFrame:
        """Implement a 1-step ahead forecast. The to take into account
        that pipeline uses lags value of the target, we first add one more row equal to
        the next prediction before calculating the lags. In addition, since
        the model needs to use lags, it filters the first data points where it's not possible
        to compute prediction.
        
        Note that the function returns the input DF plus 1 more row (the forecast) and
        a new column (the in-sample prediction).
        """
        
        input_col = X.columns
        
        # extend the dataframe
        forecast =(
            X
            .pipe(self._get_last_ds)
            .with_columns(
                pl.col(self.ds).dt.offset_by(self.freq),
                pl.lit(None).alias(self.target)
            )
            .select(input_col)
        )
        
        # TODO | BUG | 2024-12-11 | freq and max lag relation
        # ideally the offset should be max_lag * freq, there shouldn't be
        # problems for single digit frequency (1d, 1h) but for every other
        # frequency type (say 1d1w), this give wrong values
        
        # calculate the date from which we can calculate predictions
        max_lag = max(self.lags) 
        cutoff_offset = f"{max_lag}{self.freq[-1]}"
        cutt_off_date = (
            X
            .group_by(
                self.unique_id
            )
            .agg(
                cutoff_from = pl.col(self.ds).min().dt.offset_by(cutoff_offset)
            )
        )
        
        X_with_pred = (
                X 
                .vstack(forecast)
                .pipe(get_time_lags, self.lags, self.target, self.ds, self.unique_id, False)  
                .join(cutt_off_date, on=[self.unique_id], how='inner') 
                .filter(pl.col(self.ds).gt(pl.col('cutoff_from')))
                .with_columns(
                    y_hat = pl.mean_horizontal(cs.contains("lag"))
                )
                .select(input_col+["y_hat"])
        )

        return X_with_pred
    
    def forecast(self, X:pl.DataFrame, h:int):
        """Forecast just means multiple multi-step prediction into the future.
        This function applies the recursive strategy in which we iterate over 1-step
        ahead forecast and the result becomes part of the input data and is used for following prediction.
        More info here: https://skforecast.org/0.14.0/user_guides/autoregresive-forecaster
        
        Note that the function returns the input dataframe plus h more rows and one
        additional column to flag which rows are predictions
        
        

        Args:
            X (pl.DataFrame): _description_
            h (int): _description_

        Returns:
            _type_: _description_
        """
        input_col = X.columns
        latest_ds = X.pipe(self._get_last_ds, 'last_ds')
        for i in range(1,h+1):
            X = (
                X
                .pipe(self.predict)
                .with_columns(
                    pl.coalesce([self.target, 'y_hat']).alias(self.target)
                )
                .select(input_col)
            )
            
        # add a flag indicating which are forecasted value
        X = (
            X
            .join(latest_ds, on=[self.unique_id])
            .with_columns(
                IsForecasted=pl.col(self.ds).gt(pl.col('last_ds'))
            )
            .select(input_col + ['IsForecasted'])
        )
        return X
    
    def _get_last_ds(self, X:pl.DataFrame, return_name:str=None):
        if not return_name:
            return_name = self.ds
            
        return (
            X
            .group_by(self.unique_id)
            .agg(
                pl.col(self.ds).max().alias(return_name)
            )
        )



