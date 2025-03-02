from datetime import datetime 
import polars as pl 

from mlforecast import MLForecast

from src.common import get_logger
from src.model.pipeline import build_model
from src.adapters.base import  NYCTaxiRepository


logger = get_logger("train")

def split_train_test(
        df:pl.DataFrame,
        test_from: datetime,
        every:str,
        ts_col:str='ds'
    ):
    
    """ 
    This splits the dataset into multiple folds taking into 
    account the time-series structure. This functions requires
    explicit time frequency of folds instead of the exact number. 
    It leverages Polars duration string language
    
    https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.datetime_range.html#polars.datetime_range
    
    It returns a list where each item is a (train, test) tuple. This functions
    - Assumes the train size increases on each subsquent fold.
    - Ignores any window size that test size may need. 
    """

    train_from = (
        df
        .select(pl.col(ts_col).min()).item(0,0)  
    )

    test_to = (
        df
        .select(pl.col(ts_col).max()).item(0,0)  
    )

    fold_info = (
        pl.DataFrame(
            pl.datetime_range(
                start=test_from,
                end=test_to,
                interval=every,
                eager=True,
                closed='left'
            )
        )
        .select(
            pl.lit(train_from).alias('train_from'),
            pl.col('literal').alias('test_from'),
            pl.col('literal').shift(-1).fill_null(pl.lit(test_to)).alias('test_to'),
        )
        .with_row_index('fold_id')
        .to_dicts()
    )
    
    logger.info('Numbers of folds: %s', len(fold_info))

    folds = []
    for fold_i in fold_info:
        fold_data = (
            df
            .filter(
                pl.col(ts_col).is_between(fold_i.get('train_from'), fold_i.get('test_to'), closed='left')
            )
            .with_columns(
                fold_id = pl.lit(fold_i.get('fold_id')),
                is_test = pl.col(ts_col).ge(fold_i.get('test_from'))
            )
        )
        folds.append(
            ( fold_data.filter(~pl.col('is_test')), fold_data.filter(pl.col('is_test')))
        )
    return folds

def rolling_window_forecast(
    model: MLForecast,
    h: int,
    df:pl.DataFrame, 
    step_size: int | None  = None,
    ts_col: str = 'ds',
    unique_id: str = 'unique_id',
    y: str = 'y'
) -> pl.DataFrame:
    """
    Performs rolling window forecast evaluation.

    For each cutoff point in the test set, generates h-step ahead forecasts
    and combines them with actual values for evaluation.

    Args:
        model: The MLForecast model to use for predictions
        h: Forecast horizon (number of steps ahead to predict)
        df: Test dataset to evaluate on
        step_size: Size of the step between forecast origins. Defaults to h if None.

    Returns:
        DataFrame containing actual values and predictions for each cutoff point
    """
    max_lag = max(model.ts.lags)

    first_prediction = (
        df.select(pl.col(ts_col)+ pl.duration(days=max_lag))
        .item(0,0)
    )
    last_series = df.select(pl.col(ts_col).max()).item(0,0)
    
    step_size = step_size if step_size else h
    step_size = f"{step_size}{model.freq[-1]}"


    cutoffs = (
        pl.DataFrame(
        pl.datetime_range(
            start=first_prediction,
            end=last_series,
            interval=step_size,
            eager=True,
            closed='left'
            )
        )
        .select(
            pl.col('literal').alias('cutoff')
        )
        .to_series()
        .to_list()
    )


    preds = []
    for cutoff_i in cutoffs:
        pred_i = (
            model
            .predict(h=h, new_df=df.filter(pl.col(ts_col).le(cutoff_i)))
            .with_columns(
                cutoff=pl.lit(cutoff_i)
            )
        )
        preds.append(pred_i)
        
    return (
        df
        .join(pl.concat(preds), on=[unique_id, ts_col], how='inner')
        .select([unique_id, ts_col, 'cutoff', y,] + list(model.models.keys()))
    )
    
def evaluate_prediction(df: pl.DataFrame):
    return (
        df 
        .with_columns(
            error=pl.col('y_pred') - pl.col('y'),
            horizon = pl.col('ds').sub(pl.col('cutoff'))
        )
        .group_by(['unique_id', 'horizon'])
        .agg(
            bias=pl.col('error').mean(),
            mae = pl.col('error').abs().mean(),
            mae_per = pl.col('error').abs().sum().truediv(pl.col('y').sum())
        )
        .with_columns(
            score = pl.col('mae').add(pl.col('bias').abs())
        )
        .group_by('horizon')
        .agg(
            pl.col('bias').mean()
            , pl.col('mae').mean()
            , pl.col('score').mean()
        )
        .to_dicts()
    )
    
def process_result_into_keyval(eval_result:dict):
    """ 
    Utility function to transform the evaluation metrics
    by adding the horizon as part of the key value
    """
    horizon = eval_result['horizon'].days
    metrics_no_horizon = [x for x in eval_result.keys() if x != 'horizon']
    return {f"{x}_{horizon}":eval_result.get(x) for x in metrics_no_horizon}
      
def train_model(
    repo: NYCTaxiRepository,
    train_data_from:datetime,
    train_data_to:datetime,
    test_data_from:datetime,
    pickup_locations: list[int],
    max_horizon:int,
    cross_validation_split_frequency:str
    ):
    
    """Train the model and save it to disk

    Args:
        plot_predictions (bool, optional): If True, plot the predictions. Defaults to False.
    """
    logger.info("Start Training")
    logger.info("Load training data from database from %s to %s", train_data_from, train_data_to)

    
    # Loading
    df = repo.fetch_pickup_data(
        from_date=train_data_from,
        to_date=train_data_to,
        pickup_locations=pickup_locations
    )
    
    # TODO | 2025-03-02 | This should be part of the model or a function "feature engineering"
    df = (
        df
        .sort(by='pickup_datetime_hour')
        .group_by_dynamic('pickup_datetime_hour', every='1d', group_by='pickup_location_id')
        .agg(
            pl.col('num_pickup').sum()
        )
        .rename(
            {
                "pickup_location_id":'unique_id',
                "pickup_datetime_hour":"ds",
                "num_pickup":"y"
            }
        )
    )
    
    folds = split_train_test(df, test_from=test_data_from, every=cross_validation_split_frequency)
    
    logger.info("Fit the model")
    model = build_model()
    
    # Fit
    for i, (train, test) in  enumerate(folds, start=1):
                
        logger.info('training fold %s', i)
        logger.info('train shape %s', train.shape[0])
        logger.info('test shape %s', test.shape[0])

        
        model.fit(train)
        
        test_result = rolling_window_forecast(
            model=model,
            h=max_horizon,
            df=test,
        )

        logger.info('Evaluating predictions')
   
        fold_results = evaluate_prediction(test_result)
        # eval_transform = [ process_result_into_keyval(x) for x in eval]
        
        logger.info(fold_results)
        
    
    # persist
    # joblib.dump(model, MODEL_DIR / "baseline_model.pkl")
    
    logger.info("Training finished")

# if __name__ == "__main__":
#     from src.adapters.base import initialize_repository
#     repo = initialize_repository()
#     train_model(repo)

