from datetime import datetime 
import joblib
import polars as pl 
from sklearn.metrics import mean_absolute_error

from src.model.config import  TEST_DATA_FROM, TODAY_IS, TRAIN_DATA_FROM, ModelConfig
from src.common import MODEL_DIR
from src.model.pipeline import model
from src.dwh import run_database_operation
from src.logger import get_logger


logger = get_logger("train")

def split_train_test(
        df:pl.DataFrame,
        test_from: datetime,
        every:str,
        ts_col:str
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
    
    print('Numbers of folds:', len(fold_info))

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
    

# def train_model():
    
#     """Train the model and save it to disk

#     Args:
#         plot_predictions (bool, optional): If True, plot the predictions. Defaults to False.
#     """
    
#     logger.info("Start Training")
#     logger.info("Load training data from database from %s to %s for locations %s", TRAIN_DATA_FROM, TODAY_IS, ModelConfig.LOCATIONS)

    
#     # Loading
#     df = run_database_operation(
#         operation="fetch_pickup_data",
#         from_date=TRAIN_DATA_FROM,
#         to_date=TODAY_IS,
#         pickup_locations=ModelConfig.LOCATIONS
#     )
#     train, test = split_into_train_and_test(df)

#     logger.info("Fit the model")
#     # Fit
#     model.fit(train)
#     predictions = model.predict(train)
#     test_predictions = model.predict(test)
    
#     # Evaluation
#     train_with_predicitions = train.join(predictions, on=ModelConfig.TS_INDEX, how="inner")
#     test_with_predictions = test.join(test_predictions, on=ModelConfig.TS_INDEX, how="inner")
#     train_mae = mean_absolute_error(train_with_predicitions["num_pickup"], train_with_predicitions["prediction"])
#     test_mae = mean_absolute_error(test_with_predictions["num_pickup"], test_with_predictions["prediction"])
    
#     logger.info("Model Evaluation: Train MAE: %s, Test MAE: %s", train_mae, test_mae)
    
#     # persist
#     joblib.dump(model, MODEL_DIR / "baseline_model.pkl")
    
#     logger.info("Training finished")

# if __name__ == "__main__":
#     train_model()

