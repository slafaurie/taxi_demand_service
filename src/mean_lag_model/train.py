from datetime import datetime 
import joblib
import polars as pl 
from sklearn.metrics import mean_absolute_error

from src.mean_lag_model.config import  TEST_DATA_FROM, TODAY_IS, TRAIN_DATA_FROM, ModelConfig
from src.common import MODEL_DIR
from src.mean_lag_model.pipeline import model
from src.dwh import run_database_operation
from src.logger import get_logger


logger = get_logger("train")

def split_into_train_and_test(df:pl.DataFrame, cutoff_date:datetime = TEST_DATA_FROM):
    return (
        df.filter(pl.col("pickup_datetime_hour") < cutoff_date)
        ,  df.filter(pl.col("pickup_datetime_hour") >= cutoff_date)
    )
    

def train_model():
    
    """Train the model and save it to disk

    Args:
        plot_predictions (bool, optional): If True, plot the predictions. Defaults to False.
    """
    
    logger.info("Start Training")
    logger.info("Load training data from database from %s to %s for locations %s", TRAIN_DATA_FROM, TODAY_IS, ModelConfig.LOCATIONS)

    
    # Loading
    df = run_database_operation(
        operation="fetch_pickup_data",
        from_date=TRAIN_DATA_FROM,
        to_date=TODAY_IS,
        pickup_locations=ModelConfig.LOCATIONS
    )
    train, test = split_into_train_and_test(df)

    logger.info("Fit the model")
    # Fit
    model.fit(train)
    predictions = model.predict(train)
    test_predictions = model.predict(test)
    
    # Evaluation
    train_with_predicitions = train.join(predictions, on=ModelConfig.TS_INDEX, how="inner")
    test_with_predictions = test.join(test_predictions, on=ModelConfig.TS_INDEX, how="inner")
    train_mae = mean_absolute_error(train_with_predicitions["num_pickup"], train_with_predicitions["prediction"])
    test_mae = mean_absolute_error(test_with_predictions["num_pickup"], test_with_predictions["prediction"])
    
    logger.info("Model Evaluation: Train MAE: %s, Test MAE: %s", train_mae, test_mae)
    
    # persist
    joblib.dump(model, MODEL_DIR / "baseline_model.pkl")
    
    logger.info("Training finished")

if __name__ == "__main__":
    train_model()

