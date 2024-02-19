from datetime import datetime 
import joblib
import polars as pl 
from sklearn.metrics import mean_absolute_error

from src.config import LOCATIONS, TEST_DATA_FROM, TODAY_IS, TRAIN_DATA_FROM, TS_INDEX
from src.paths import MODEL_DIR
from src.pipeline import model

from src.dwh import run_database_operation
from src.plots import plot_ts


def split_into_train_and_test(df:pl.DataFrame, cutoff_date:datetime = TEST_DATA_FROM):
    return (
        df.filter(pl.col("pickup_datetime_hour") < cutoff_date)
        ,  df.filter(pl.col("pickup_datetime_hour") >= cutoff_date)
    )
    

def train_model(plot_predictions=False):
    
    """Train the model and save it to disk

    Args:
        plot_predictions (bool, optional): If True, plot the predictions. Defaults to False.
    """
    
    # Loading
    df = run_database_operation(
        operation="fetch_pickup_data",
        from_date=TRAIN_DATA_FROM,
        to_date=TODAY_IS,
        pickup_locations=LOCATIONS
    )
    train, test = split_into_train_and_test(df)

    # Fit
    model.fit(train)
    predictions = model.predict(train)
    test_predictions = model.predict(test)
    
    # Evaluation
    train_with_predicitions = train.join(predictions, on=TS_INDEX, how="inner")
    test_with_predictions = test.join(test_predictions, on=TS_INDEX, how="inner")
    train_mae = mean_absolute_error(train_with_predicitions["num_pickup"], train_with_predicitions["prediction"])
    test_mae = mean_absolute_error(test_with_predictions["num_pickup"], test_with_predictions["prediction"])
    
    if plot_predictions:
        plot_ts(train_with_predicitions, ["num_pickup", "prediction"])
        plot_ts(test_with_predictions, ["num_pickup", "prediction"])
    print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
    
    # persist
    joblib.dump(model, MODEL_DIR / "baseline_model.pkl")

if __name__ == "__main__":
    train_model()

