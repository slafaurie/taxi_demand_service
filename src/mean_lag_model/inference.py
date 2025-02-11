
import joblib
from datetime import datetime, timedelta
import polars as pl
import argparse

from src.mean_lag_model.config import ModelConfig
from src.common import MODEL_DIR
from src.dwh import run_database_operation


def make_prediction(reference_date:datetime, pickup_locations=ModelConfig.LOCATIONS) -> pl.DataFrame:
    """Make a prediction for the given reference date and pickup locations

    Args:
        reference_date (datetime): The date for which we want to make the prediction
        pickup_locations (List[str], optional): The list of pickup locations for which we want to make the prediction. Defaults to LOCATIONS.

    Returns:
        pl.DataFrame: A DataFrame containing the predictions for the given reference date and pickup locations
    """

    # + 1 guarantees we have enough data for the prediction of the reference date
    # database operation is not inclusive of the to_date
    df = run_database_operation(
        operation="fetch_pickup_data",
        from_date= reference_date - timedelta(days=max(ModelConfig.LAGS)), 
        to_date=reference_date + timedelta(days=1),
        pickup_locations=pickup_locations
    )
    
    model = joblib.load(MODEL_DIR / "baseline_model.pkl")
    
    return model.predict(df)


 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for a given date")
    parser.add_argument("--date", type=str, required=True, help="Reference date for prediction in YYYY-MM-DD format")
    args = parser.parse_args()
    reference_date = datetime.strptime(args.date, "%Y-%m-%d")
    predictions = make_prediction(reference_date)
    print(predictions)
