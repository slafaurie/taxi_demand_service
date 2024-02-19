
import joblib
from datetime import datetime, timedelta
import polars as pl

from src.config import LAGS, LOCATIONS
from src.paths import MODEL_DIR
from src.dwh import run_database_operation


def make_prediction(reference_date:datetime, pickup_locations=LOCATIONS) -> pl.DataFrame:
    """Make a prediction for the given reference date and pickup locations

    Args:
        reference_date (datetime): The date for which we want to make the prediction
        pickup_locations (List[str], optional): The list of pickup locations for which we want to make the prediction. Defaults to LOCATIONS.

    Returns:
        pl.DataFrame: A DataFrame containing the predictions for the given reference date and pickup locations
    """

    # + 1 guarantees we have enough data for the prediction of the reference date
    df = run_database_operation(
        operation="fetch_pickup_data",
        from_date= reference_date - timedelta(days=max(LAGS)+1), 
        to_date=reference_date,
        pickup_locations=pickup_locations
    )
    
    model = joblib.load(MODEL_DIR / "baseline_model.pkl")
    
    return model.predict(df)
