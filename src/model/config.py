from datetime import datetime, timedelta


# variables are in days
TEST_DATA_SIZE = 90
TEST_DATA_FROM = timedelta(days=TEST_DATA_SIZE)
TRAIN_DATA_FROM =  datetime(2022,1,1)


# model config

model_params = {
    "lags": [1,7,14,28],
    "target": "num_pickup",
    "ds": "pickup_datetime_hour",
    "unique_id": "pickup_location_id"
}