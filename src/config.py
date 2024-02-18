from datetime import datetime, timedelta

TODAY_IS = datetime(2023,6,1)

# variables are in days
TRAIN_DATA_SIZE = 365
TEST_DATA_SIZE = 90
TEST_DATA_FROM = TODAY_IS - timedelta(days=TEST_DATA_SIZE)
TRAIN_DATA_FROM = TEST_DATA_FROM - timedelta(days=TRAIN_DATA_SIZE) 


# model config
LAGS = [1,7,14,28]

TARGET = "num_pickup"
TS_INDEX = "pickup_datetime_hour"