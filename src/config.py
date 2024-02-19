from datetime import datetime, timedelta

TODAY_IS = datetime(2023,6,1)

# variables are in days
TEST_DATA_SIZE = 90
TEST_DATA_FROM = TODAY_IS - timedelta(days=TEST_DATA_SIZE)
TRAIN_DATA_FROM =  datetime(2022,1,1)


# model config
LOCATIONS = [43]
LAGS = [1,7,14,28]

TARGET = "num_pickup"
TS_INDEX = "pickup_datetime_hour"