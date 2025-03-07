from datetime import datetime, timedelta


# training default options
TRAIN_DATA_FROM =  datetime(2022,1,1)
TRAIN_DATA_TO = datetime.today()
MAX_HORIZON = 7 # we'll make predictions 7 days in advance
TEST_DATA_FROM = TRAIN_DATA_TO - timedelta(days=365) # we leave one year open
CROSS_VALIDATION_FREQUENCY = '3mo' # to create the time-based split, we create folds of 3 month each
PICKUPS_LOCATION = [43]


# model config
MODEL_PARAMS = {
    "lags": [1,7,14,28]
}