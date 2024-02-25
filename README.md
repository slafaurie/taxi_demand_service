# Taxi demand predictor service

## Description
Build a service that predicts the demand of the next day.


## Epics
1. Create setup script to install dependencies and project structure.
- install dependencies
- creates folder structure
- create database
- create tables

2. Data Engineering: Create Pipeline that loads from source and writes to DWH
- Load data from source
- Write to DWH


3. Model Development:

- Feature Pipeline

- Training Pipeline

- Inference Pipeline


4. Deployment and Monitoring:



## TODO
- ~~Abstract daily aggregation function in pipeline.py into a database operation~~
- Generalize to other locations, not only central park.
- ~~build function to save the model~~
- ~~build train pipeline -> from loading data to saving the model~~
- build how to schedule the inference pipeline to mimic real operations (hint, i must change the today_is var)
- build model monitoring and frontend to visualize the predictions and the real values
- Add prediction intervals to the model
- replace model with proper timeseries model

