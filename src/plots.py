from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def is_numeric(series: pd.Series) -> bool:
    """
    This function takes a pandas Series as an argument and returns True if the series data type is numeric (float or int).
    """
    return pd.api.types.is_numeric_dtype(series)

def plot_ts(
    ts_data: pd.DataFrame, 
    series:list[str] = ["num_pickup"], 
    locations: list[int] | None = None, 
    plot_from:datetime = None, 
    fill_between:list[str] = None,
    target: str | None = None
    ):
    """
    Plot time-series data
    """
    
    if isinstance(series, str):
        series = [series]
        
    
    ts_data_to_plot = ts_data[ts_data["pickup_location_id"].isin(locations)] if locations else ts_data
    
    if plot_from:
        ts_data_to_plot = ts_data_to_plot[ts_data_to_plot["pickup_datetime_hour"] > plot_from]


    
    fig = go.Figure()

    for serie in series:
        fig.add_trace(go.Scatter(
            x=ts_data_to_plot["pickup_datetime_hour"],
            y=ts_data_to_plot[serie],
            mode="lines",
            name=serie
            )
        )
    
    if target:
        fig.add_trace(
        go.Scatter(
            x = ts_data_to_plot["pickup_datetime_hour"]
            , y = ts_data_to_plot[target]
            , mode="markers"
            , name=target
        )
    )
        
    if fill_between is not None:
        fig.add_trace(
            go.Scatter(
                x = ts_data_to_plot["pickup_datetime_hour"]
                , y = ts_data_to_plot[fill_between[1]]
                , showlegend=False
                , line=dict(width=0)
                , name = "upper 95% CI"
            )
        )
        fig.add_trace(
            go.Scatter(
                x = ts_data_to_plot["pickup_datetime_hour"]
                , y = ts_data_to_plot[fill_between[0]]
                , fill="tonexty"
                , fillcolor='rgba(68, 68, 68, 0.3)'
                , showlegend=False
                , line=dict(width=0)
                , name = "lower 95% CI"
            )
        )


    fig.show()

def plot_relation_between_target_and_covariates(
        data: pd.DataFrame,
        target:str, 
        covariates:list[str], 
        ncols:int=3, 
        **kwargs
    ):
    """
    This function takes a pandas DataFrame, a target column name, a list of covariate column names, and an optional number of columns for the plot grid as arguments.
    It then plots scatterplots for numeric covariates and boxplots for categorical covariates, with the target column on the y-axis and each covariate on the x-axis.
    Additionally, for numeric covariates, it annotates the plot with the Pearson correlation value between the target and the covariate.
    The function returns nothing, but displays the plot.
    """
    nrows = int(np.ceil(len(covariates) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))
    axs = axs.flatten()
    for i, covariate in enumerate(covariates):
        if is_numeric(data[covariate]):
            sns.regplot(x=covariate, y=target, data=data, ax=axs[i], **kwargs)
            # Calculate and annotate Pearson correlation
            correlation = data[[covariate, target]].corr().iloc[0,1]
            axs[i].annotate(
                f'Pearson: {correlation:.2f}'
                , xy=(0.05, 0.95)
                , xycoords='axes fraction'
                , ha='left'
                , va='top'
                , fontsize=10
                , bbox=dict(boxstyle="round", alpha=0.5, color="w")
            )
        else:
            sns.boxplot(x=covariate, y=target, data=data, ax=axs[i], **kwargs)
        axs[i].set_title(f'{target} vs {covariate}')
    plt.tight_layout()
    plt.show()