import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from processing.timeseries_processing import label_spikes
from utils.utils import make_output_folder
import xarray as xr


# TIMESERIES PLOT





def plot_timeseries(station, date, in_path="./results/timeseries/"):
    dir_in = in_path + str(station) + "/" + date
    df_timeseries = pd.read_parquet(dir_in, engine="pyarrow")

    df_timeseries.index = pd.to_datetime(
        df_timeseries.index, format="ISO8601"  # format="%Y-%m-%d %H:%M:%S.%f%z"
    )
    # df_timeseries.set_index(pd.to_datetime(df_timeseries["dates"], format="mixed"))

    # downsample for plotting
    # df_timeseries = df_timeseries.resample("5min")

    # should just add to saved df
    magnitude = np.sqrt(
        df_timeseries["vert"] ** 2
        + df_timeseries["north"] ** 2
        + df_timeseries["east"] ** 2
    )

    print(df_timeseries)
    df_timeseries["magnitude"] = magnitude
    print(
        np.min(magnitude[df_timeseries["spikes"] == 1]),
        np.max(magnitude[df_timeseries["spikes"] == 1]),
    )
    print(
        np.min(magnitude[df_timeseries["spikes"] == 0]),
        np.max(magnitude[df_timeseries["spikes"] == 0]),
    )

    # change to just amplitude...?
    timeseries_fig = px.line(
        df_timeseries,
        # x=df_keep.index,
        y=["magnitude"],
        # color_discrete_sequence=["rgba(100, 100, 100, 0.1)"],
        color="spikes",
    )

    return timeseries_fig


def save_all_timeseries_plot():
    dir = "./results/timeseries/"
    for station in [os.listdir(dir)[0]]:
        for file in os.listdir(dir + "/" + station + "/")[2:4]:
            timeseries_fig = plot_timeseries(station, file)

            output_dir = "./results/figures/timeseries"
            make_output_folder(output_dir + "/")
            make_output_folder(output_dir + "/" + str(station) + "/")
            timeseries_fig.write_image(output_dir + "/" + station + "/" + file + ".png")


def plot_raydec():
    return None


def plot_temperature():
    path = "./data/temperature/"
    # plot average from all stations in the background.
    df = pd.read_csv(
        path + station + ".csv",
        names=["millisecond_since_epoch", "yyyy-MM-ddThh:mm:ss", "data_value"],
        skiprows=[0],
    )
    fig = go.Figure(go.Scatter(x=df["yyyy-MM-ddThh:mm:ss"], y=df["data_value"]))

    return fig
