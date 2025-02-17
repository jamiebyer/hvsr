import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from utils.utils import make_output_folder
import xarray as xr


# TIMESERIES PLOT


def plot_timeseries(station, date, in_path="./results/timeseries/"):
    dir_in = in_path + str(station) + "/" + date + ".parquet"
    dir_out = "./results/figures/timeseries/" + str(station) + "/" + date + ".png"

    make_output_folder("./results/figures/timeseries/")
    make_output_folder("./results/figures/timeseries/" + str(station) + "/")

    df_timeseries = pd.read_parquet(dir_in, engine="pyarrow")

    df_timeseries.index = pd.to_datetime(
        df_timeseries.index  # , format="ISO8601"  # format="%Y-%m-%d %H:%M:%S.%f%z"
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

    df_timeseries["magnitude"] = magnitude

    # change to just amplitude...?
    timeseries_fig = px.line(
        df_timeseries,
        # x=df_keep.index,
        y=["magnitude"],
        # color_discrete_sequence=["rgba(100, 100, 100, 0.1)"],
        color="spikes",
    )

    timeseries_fig.write_image(dir_out)

    # return timeseries_fig


def save_all_timeseries_plot():
    dir = "./results/timeseries/clipped/"
    for station in os.listdir(dir):
        for file in os.listdir(dir + "/" + station + "/"):
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
