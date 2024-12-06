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


def plot_timeseries_app(station, date, max_amplitude):
    # only do general layout once
    # any timeseries processing for plot has to also be done before running raydec
    # *** only read in the csv once ***

    dir_in = "./results/timeseries/" + str(station) + "/" + date
    df_timeseries = pd.read_parquet(dir_in, engine="pyarrow")

    df_timeseries.index = pd.to_datetime(
        df_timeseries.index, format="ISO8601"  # format="%Y-%m-%d %H:%M:%S.%f%z"
    )
    # df_timeseries.set_index(pd.to_datetime(df_timeseries["dates"], format="mixed"))

    # print(df_timeseries)

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

    timeseries_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=2, label="2h", step="hour", stepmode="backward"),
                    # dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=5, label="5h", step="hour", stepmode="backward"),
                    # dict(step="all")
                ]
            )
        ),
    )
    # """
    timeseries_fig.update_layout(
        yaxis_range=[np.min(df_timeseries["vert"]), np.max(df_timeseries["vert"])],
        # yaxis_range=[-0.3, 0.3],
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return timeseries_fig


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


if __name__ == "__main__":
    """
    run from terminal
    """

    plot_from_xml()
