from obspy import read, Stream
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
from scipy import fft
import plotly.express as px
import plotly.graph_objects as go
from utils import make_output_folder
from process_data import remove_outliers, remove_spikes

"""
TODO:
- set up linters.
- add unit information
"""

"""
RAW DATA AND STATION INFORMATION
"""


def plot_from_xml():
    path = "./data/FDSN_Information.xml"

    with open(path, "r") as f:
        file = f.read()

    soup = BeautifulSoup(file, "xml")
    lats = [float(lat.text) for lat in soup.find_all("Latitude")]
    lons = [float(lat.text) for lat in soup.find_all("Longitude")]

    plt.scatter(lons, lats)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def plot_3d_locations():
    df = pd.read_csv("./data/parsed_xml.csv")
    path = "./figures/station_3d_locations.html"

    fig = px.scatter_3d(
        df,
        x="Longitude",
        y="Latitude",
        z="Elevation",
        text="Site",
        # mode='markers'
    )
    fig.write_html(path)


"""
PLOTS OF RAYDEC PROCESSING
"""


def plot_timeseries_slice():
    station = 24025
    dir_in = "./timeseries/" + str(station) + "/"
    for file_name in os.listdir(dir_in):
        df = pd.read_csv(dir_in + file_name)
        plt.plot(df["times"], df["east"])
        plt.show()


"""
APP PLOTTING
"""


def plot_map():
    df = pd.read_csv("./data/parsed_xml.csv")
    # path = "./figures/station_map_locations.html"

    fig = go.Figure(
        go.Scattermap(
            lat=df["Latitude"],
            lon=df["Longitude"],
            mode="markers",
            text=df["Site"],
        )
    )
    fig.update_layout(
        map=dict(center=dict(lat=60.74, lon=-135.08), zoom=10),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    fig.add_trace(
        go.Scattermap(lat=[None], lon=[None], mode="markers", marker_color="red")
    )
    # fig.write_html(path)
    return fig


# TIMESERIES PLOT


def plot_timeseries(station, date, max_amplitude):
    # only do general layout once
    # any timeseries processing for plot has to also be done before running raydec

    path_timeseries = "./results/timeseries/" + str(station) + "/" + str(date) + ".csv"
    df_timeseries = pd.read_csv(path_timeseries, index_col=0)

    df_timeseries = remove_spikes(df_timeseries, max_amplitude)
    # print(df_timeseries)

    outliers = df_timeseries["outliers"]
    df_timeseries = df_timeseries.drop("outliers", axis=1)

    df_outliers = df_timeseries[outliers == 1]
    df_keep = df_timeseries[outliers == 0]

    stats = df_keep["vert"].describe()
    # print(stats)
    print(df_keep)

    # change to just amplitude...?
    timeseries_fig = px.line(
        df_keep,
        x="dates",
        y=["vert", "north", "east"],
        color_discrete_sequence=["rgba(100, 100, 100, 0.1)"],
    )

    if df_outliers.shape[0] > 1:
        timeseries_fig.add_traces(
            list(
                px.scatter(
                    df_outliers,
                    x="dates",
                    y="vert",
                    color_discrete_sequence=["rgba(255, 0, 0, 0.5)"],
                ).select_traces()
            )  # +
            # list(px.line(x=df_keep["dates"], y=stats["mean"], color_discrete_sequence=["rgba(0, 0, 255, 0.5)"]).select_traces()) +
            # list(px.line(x=df_keep["dates"], y=stats["min"], color_discrete_sequence=["rgba(0, 0, 255, 0.5)"]).select_traces()) +
            # list(px.line(x=df_keep["dates"], y=stats["max"], color_discrete_sequence=["rgba(0, 0, 255, 0.5)"]).select_traces())
        )

        # groups = ["vert", "north", "east", "outliers"]#, "mean", "max", "min"]
        for ind, trace in enumerate(timeseries_fig["data"][3:]):
            trace["legendgroup"] = "outliers"  # groups[ind]

    timeseries_fig.update_layout(
        yaxis_range=[np.min(df_timeseries["vert"]), np.max(df_timeseries["vert"])],
        # yaxis_range=[-0.3, 0.3],
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return timeseries_fig


# RAYDEC PLOT


def plot_raydec(df_raydec, station, date, fig_dict, scale_factor):
    # ideally the outliers are dropped and a new df is saved to read in,
    # but after a good threshold is set

    # skip nans
    # plot raydec
    df_raydec = df_raydec.T.dropna()
    df_raydec.index = pd.to_numeric(df_raydec.index)

    # remove outlier windows
    df_raydec = remove_outliers(df_raydec, scale_factor)

    outliers = df_raydec.loc["outliers"]
    df_raydec = df_raydec.drop("outliers")

    mean = df_raydec["mean"]
    df_raydec = df_raydec.drop("mean", axis=1)
    outliers = outliers.drop("mean")

    df_outliers = df_raydec[outliers.index[outliers == 1]]
    df_keep = df_raydec[outliers.index[outliers == 0]]

    fig_dict["outliers"] = str(
        df_outliers.shape[1] / (df_outliers.shape[1] + df_keep.shape[1])
    )

    stats = df_keep.T.describe().T

    raydec_fig = px.line(
        df_keep,
        color_discrete_sequence=["rgba(100, 100, 100, 0.2)"],
        log_x=True,
    )

    raydec_fig.add_traces(
        list(
            px.line(
                df_outliers,
                color_discrete_sequence=["rgba(255, 0, 0, 0.2)"],
                log_x=True,
            ).select_traces()
        )
        + list(
            px.line(
                stats["mean"], color_discrete_sequence=["rgba(0, 0, 0, 1)"], log_x=True
            ).select_traces()
        )
        + list(
            px.line(
                stats["min"],
                color_discrete_sequence=["rgba(0, 0, 255, 0.5)"],
                log_x=True,
            ).select_traces()
        )
        + list(
            px.line(
                stats["max"],
                color_discrete_sequence=["rgba(0, 0, 255, 0.5)"],
                log_x=True,
            ).select_traces()
        )
    )

    groups = (
        df_keep.shape[1] * ["keep"]
        + df_outliers.shape[1] * ["outliers"]
        + ["mean", "max", "min"]
    )
    for ind, trace in enumerate(raydec_fig["data"]):
        trace["legendgroup"] = groups[ind]

    raydec_fig.update_layout(
        title=str(station) + ": " + str(date),
        xaxis_title="frequency (Hz)",
        yaxis_title="ellipticity",
        yaxis_range=[0, 10],
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
    )

    raydec_fig.add_annotation(
        x=1,
        y=9,
        text=str(fig_dict),
        showarrow=False,
    )

    # plot confidence interval above/below

    return raydec_fig


# TEMPERATURE PLOT


def plot_temperature():
    # *** parse time zone info ***
    df = pd.read_csv("./data/weatherstats_whitehorse_hourly.csv")

    df["date_time_local"] = pd.to_datetime(df["date_time_local"])
    df["month"] = df["date_time_local"].dt.month
    df["year"] = df["date_time_local"].dt.year
    inds = (
        (df["month"] >= 6).values
        & (df["month"] <= 9).values
        & (df["year"] == 2024).values
    )

    min_temp = df["min_air_temp_pst1hr"]
    max_temp = df["max_air_temp_pst1hr"]
    avg_temp = (min_temp + max_temp) / 2
    df["avg_temp"] = avg_temp

    fig = px.scatter(df[inds], "date_time_local", "avg_temp")
    return fig


if __name__ == "__main__":
    """
    run from terminal
    """

    plot_temperature()
