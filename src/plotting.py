import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import plotly.express as px
import plotly.graph_objects as go
from timeseries_processing import label_spikes
from ellipticity_processing import remove_window_outliers
import json
from utils import make_output_folder
import xarray as xr


###### PLOTTING STATION LOCATIONS ######


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


###### APP PLOTTING ######


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
    # *** only read in the csv once ***

    dir_in = "./results/timeseries/" + str(station) + "/" + date
    df_timeseries = pd.read_parquet(dir_in, engine="pyarrow")

    df_timeseries.index = pd.to_datetime(
        df_timeseries.index, format="ISO8601"  # format="%Y-%m-%d %H:%M:%S.%f%z"
    )
    # df_timeseries.set_index(pd.to_datetime(df_timeseries["dates"], format="mixed"))

    # df_timeseries = df_timeseries[df_timeseries.index.hour <= 2]

    # df_timeseries = df_timeseries.resample("1s").mean()

    # df_timeseries = remove_spikes(df_timeseries, max_amplitude)

    # downsample for plotting
    # df_timeseries = df_timeseries.resample("5min")

    # outliers = df_timeseries["outliers"]
    # df_timeseries = df_timeseries.drop("outliers", axis=1)

    # df_outliers = df_timeseries[outliers == 1]
    # df_keep = df_timeseries[outliers == 0]
    df_keep = df_timeseries

    stats = df_keep["vert"].describe()
    # print(stats)

    # change to just amplitude...?
    timeseries_fig = px.line(
        df_keep,
        # x=df_keep.index,
        y=["vert", "north", "east"],
        color_discrete_sequence=["rgba(100, 100, 100, 0.1)"],
    )

    """
    if df_outliers.shape[0] > 1:
        timeseries_fig.add_traces(
            list(
                px.scatter(
                    df_outliers,
                    x=df_outliers.index,
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
    """
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


def save_all_timeseries_plot():
    dir = "./results/timeseries/"
    for station in os.listdir(dir):
        print(station)
        for file in os.listdir(dir + "/" + station + "/"):
            file = file.replace(".csv", "")
            timeseries_fig = plot_timeseries(station, file, max_amplitude=0.1)
            output_dir = "./results/figures/timeseries"
            make_output_folder(output_dir + "/")
            make_output_folder(output_dir + "/" + str(station) + "/")
            timeseries_fig.write_image(output_dir + "/" + station + "/" + file + ".png")


# RAYDEC PLOT


def plot_raydec(da_raydec, station, date, fig_dict, scale_factor):
    # ideally the outliers are dropped and a new df is saved to read in,
    # but after a good threshold is set

    fig_dict = {
        "station": station,
        "date": date.rsplit("-", 1)[0],
        "fmin": da_raydec["fmin"].values,
        "fmax": da_raydec["fmax"].values,
        "fsteps": da_raydec["fsteps"].values,
        "nwind": len(da_raydec.coords["wind"].values),
        "cycles": da_raydec["cycles"].values,
        "dfpar": da_raydec["dfpar"].values,
    }

    # skip nans
    # plot raydec
    da_raydec = da_raydec.dropna(dim="freqs")
    # df_raydec.index = pd.to_numeric(df_raydec.index)

    # remove outlier windows
    # df_raydec = remove_window_outliers(df_raydec, scale_factor)

    # outliers = df_raydec.loc["outliers"]
    # df_raydec = df_raydec.drop("outliers")

    mean = da_raydec.mean(dim="wind")

    # df_raydec = df_raydec.drop("mean", axis=1)
    """outliers = outliers.drop("mean")

    df_outliers = df_raydec[outliers.index[outliers == 1]]
    df_keep = df_raydec[outliers.index[outliers == 0]]

    fig_dict["outliers"] = str(
        df_outliers.shape[1] / (df_outliers.shape[1] + df_keep.shape[1])
    )"""

    df_keep = da_raydec
    # stats = df_keep.describe()

    raydec_fig = px.line(
        df_keep,
        color_discrete_sequence=["rgba(100, 100, 100, 0.2)"],
        log_x=True,
    )

    raydec_fig.add_traces(
        # list(
        #    px.line(
        #        df_outliers,
        #        color_discrete_sequence=["rgba(255, 0, 0, 0.2)"],
        #        log_x=True,
        #    ).select_traces()
        # )
        # +
        list(
            px.line(
                pd.DataFrame(mean, df_keep.coords["freqs"].values),
                color_discrete_sequence=["rgba(0, 0, 0, 1)"],
                log_x=True,
            ).select_traces()
        )
    )

    groups = (
        ["mean"]
        # + df_outliers.shape[1] * ["outliers"]
        # ["mean", "max", "min"]
        + df_keep.shape[1] * ["keep"]
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

    text = ""
    for k, v in fig_dict.items():
        if k == "name":
            continue
        text += k + ": " + str(v) + "<br>"

    raydec_fig.add_annotation(
        x=1,
        y=9,
        text=text,
        showarrow=False,
    )

    # plot confidence interval above/below

    return raydec_fig


def plot_sensitivity_test():
    station = 24614
    # json_path = "./results/raydec/raydec_info.json"
    for date in os.listdir("./results/raydec/24614/"):
        date = date.removesuffix(".nc")
        file_name = str(station) + "/" + str(date)
        path_raydec = "./results/raydec/" + file_name + ".nc"

        da_raydec = xr.open_dataarray(path_raydec)

        raydec_fig = plot_raydec(
            da_raydec, 24614, date.rsplit("-", 1)[0], scale_factor=1
        )

        raydec_fig.write_image(
            "./results/figures/sensitivity_analysis/" + date + ".png"
        )


# STACKING PLOT


def plot_stacking():
    for station in os.listdir("./results/raydec/")[2:]:
        in_dir = "./results/raydec/stacked/" + station + ".nc"
        da_raydec = xr.open_dataarray(in_dir)
        da_raydec = da_raydec.dropna(dim="freqs")

        df = pd.DataFrame(
            da_raydec, columns=da_raydec.coords["date"], index=da_raydec.coords["freqs"]
        )

        raydec_fig = px.line(
            df,  # da_raydec,
            x=df.index,
            y=df.columns,
            # "date",
            # "freqs",
            # x=da_raydec.index,  # "freqs",  # da_raydec.coords["freqs"],
            # y=da_raydec.columns,  # "date",  # np.arange(len(da_raydec.coords["date"])),
            # line_group=da_raydec.coords["date"][0],
            color_discrete_sequence=["rgba(100, 100, 100, 0.2)"] * (len(df.columns) - 1)
            + ["rgba(0, 0, 0, 1)"],
            log_x=True,
        )
        raydec_fig.update_layout(
            yaxis_range=[0, 8],
            title=station,
        )

        raydec_fig.write_image("./results/figures/stacked/" + station + ".png")


# TEMPERATURE PLOT


def plot_temperature():
    # *** parse time zone info ***
    df = pd.read_csv("./data/weatherstats_whitehorse_hourly.csv")

    df["date_time_local"] = pd.to_datetime(
        df["date_time_local"], format="%Y-%m-%d %H:%M:%S MST"
    )
    df["date_time_local"] = (
        df["date_time_local"].dt.tz_localize("UTC").dt.tz_convert("US/Mountain")
    )
    """
    inds = (
        (df["date_time_local"].dt.year == date.year).values
        & (df["date_time_local"].dt.month == date.month).values
        & (df["date_time_local"].dt.day == date.day).values
    )"""
    # inds = df["date_time_local"]

    min_temp = df["min_air_temp_pst1hr"]
    max_temp = df["max_air_temp_pst1hr"]
    avg_temp = (min_temp + max_temp) / 2
    df["avg_temp"] = avg_temp

    fig = px.line(df, "date_time_local", "avg_temp")
    return fig


if __name__ == "__main__":
    """
    run from terminal
    """

    # plot_temperature()
    # save_all_timeseries_plot()
    # save_all_timeseries_plot()
    # plot_stacking()

    # fig = plot_map()
    fig = plot_from_xml()
    fig.show()
