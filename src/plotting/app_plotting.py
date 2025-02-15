import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from processing.data_parsing import parse_xml
import base64

##### APP PLOTTING ######


# MAP PLOT
def plot_station_locations():
    df = pd.read_csv("./results/xml_info.csv", index_col=0)

    lats = np.array(df["lat"]).flatten()
    lons = np.array(df["lon"]).flatten()

    fig = go.Figure(
        go.Scattermap(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=go.scattermap.Marker(size=8),
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=[None],
            lon=[None],
            mode="markers",
            marker=go.scattermap.Marker(size=10),
            marker_color="red",
        )
    )

    fig.update_layout(
        hovermode="closest",
        map=dict(center=go.layout.map.Center(lat=60.71, lon=-135.08), zoom=9),
    )

    return fig


# TIMESERIES PLOT
def plot_timeseries(xml_df, lat, lon, date):
    selected_station = xml_df[
        (xml_df["lat"] == lat)
        & (xml_df["lon"] == lon)
        & (xml_df["end_date"] >= date)
        & (xml_df["start_date"] <= date)
    ]

    serial_number = selected_station["serial"].values[0]
    start_date = selected_station["start_date"].values[0].split("T")[0]

    # serial_folder = "./results/timeseries/clipped/" + str(serial_number) + "/"
    serial_folder = "./results/figures/timeseries/" + str(serial_number) + "/"
    files = os.listdir(serial_folder)
    for f in files:
        if start_date in f:
            # df_timeseries = pd.read_parquet(serial_folder + f, engine="pyarrow")
            path = serial_folder + f
            break

    """
    df_timeseries.index = pd.to_datetime(
        df_timeseries.index, format="ISO8601"  # format="%Y-%m-%d %H:%M:%S.%f%z"
    )
    timeseries_fig = px.line(
        df_timeseries,
        y=["magnitude"],
    )
    """

    encoded_image = base64.b64encode(open(path, "rb").read())
    src = "data:image/png;base64,{}".format(encoded_image.decode())

    return src


# ELLIPTICITY PLOT
def plot_ellipticity(xml_df, lat, lon, date):
    selected_station = xml_df[
        (xml_df["lat"] == lat)
        & (xml_df["lon"] == lon)
        & (xml_df["end_date"] >= date)
        & (xml_df["start_date"] <= date)
    ]

    serial_number = selected_station["serial"].values[0]
    start_date = selected_station["start_date"].values[0].split("T")[0]

    serial_folder = "./results/figures/ellipticity/" + str(serial_number) + "/"
    files = os.listdir(serial_folder)
    for f in files:
        if start_date in f:
            path = serial_folder + f
            break

    encoded_image = base64.b64encode(open(path, "rb").read())
    src = "data:image/png;base64,{}".format(encoded_image.decode())

    return src


# TEMPERATURE PLOT
def plot_temperature(station):
    # *** parse time zone info ***
    df = pd.read_csv("./data/temperature/" + station + ".csv")

    df["date_time_local"] = pd.to_datetime(
        df["date_time_local"], format="%Y-%m-%d %H:%M:%S MST"
    )
    df["date_time_local"] = (
        df["date_time_local"].dt.tz_localize("UTC").dt.tz_convert("US/Mountain")
    )

    fig = px.line(df, "date_time_local", "avg_temp")

    return fig
