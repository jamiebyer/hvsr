import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go


##### APP PLOTTING ######


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

    plot_from_xml()
