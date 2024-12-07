import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from processing.data_parsing import parse_xml

##### APP PLOTTING ######


def plot_map():
    stations = parse_xml()
    
    sites, lats, lons = [], [], []
    for s, c in stations.items():
        for coords in c:
            sites += [s]
            lats += [coords["lat"]]
            lons += [coords["lon"]]

    fig = go.Figure(
        go.Scattermap(
            lat=lats,
            lon=lons,
            mode="markers",
            text=sites,
        )
    )
    fig.update_layout(
        map=dict(center=dict(lat=60.74, lon=-135.08), zoom=10),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    #fig.add_trace(
    #    go.Scattermap(stations[station]["lat"], stations[station]["lon"], mode="markers", marker_color="red")
    #)
    # fig.write_html(path)
    return fig


# TEMPERATURE PLOT


def plot_temperature(station):
    # *** parse time zone info ***
    df = pd.read_csv("./data/temperature/" + station + ".csv")
    print(df)

    df["date_time_local"] = pd.to_datetime(
        df["date_time_local"], format="%Y-%m-%d %H:%M:%S MST"
    )
    df["date_time_local"] = df["date_time_local"].dt.tz_localize("UTC").dt.tz_convert("US/Mountain")

    fig = px.line(df, "date_time_local", "avg_temp")

    return fig


