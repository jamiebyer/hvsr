from os import environ
import dash
from dash import no_update
from dash.dependencies import Input, Output, State
from flask import Flask
import plotly.graph_objects as go
import os
from app.layout import layout
import pandas as pd
import plotly.graph_objects as go

import numpy as np
from plotting.app_plotting import (
    plot_station_locations,
    plot_temperature,
    plot_timeseries,
    plot_ellipticity,
)


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

server = Flask(__name__)
app = dash.Dash(
    server=server,
    url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
    external_stylesheets=external_stylesheets,
)

app.layout = layout

xml_df = pd.read_csv("./results/xml_info.csv", index_col=0)


# APP WIDGET CALLBACKS


@app.callback(
    Output(component_id="map", component_property="figure"),
    Input(component_id="map", component_property="clickData"),
    Input(component_id="map", component_property="figure"),
)
def update_map_figure(click_data, map_fig):
    if click_data is None:
        return no_update

    point_lat = click_data["points"][0]["lat"]
    point_lon = click_data["points"][0]["lon"]

    map_fig["data"][1].update({"lon": [point_lon], "lat": [point_lat]})

    return map_fig


@app.callback(
    Output(component_id="dates", component_property="min_date_allowed"),
    Output(component_id="dates", component_property="max_date_allowed"),
    Input(component_id="map", component_property="clickData"),
)
def update_date_options(click_data):
    if click_data is None:
        return None, None
    lat = click_data["points"][0]["lat"]
    lon = click_data["points"][0]["lon"]

    station = xml_df[(xml_df["lat"] == lat) & (xml_df["lon"] == lon)]
    start_date = station["start_date"].min()
    end_date = station["end_date"].max()

    return start_date, end_date


@app.callback(
    Output(component_id="timeseries_div", component_property="style"),
    Output(component_id="temperature_div", component_property="style"),
    Input(component_id="display_plots", component_property="value"),
)
def update_display_plots(display_plots):
    plots = ["timeseries", "temperature"]

    return [
        {"display": "block"} if p in display_plots else {"display": "none"}
        for p in plots
    ]


@app.callback(
    Output(component_id="timeseries_fig", component_property="src"),
    Output(component_id="temperature_fig", component_property="figure"),
    Output(component_id="ellipticity_fig", component_property="src"),
    Input(component_id="dates", component_property="date"),
    Input(component_id="display_plots", component_property="value"),
    Input(component_id="map", component_property="clickData"),
)
def update_figures(date, display_plots, click_data):
    if click_data is None or date is None:
        return None, go.Figure(), None

    timeseries_fig, temp_fig, ellipticity_fig = None, None, None

    lat = click_data["points"][0]["lat"]
    lon = click_data["points"][0]["lon"]

    if "timeseries" in display_plots:
        timeseries_fig = plot_timeseries(xml_df, lat, lon, date)
        # pass
    if "temperature" in display_plots:
        # temp_fig = plot_temperature()
        pass
    if "ellipticity" in display_plots:
        ellipticity_fig = plot_ellipticity(xml_df, lat, lon, date)

    return timeseries_fig, temp_fig, ellipticity_fig
