from os import environ

import dash
from dash import no_update
from dash.dependencies import Input, Output, State
from flask import Flask
import plotly.graph_objects as go
import os
from layout import layout
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime

from plotting import plot_timeseries, plot_raydec, plot_temperature
from ellipticity_processing import write_raydec_df, stack_station_windows
from utils import make_output_folder

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

server = Flask(__name__)
app = dash.Dash(
    server=server,
    url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
    external_stylesheets=external_stylesheets,
)

app.layout = layout


# APP WIDGET CALLBACKS


@app.callback(
    Output(component_id="station", component_property="data"),
    Input(component_id="map", component_property="clickData"),
)
def set_station_value(click_data):
    if click_data is None:
        return None
    station = click_data["points"][0]["text"]
    return station


@app.callback(
    Output(component_id="timeseries_dates", component_property="options"),
    Output(component_id="raydec_dates", component_property="options"),
    Input(component_id="station", component_property="data"),
    prevent_initial_call=True,
)
def update_date_options(station):
    """
    update options in date dropdowns based on
    """
    if station is None:
        return [], []

    path = "./results/timeseries/" + str(station) + "/"
    read_timeseries_dates = [p.replace(".csv", "") for p in os.listdir(path)]

    path = "./results/raydec/" + str(station) + "/"
    read_raydec_dates = [p.replace(".csv", "") for p in os.listdir(path)]

    return read_timeseries_dates, read_raydec_dates


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
    Output(component_id="timeseries_div", component_property="style"),
    Output(component_id="filter_div", component_property="style"),
    Output(component_id="temperature_div", component_property="style"),
    Output(component_id="raydec_div", component_property="style"),
    Output(component_id="stacking_div", component_property="style"),
    Input(component_id="display_plots", component_property="value"),
)
def update_display_plots(display_plots):
    plots = [
        "timeseries",
        "filter",
        "temperature",
        "raydec",
        "stacking",
    ]

    return [
        {"display": "block"} if p in display_plots else {"display": "none"}
        for p in plots
    ]


# TIMESERIES PLOT


@app.callback(
    Output(component_id="timeseries_figure", component_property="figure"),
    Input(component_id="station", component_property="data"),
    Input(component_id="timeseries_dates", component_property="value"),
    Input(component_id="max_amplitude", component_property="value"),
)
def update_timeseries_figure(station, date, max_amplitude):
    if station is None or date is None:
        return go.Figure()

    timeseries_fig = plot_timeseries(station, date, max_amplitude)

    return timeseries_fig


# TEMPERATURE PLOT


@app.callback(
    Output(component_id="temperature_figure", component_property="figure"),
    Input(component_id="timeseries_figure", component_property="figure"),
)
def update_temperature_figure(timeseries_figure):
    if timeseries_figure is None:
        return go.Figure()

    temperature_fig = plot_temperature()
    if timeseries_figure is not None:
        temperature_fig.update_layout(
            xaxis_range=timeseries_figure["layout"]["xaxis"]["range"]
        )

    return temperature_fig


# RAYDEC PLOT


@app.callback(
    Output(component_id="raydec_figure", component_property="figure"),
    Input(component_id="station", component_property="data"),
    Input(component_id="raydec_dates", component_property="value"),
    Input(component_id="diff_from_mean", component_property="value"),
)
def update_raydec_figure(
    station, date, scale_factor, json_path="./results/raydec/raydec_info.json"
):
    if station is None or date is None:
        return go.Figure()

    file_name = str(station) + "/" + str(date)
    path_raydec = "./results/raydec/" + file_name + ".csv"
    df_raydec = pd.read_csv(path_raydec, index_col=0)
    with open(json_path, "r") as file:
        raydec_info = json.load(file)["raydec_info"]

    fig_dict = {}
    for i in range(len(raydec_info)):
        if raydec_info[i]["name"] == file_name:
            fig_dict = raydec_info[i]
            break
    raydec_fig = plot_raydec(
        df_raydec, station, date.rsplit("-", 1)[0], fig_dict, scale_factor
    )

    return raydec_fig


def write_json(raydec_info, filename="./results/raydec/raydec_info.json"):
    with open(filename, "r+") as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        if (len(file_data["raydec_info"])) == 0:
            file_data["raydec_info"].append(raydec_info)
        else:
            for i in range(len(file_data["raydec_info"])):
                if raydec_info["name"] == file_data["raydec_info"][i]["name"]:
                    file_data["raydec_info"][i] = raydec_info
                elif i == len(file_data["raydec_info"]) - 1:
                    file_data["raydec_info"].append(raydec_info)
        # Sets file's current position at offset.
        file.seek(0)
        json.dump(file_data, file, indent=4)


@app.callback(
    Output(component_id="raydec_dates", component_property="value"),
    State(component_id="station", component_property="data"),
    State(component_id="timeseries_dates", component_property="value"),
    State(component_id="f_min", component_property="value"),
    State(component_id="f_max", component_property="value"),
    State(component_id="f_steps", component_property="value"),
    State(component_id="cycles", component_property="value"),
    State(component_id="df_par", component_property="value"),
    Input(component_id="save_raydec", component_property="n_clicks"),
    prevent_initial_call=True,
)
def write_raydec_df(station, date, f_min, f_max, f_steps, cycles, df_par, _):
    if station is None or date is None:
        return go.Figure()

    raydec_df = write_raydec_df(
        station,
        date,
        f_min,
        f_max,
        f_steps,
        cycles,
        df_par,
    )

    make_output_folder("./results/raydec/")
    make_output_folder("./results/raydec/" + str(station) + "/")
    # write station df to csv
    raydec_df.to_csv("./results/raydec/" + str(station) + "/" + date + ".csv")

    # python object to be appended
    raydec_info = {
        "name": str(station) + "/" + date,
        "f_min": f_min,
        "f_max": f_max,
        "f_steps": f_steps,
        "cycles": cycles,
        "df_par": df_par,
        "n_wind": raydec_df.shape,
    }

    write_json(raydec_info)

    return date


###### STACKING ######

# update date limits from file names


# grey out box with check
@app.callback(
    Output(component_id="len_wind_filter", component_property="disabled"),
    Output(component_id="f_min_filter", component_property="disabled"),
    Output(component_id="f_max_filter", component_property="disabled"),
    Output(component_id="f_steps_filter", component_property="disabled"),
    Output(component_id="cycles_filter", component_property="disabled"),
    Output(component_id="df_par_filter", component_property="disabled"),
    Input(component_id="len_wind_check", component_property="value"),
    Input(component_id="f_min_check", component_property="value"),
    Input(component_id="f_max_check", component_property="value"),
    Input(component_id="f_steps_check", component_property="value"),
    Input(component_id="cycles_check", component_property="value"),
    Input(component_id="df_par_check", component_property="value"),
)
def update_output(
    len_wind_check,
    f_min_check,
    f_max_check,
    f_steps_check,
    cycles_check,
    df_par_check,
):

    output = []
    for b in [
        len_wind_check,
        f_min_check,
        f_max_check,
        f_steps_check,
        cycles_check,
        df_par_check,
    ]:
        if b is None or len(b) == 0:
            output.append(False)
        else:
            output.append(True)
    return output


@app.callback(
    Output(component_id="stacking_figure", component_property="figure"),
    State(component_id="station", component_property="data"),
    State(component_id="stacking_dates", component_property="start_date"),
    State(component_id="stacking_dates", component_property="end_date"),
    State(component_id="f_min_check", component_property="value"),
    State(component_id="f_min_filter", component_property="value"),
    State(component_id="f_max_check", component_property="value"),
    State(component_id="f_max_filter", component_property="value"),
    State(component_id="f_steps_check", component_property="value"),
    State(component_id="f_steps_filter", component_property="value"),
    State(component_id="cycles_check", component_property="value"),
    State(component_id="cycles_filter", component_property="value"),
    State(component_id="df_par_check", component_property="value"),
    State(component_id="df_par_filter", component_property="value"),
    Input(component_id="stack_station", component_property="n_clicks"),
)
def stack_station(
    station,
    start_date,
    end_date,
    f_min_check,
    f_min_filter,
    f_max_check,
    f_max_filter,
    f_steps_check,
    f_steps_filter,
    cycles_check,
    cycles_filter,
    df_par_check,
    df_par_filter,
    _,
):
    if start_date is None or end_date is None:
        return go.Figure()

    props = ["f_min", "f_max", "f_steps", "cycles", "df_par"]

    raydec_properties = {}
    for p in props:
        if locals().get_attr(props + "_check"):
            raydec_properties[p] = locals().get_attr(props + "_filter")

    stack_station_windows(station, [start_date, end_date], raydec_properties)

    return go.Figure


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
