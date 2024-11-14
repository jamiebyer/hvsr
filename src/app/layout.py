from dash import dcc, html
from plotting import plot_map

# STATION SELECTION WITH MAP
map_layout = html.Div(
    [
        dcc.Graph(
            id="map",
            figure=plot_map(),
            style={
                "width": "45%",
                "display": "inline-block",
                "vertical-align": "middle",
            },
        ),
        dcc.Checklist(
            id="display_plots",
            options=[
                "timeseries",
                "filter",
                "temperature",
                "raydec",
                "stacking",
            ],
            value=[
                "timeseries",
                # "filter",
                # "temperature",
                "raydec",
                # "stacking",
            ],
            style={
                "width": "30%",
                "display": "inline-block",
                "vertical-align": "middle",
                "padding-left": "10%",
            },
        ),
    ],
)


# TIMESERIES PLOT
timeseries_layout = html.Div(
    id="timeseries_div",
    children=[
        dcc.Graph(
            id="timeseries_figure",
            style={
                "width": "60%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
        html.Div(
            [
                # DATES THAT HAVE A TIMESERIES FILE
                dcc.Markdown("Timeseries files"),
                dcc.Dropdown(id="timeseries_dates", value="2024-06-10"),
                # fetch time range of time series
                # view list of saved files
                # SPIKE REMOVAL
                html.Div(
                    [
                        dcc.Markdown(
                            "max amplitude",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="max_amplitude",
                            type="number",
                            value=0.2,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "diff from mean",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="diff_from_mean",
                            type="number",
                            value=3,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ]
                ),
                # INPUT PARAMS FOR RAYDEC
                html.Div(
                    [
                        dcc.Markdown(
                            "len_wind",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="len_wind",
                            type="number",
                            value=30,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "f_min",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="f_min",
                            type="number",
                            value=0.8,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "f_max",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="f_max",
                            type="number",
                            value=40,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "f_steps",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="f_steps",
                            type="number",
                            value=100,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "cycles",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="cycles",
                            type="number",
                            value=10,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "df_par",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="df_par",
                            type="number",
                            value=0.1,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Button("Save", id="save_raydec", n_clicks=0),
            ],
            style={
                "width": "35%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
    ],
)

### TEMPERATURE PLOT ^ ^ ###
temperature_layout = html.Div(
    id="temperature_div",
    children=[
        dcc.Graph(
            id="temperature_figure",
            style={
                "width": "60%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
    ],
)

### FILTER PLOT ####
filter_layout = html.Div(id="filter_div")


### READ SAVED RAYDEC AND PLOT ###
raydec_layout = html.Div(
    id="raydec_div",
    children=[
        dcc.Graph(
            id="raydec_figure",
            style={
                "width": "60%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
        html.Div(
            [
                # dates that have a timeseries file
                dcc.Markdown("Raydec files"),
                dcc.Dropdown(id="raydec_dates", value="2024-06-10"),
                # date range to stack
                dcc.DatePickerRange(
                    id="stacking_dates",
                    # min_date_allowed=date(1995, 8, 5),
                    # max_date_allowed=date(2017, 9, 19),
                    # initial_visible_month=date(2017, 8, 5),
                    # end_date=date(2017, 8, 25)
                ),
                # PARAMS TO FILTER STACKING
                html.Div(
                    [
                        dcc.Markdown(
                            "len_wind",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="len_wind_filter",
                            type="number",
                            value=30,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="len_wind_check",
                            options=[{"label": "", "value": True}],
                            style={"width": "5%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "f_min",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="f_min_filter",
                            type="number",
                            value=0.8,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="f_min_check",
                            options=[{"label": "", "value": True}],
                            style={"width": "5%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "f_max",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="f_max_filter",
                            type="number",
                            value=40,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="f_max_check",
                            options=[{"label": "", "value": True}],
                            style={"width": "5%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "f_steps",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="f_steps_filter",
                            type="number",
                            value=100,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="f_steps_check",
                            options=[{"label": "", "value": True}],
                            style={"width": "5%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "cycles",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="cycles_filter",
                            type="number",
                            value=10,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="cycles_check",
                            options=[{"label": "", "value": True}],
                            style={"width": "5%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "df_par",
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id="df_par_filter",
                            type="number",
                            value=0.1,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="df_par_check",
                            options=[{"label": "", "value": True}],
                            style={"width": "5%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Button("Stack station results", id="stack_station", n_clicks=0),
            ],
            style={
                "width": "35%",
                "display": "inline-block",
                "vertical-align": "top",
            },
            # params to filter by for stacking, with checkboxes to enable
        ),
    ],
)

### STACKING PLOT ###
stacking_layout = html.Div(
    id="stacking_div",
    children=[
        dcc.Graph(
            id="stacking_figure",
            style={
                "width": "60%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
    ],
)


layout = html.Div(
    [
        dcc.Markdown("## HVSR processing"),
        map_layout,
        timeseries_layout,
        temperature_layout,
        filter_layout,
        raydec_layout,
        stacking_layout,
        dcc.Store(id="station"),
    ]
)
