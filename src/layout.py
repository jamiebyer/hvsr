import dash
from dash import dcc, html, no_update
from plotting import plot_map

default_vals = {}


layout = html.Div(
    [
        # STATION SELECTION WITH MAP
        html.Div(
            [
                dcc.Graph(id="map", figure=plot_map(), style={"width": "45%", "display": "inline-block"}),
            ],
        ),

        # TIMESERIES PLOT
        html.Div(
            [
                dcc.Graph(id="timeseries_figure", style={"width": "60%", "display": "inline-block", "vertical-align": "top"}),
                html.Div(
                    [
                        # dates that have a timeseries file
                        dcc.Markdown("Timeseries files"),
                        dcc.Dropdown(id="timeseries_dates", value="2024-06-10"),
                        # fetch time range of time series
                        # view list of saved files
                        html.Div(
                            [
                                dcc.Markdown("max amplitude", style={"width": "30%", "display": "inline-block"}),
                                dcc.Input(id="max_amplitude", type="number", value=0.2, style={"width": "30%", "display": "inline-block"}),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Markdown("diff from mean", style={"width": "30%", "display": "inline-block"}),
                                dcc.Input(id="diff_from_mean", type="number", value=3, style={"width": "30%", "display": "inline-block"}),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Markdown("len_wind", style={"width": "30%", "display": "inline-block"}),
                                dcc.Input(id="len_wind", type="number", value=30, style={"width": "30%", "display": "inline-block"}),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Markdown("f_min", style={"width": "30%", "display": "inline-block"}),
                                dcc.Input(id="f_min", type="number", value=0.8, style={"width": "30%", "display": "inline-block"}),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Markdown("f_max", style={"width": "30%", "display": "inline-block"}),
                                dcc.Input(id="f_max", type="number", value=40, style={"width": "30%", "display": "inline-block"}),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Markdown("f_steps", style={"width": "30%", "display": "inline-block"}),
                                dcc.Input(id="f_steps", type="number", value=100, style={"width": "30%", "display": "inline-block"}),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Markdown("cycles", style={"width": "30%", "display": "inline-block"}),
                                dcc.Input(id="cycles", type="number", value=10, style={"width": "30%", "display": "inline-block"}),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Markdown("df_par", style={"width": "30%", "display": "inline-block"}),
                                dcc.Input(id="df_par", type="number", value=0.1, style={"width": "30%", "display": "inline-block"}),
                            ]
                        ),
                        html.Button('Save', id='save_raydec', n_clicks=0),
                    ], style={"width": "35%", "display": "inline-block", "vertical-align": "top"}
                ),
            ]
        ),

        # READ RAYDEC DF AND PLOT
        html.Div(
            [
                dcc.Graph(id="raydec_figure", style={"width": "60%", "display": "inline-block", "vertical-align": "top"}),
                html.Div(
                    [
                        # dates that have a timeseries file
                        dcc.Markdown("Raydec files"),
                        dcc.Dropdown(id="raydec_dates", value="2024-06-10"),
                    ], style={"width": "35%", "display": "inline-block", "vertical-align": "top"}
                ),
            ]
        ),

        dcc.Store(id="station", data=24025),
    ]
)