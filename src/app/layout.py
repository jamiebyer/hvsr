from dash import dcc, html
from plotting.app_plotting import plot_station_locations


figure_layout = html.Div(
    [
        html.Div(
            id="timeseries_div",
            children=[
                dcc.Graph(
                    id="timeseries_fig",
                ),
            ],
        ),
        html.Div(
            id="temperature_div",
            children=[
                dcc.Graph(
                    id="temperature_fig",
                ),
            ],
        ),
    ],
    style={
        "width": "60%",
        "display": "inline-block",
        "vertical-align": "top",
    },
)
widget_layout = html.Div(
    [
        dcc.DatePickerSingle(id="dates"),
        dcc.Checklist(
            id="display_plots",
            options=[
                "timeseries",
                "temperature",
            ],
            value=["timeseries"],
        ),
        dcc.Graph(id="map", figure=plot_station_locations()),
    ],
    style={"width": "35%", "display": "inline-block"},
)

layout = html.Div(
    [
        dcc.Markdown("## Whitehorse timeseries processing"),
        figure_layout,
        widget_layout,
        dcc.Store(id="station"),
    ]
)
