from dash import dcc, html
from plotting.app_plotting import plot_map


# TIMESERIES PLOT
figure_layout = html.Div(
    [
        dcc.Graph(
            id="figure",
            style={
                "width": "60%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
        html.Div(
            [
                dcc.Graph(
                    id="map",
                    figure=plot_map()
                ),
                dcc.Markdown("Files"),
                dcc.Dropdown(id="dates", value="2024-06-10"),
                dcc.Checklist(
                    id="display_plots",
                    options=[
                        "timeseries",
                        "temperature",
                        "ellipticity",
                    ],
                    value=[
                        "timeseries"
                    ],
                ),
            ],
            style={"width": "35%", "display": "inline-block"},
        ),
    ],
)


layout = html.Div(
    [
        dcc.Markdown("## HVSR processing"),
        figure_layout,
        dcc.Store(id="station"),
    ]
)
