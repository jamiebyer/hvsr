import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import plotly.express as px
import plotly.graph_objects as go
from utils.utils import make_output_folder
import xarray as xr

# from scalebar import scale_bar
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cf

from matplotlib import colors
from matplotlib import patheffects
from math import floor
import plotly.graph_objects as go
from scipy.signal import find_peaks
import geopandas as gpd
import contextily as cx
from plotly.subplots import make_subplots
from pandas.api.types import is_string_dtype

# from matplotlib_scalebar.scalebar import ScaleBar
from processing.data_parsing import change_coords


def plot_globe():
    # province boundaries
    resol = "50m"
    resol2 = "110m"
    provinc_bodr = cf.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale=resol,
        facecolor="none",
        edgecolor="k",
    )

    lat_PR = 60.7216
    lon_PR = -135.0549
    lat_DC = 56.0027
    lon_DC = -119.7426

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(
        projection=ccrs.NearsidePerspective(
            satellite_height=2000000.0,
            central_longitude=lon_PR,
            central_latitude=lat_PR,
        ),
    )

    ax.coastlines(resolution=resol2)
    ax.add_feature(cf.BORDERS)
    ax.add_feature(
        provinc_bodr, linestyle="--", linewidth=0.6, edgecolor="k", zorder=10
    )

    extent = ax.get_extent()
    ax.plot(
        lon_PR,
        lat_PR,
        marker="*",
        color="red",
        markerfacecolor="red",
        markersize=8,
        linewidth=20,
        alpha=0.7,
        transform=ccrs.Geodetic(),
        # transform=ccrs.PlateCarree(),
    )

    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])

    # Save figure as SVG
    plt.savefig("./results/figures/globe_whitehorse.pdf")


def utm_from_lon(lon):
    """
    utm_from_lon - UTM zone for a longitude

    Not right for some polar regions (Norway, Svalbard, Antartica)

    :param float lon: longitude
    :return: UTM zone number
    :rtype: int
    """
    return floor((lon + 180) / 6) + 1


def scale_bar(
    ax, proj, length, location=(0.5, 0.05), linewidth=4, units="km", m_per_unit=1000
):
    """

    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit
    """
    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    # Projection in metres
    utm = ccrs.UTM(utm_from_lon((x0 + x1) / 2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit / 2, sbcx + length * m_per_unit / 2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=6, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(
        bar_xs,
        [sbcy, sbcy],
        transform=utm,
        color="k",
        linewidth=linewidth,
        path_effects=buffer,
    )
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=6, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(
        sbcx,
        sbcy + 1,
        str(length) + " " + units,
        fontsize=12,
        transform=utm,
        horizontalalignment="center",
        verticalalignment="bottom",
        path_effects=buffer,
        zorder=2,
    )
    left = x0 + (x1 - x0) * 0.05
    # Plot the N arrow
    t1 = ax.text(
        left,
        sbcy + 1,
        "\u25b2\nN",
        fontsize=10,
        weight="bold",
        transform=utm,
        horizontalalignment="center",
        verticalalignment="bottom",
        path_effects=buffer,
        zorder=2,
    )
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(
        bar_xs, [sbcy, sbcy], transform=utm, color="k", linewidth=linewidth, zorder=3
    )


###### PLOTTING STATION LOCATIONS ######


def get_phase_stations():
    """
    start dates
    P1: '2024-06-06' '2024-06-07' '2024-06-08'
    P2: '2024-06-22' '2024-06-23' '2024-06-24'
    P3: '2024-07-13' '2024-07-14' '2024-07-15'
    P4: '2024-07-28' '2024-07-29' '2024-08-15']

    P1: >= '2024-06-06'; < '2024-06-22'
    P2: >= '2024-06-22'; < '2024-07-13'
    P3: >= '2024-07-13'; < '2024-07-28'
    P4: >= '2024-07-28';


    end dates
    P1: '2024-06-20' '2024-06-21'
    P2: '2024-07-03'
    P3: '2024-07-26' '2024-07-27'
    P4: '2024-08-13' '2024-08-17' '2024-08-20' '2024-08-21' '2024-08-22'

    P1: >= '2024-06-20'; < '2024-07-03'
    P2: >= '2024-07-03'; < '2024-07-26'
    P3: >= '2024-07-26'; < '2024-08-13'
    P4: >= '2024-08-13';

    """
    # print unique start and end dates for stations
    stations_df = pd.read_csv("./data/df_mapping.csv")
    stations_df = stations_df.drop_duplicates("site")

    start_time = stations_df["Start_time (UTC)"]
    end_time = stations_df["End_time (UTC)"]

    P1 = stations_df[
        ((start_time >= "2024-06-06") & (start_time < "2024-06-22"))
        | ((end_time >= "2024-06-20") & (end_time < "2024-07-03"))
    ]
    P2 = stations_df[
        ((start_time >= "2024-06-22") & (start_time < "2024-07-13"))
        | ((end_time >= "2024-07-03") & (end_time < "2024-07-26"))
    ]
    P3 = stations_df[
        ((start_time >= "2024-07-13") & (start_time < "2024-07-28"))
        | ((end_time >= "2024-07-26") & (end_time < "2024-08-13"))
    ]
    P4 = stations_df[(start_time >= "2024-07-28") | (end_time >= "2024-08-13")]

    return P1, P2, P3, P4


def plot_station_phases():
    stations_df = pd.read_csv("./data/df_mapping.csv")
    stations_df = stations_df.drop_duplicates("site")
    P1, P2, P3, P4 = get_phase_stations()

    # Google image tiling
    request1 = cimgt.GoogleTiles(style="satellite")
    request2 = cimgt.GoogleTiles(
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}.jpg"
    )

    # Map projection
    proj = ccrs.AlbersEqualArea(
        central_longitude=-135.076167,
        central_latitude=60.729549,
        false_easting=0.0,
        false_northing=0.0,
        standard_parallels=(50, 70.0),
        globe=None,
    )

    for ind, phase in enumerate([P1, P2, P3, P4]):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection=proj)

        ax.set_extent([-135.25, -134.9, 60.65, 60.81])

        # Add background
        ax.add_image(request2, 13)
        ax.add_image(request1, 13, alpha=0.5)

        # Draw gridlines
        gl1 = ax.gridlines(
            draw_labels=True,
            xlocs=np.arange(-136.0, -134.0, 0.1),
            ylocs=np.arange(60.0, 61.0, 0.1),
            linestyle=":",
            color="w",
            zorder=2,
        )

        # Turn off labels on certin sides of figure
        gl1.top_labels = False
        gl1.right_labels = False

        # Update label fontsize
        gl1.xlabel_style = {"size": 16}
        gl1.ylabel_style = {"size": 16}

        # plot blank sites
        ax.scatter(
            stations_df["GNSS_longitude_rounded"].values,
            stations_df["GNSS_latitude_rounded"].values,
            color="k",
            marker="^",
            s=45,
            transform=ccrs.PlateCarree(),
            zorder=9,
            # label=names,
        )

        lat, lon = (
            phase["GNSS_latitude_rounded"].values,
            phase["GNSS_longitude_rounded"].values,
        )
        ax.scatter(
            lon,
            lat,
            color="r",
            marker="^",
            s=45,
            transform=ccrs.PlateCarree(),
            zorder=9,
        )

        # Add scalebar
        scale_bar(ax, proj, 4)

        # plt.title()

        # Save figure
        plt.savefig(
            "./figures/site/stations_map_P" + str(ind + 1) + ".png",
            dpi=300,
            bbox_inches="tight",
        )


def plot_stations_map():
    stations_df = pd.read_csv("./results/site/df_mapping.csv")

    stations_df = stations_df.drop_duplicates("site")
    lats, lons = (
        stations_df["GNSS_latitude_rounded"],
        stations_df["GNSS_longitude_rounded"],
    )

    # Google image tiling
    request1 = cimgt.GoogleTiles(style="satellite")
    request2 = cimgt.GoogleTiles(
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}.jpg"
    )

    # Map projection
    proj = ccrs.AlbersEqualArea(
        central_longitude=-135.076167,
        central_latitude=60.729549,
        false_easting=0.0,
        false_northing=0.0,
        standard_parallels=(50, 70.0),
        globe=None,
    )

    # Create figure and axis (you might want to edit this to focus on station coverage)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=proj)

    ax.set_extent([-135.25, -134.9, 60.65, 60.81])

    # Add background
    ax.add_image(request2, 13)
    ax.add_image(request1, 13, alpha=0.5)

    # Draw gridlines
    gl1 = ax.gridlines(
        draw_labels=True,
        xlocs=np.arange(-136.0, -134.0, 0.1),
        ylocs=np.arange(60.0, 61.0, 0.1),
        linestyle=":",
        color="w",
        zorder=2,
    )

    # Turn off labels on certin sides of figure
    gl1.top_labels = False
    gl1.right_labels = False

    # Update label fontsize
    gl1.xlabel_style = {"size": 16}
    gl1.ylabel_style = {"size": 16}

    # plot blank sites
    ax.scatter(
        lons,
        lats,
        color="k",
        marker="^",
        s=45,
        transform=ccrs.PlateCarree(),
        zorder=9,
        # label=names,
    )

    for site in [5, 64, 17]:
        lat, lon = (
            stations_df[stations_df["site"] == site]["GNSS_latitude_rounded"].values[0],
            stations_df[stations_df["site"] == site]["GNSS_longitude_rounded"].values[
                0
            ],
        )
        print(lat, lon)
        ax.scatter(
            lon,
            lat,
            color="r",
            marker="o",
            s=200,
            facecolors="none",
            transform=ccrs.PlateCarree(),
        )

    # Add scalebar
    scale_bar(ax, proj, 4)

    """
    # add well locations
    df = pd.read_csv("./data/yukon_datasets/Water_wells.csv")

    lons = df["X"]
    lats = df["Y"]

    ax.scatter(lons, lats, c="red")
    """

    # Save figure
    plt.savefig("./results/figures/site/stations_map.png", dpi=300, bbox_inches="tight")


def plot_stations_wells_map():

    lon_lim = [-135.25, -134.9]
    lat_lim = [60.65, 60.81]

    variable, lat_name, lon_name = "DEPBEDROCK", "LAT_DD", "LONG_DD"
    dir_name, shp_name = "DrillHoles/WaterWells", "Water_Wells"

    shp_path = "./data/yukon_datasets/" + dir_name + "/" + shp_name + ".shp"
    # gdb_file = gpd.read_file(dir_path + data_name + ".gdb", driver="OpenFileGDB")
    shp_file = gpd.read_file(shp_path, driver="ESRI Shapefile")
    # kml_file = gpd.read_file(dir_path + data_name + ".kmz", driver="libkml")

    # shp_file = shp_file.to_crs(epsg=3578)

    # print(shp_file.crs)
    shp_file = shp_file.to_crs("EPSG:4326")

    # filter to site bounds
    """
    shp_file = shp_file[
        (shp_file[lat_name] >= lat_lim[0])
        & (shp_file[lat_name] <= lat_lim[1])
        & (shp_file[lon_name] >= lon_lim[0])
        & (shp_file[lon_name] <= lon_lim[1])
    ]
    """

    # Google image tiling
    request1 = cimgt.GoogleTiles(style="satellite")
    request2 = cimgt.GoogleTiles(
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}.jpg"
    )

    # Map projection
    proj = ccrs.AlbersEqualArea(
        central_longitude=-135.076167,
        central_latitude=60.729549,
        false_easting=0.0,
        false_northing=0.0,
        standard_parallels=(50, 70.0),
        globe=None,
    )

    # Create figure and axis (you might want to edit this to focus on station coverage)

    # """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=proj)

    # if is_string_dtype(shp_file[variable]):
    if (
        shp_file[variable].str.contains(">").any()
        or shp_file[variable].str.contains("<").any()
    ):
        exact_ds = shp_file[
            (~shp_file[variable].str.contains(">", na=False))
            & (~shp_file[variable].str.contains("<", na=False))
        ]
        exact_ds[variable] = exact_ds[variable].values.astype(float) * 0.3048

        g_ds = shp_file[shp_file[variable].str.contains(">", na=False)]
        g_ds[variable] = (
            g_ds[variable].str.replace(">", "").values.astype(float) * 0.3048
        )

        l_ds = shp_file[shp_file[variable].str.contains("<", na=False)]
        l_ds[variable] = l_ds[variable].str.replace("<", "").values.astype(float)

        ax = g_ds.plot(
            # ax=ax,
            column=variable,
            # legend=True,
            # categorical=True,
            # legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
            legend_kwds={"label": "depth(m)"},
            edgecolors="red",
            aspect=None,
        )

        ax = exact_ds.plot(
            ax=ax,
            column=variable,
            legend=True,
            # categorical=True,
            # legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
            # legend_kwds={"label": variable},
            legend_kwds={"label": "depth(m)"},
            aspect=None,
        )

        g_ds = g_ds.dropna(subset=[variable])
        exact_ds = exact_ds.dropna(subset=[variable])

        c = g_ds[variable].astype(float).values
        ax.scatter(
            g_ds[lon_name],
            g_ds[lat_name],
            c=c,
            # legend=True,
            # categorical=True,
            # legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
            # legend_kwds={"label": "depth(m)"},
            edgecolors="red",
            # aspect=None,
        )

        c = exact_ds[variable].astype(float).values
        ax.scatter(
            exact_ds[lon_name],
            exact_ds[lat_name],
            c=c,
            # legend=True,
            # categorical=True,
            # legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
            # legend_kwds={"label": variable},
            # legend_kwds={"label": "depth(m)"},
            # aspect=None,
        )

        # plt.colorbar()

    stations_df = pd.read_csv("./data/df_mapping.csv")

    stations_df = stations_df.drop_duplicates("site")
    lats, lons = (
        stations_df["GNSS_latitude_rounded"],
        stations_df["GNSS_longitude_rounded"],
    )

    # ax.set_extent([-135.25, -134.9, 60.65, 60.81])
    """
    # Add background
    ax.add_image(request2, 13)
    ax.add_image(request1, 13, alpha=0.5)

    # Draw gridlines
    gl1 = ax.gridlines(
        draw_labels=True,
        xlocs=np.arange(-136.0, -134.0, 0.1),
        ylocs=np.arange(60.0, 61.0, 0.1),
        linestyle=":",
        color="w",
        zorder=2,
    )

    # Turn off labels on certin sides of figure
    gl1.top_labels = False
    gl1.right_labels = False

    # Update label fontsize
    gl1.xlabel_style = {"size": 16}
    gl1.ylabel_style = {"size": 16}

    # plot blank sites
    ax.scatter(
        lons,
        lats,
        color="k",
        marker="^",
        s=45,
        transform=ccrs.PlateCarree(),
        zorder=9,
        # label=names,
    )

    # Add scalebar
    scale_bar(ax, proj, 4)
    """
    # Save figure
    plt.savefig("./figures/site/stations_wells_map.png", dpi=300, bbox_inches="tight")


### PLOT YUKON DATASETS
def plot_yukon_datasets_plotly(
    dir_name, shp_name, variable, lat_name, lon_name, lat_lim, lon_lim, plot_type
):
    shp_path = "./data/yukon_datasets/" + dir_name + "/" + shp_name + ".shp"
    # gdb_file = gpd.read_file(dir_path + data_name + ".gdb", driver="OpenFileGDB")
    shp_file = gpd.read_file(shp_path, driver="ESRI Shapefile")
    # kml_file = gpd.read_file(dir_path + data_name + ".kmz", driver="libkml")

    shp_file = shp_file.to_crs("EPSG:4326")

    # filter to site bounds
    shp_file = shp_file[
        (shp_file[lat_name] >= lat_lim[0])
        & (shp_file[lat_name] <= lat_lim[1])
        & (shp_file[lon_name] >= lon_lim[0])
        & (shp_file[lon_name] <= lon_lim[1])
    ]

    # xlim(-135.3, -134.9)
    # ylim(60.65, 60.81)

    # fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "scatter":
        fig = px.scatter_map(
            shp_file,
            lat=lat_name,
            lon=lon_name,
            color=variable,  # , size_max=15, zoom=10
            size=15,
            width=750,
            height=750,
        )

    else:
        ax = shp_file.plot(
            # ax=ax,
            column=variable,
            legend=True,
            categorical=True,
            legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
        )

    # cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    # cx.add_basemap(ax, crs=shp_file.crs)
    # cx.add_basemap(
    #    ax, crs=shp_file.crs.to_string(), source=cx.providers.CartoDB.Voyager
    # )

    # ax.set_xlim(*lon_lim)
    # ax.set_ylim(*lat_lim)
    # fig.update_layout(xaxis_range=lon_lim, yaxis_range=lat_lim)
    # """
    fig.update_layout(
        map_bounds={
            "west": lon_lim[0],
            "east": lon_lim[1],
            "south": lat_lim[0],
            "north": lat_lim[1],
        }
    )
    # """
    fig.show()

    # plt.title(variable)
    # plt.show()


def plot_yukon_datasets_plotly_subplots(
    dir_name, shp_name, variable, lat_name, lon_name, plot_type
):
    shp_path = "./data/yukon_datasets/" + dir_name + "/" + shp_name + ".shp"
    # gdb_file = gpd.read_file(dir_path + data_name + ".gdb", driver="OpenFileGDB")
    shp_file = gpd.read_file(shp_path, driver="ESRI Shapefile")
    # kml_file = gpd.read_file(dir_path + data_name + ".kmz", driver="libkml")

    # shp_file = shp_file.to_crs("EPSG:4326")
    # shp_file = shp_file.to_crs("ESRI:102001")

    # full bounds
    full_lon_lim = [-135.5, -134.5]
    full_lat_lim = [60.45, 60.95]

    # whitehorse bounds
    wh_lon_lim = [-135.3, -134.9]
    wh_lat_lim = [60.65, 60.81]

    # filter to site bounds
    shp_file = shp_file[
        (shp_file[lat_name] >= full_lat_lim[0])
        & (shp_file[lat_name] <= full_lat_lim[1])
        & (shp_file[lon_name] >= full_lon_lim[0])
        & (shp_file[lon_name] <= full_lon_lim[1])
    ]

    # xlim(-135.3, -134.9)
    # ylim(60.65, 60.81)

    # fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "scatter":
        # fig = px.scatter_geo(
        fig = px.scatter_map(
            shp_file,
            lat=lat_name,
            lon=lon_name,
            color=variable,  # , size_max=15, zoom=10
            width=750,
            height=750,
            # projection="conic equal area",
            # projection="albers usa",
            # map_bounds={
            #    "west": full_lon_lim[0],
            #    "east": full_lon_lim[1],
            #    "south": full_lat_lim[0],
            #    "north": full_lat_lim[1],
            # },
            map_style="carto-voyager",
        )
        """
        fig.add_trace(
            go.Scatter(
                x=[wh_lon_lim[0], wh_lon_lim[1], wh_lon_lim[1], wh_lon_lim[0]],
                y=[wh_lat_lim[0], wh_lat_lim[0], wh_lat_lim[1], wh_lat_lim[1]],
            )
        )
        """
        """
        fig.add_shape(
            xref="x",
            yref="y",
            type="rect",
            x0=wh_lon_lim[0],
            y0=wh_lat_lim[0],
            x1=wh_lon_lim[1],
            y1=wh_lat_lim[1],
            line=dict(color="red"),
            # fillcolor="LightSkyBlue",
        )
        """
    else:
        pass

    fig.update_layout(
        map_bounds={
            "west": full_lon_lim[0],
            "east": full_lon_lim[1],
            "south": full_lat_lim[0],
            "north": full_lat_lim[1],
        }
    )

    fig.show()

    # plt.title(variable)
    # plt.show()


def plot_yukon_datasets(
    dir_name,
    shp_name,
    variable,
    lat_lim,
    lon_lim,
    plot_type,
    lat_name=None,
    lon_name=None,
):
    """
    stations_df = pd.read_csv("./results/site/df_mapping.csv")
    stations_df = stations_df.drop_duplicates("site")
    lats, lons = (
        stations_df["GNSS_latitude_rounded"],
        stations_df["GNSS_longitude_rounded"],
    )
    """

    shp_path = "./data/yukon_datasets/" + dir_name + "/" + shp_name + ".shp"
    # gdb_file = gpd.read_file(dir_path + data_name + ".gdb", driver="OpenFileGDB")
    shp_file = gpd.read_file(shp_path, driver="ESRI Shapefile")
    # kml_file = gpd.read_file(dir_path + data_name + ".kmz", driver="libkml")

    # shp_file = shp_file.to_crs(epsg=3578)

    # print(shp_file.crs)
    shp_file = shp_file.to_crs("EPSG:4326")
    # print(shp_file.crs)
    # shp_file = shp_file.to_crs("ESRI:102001")
    # print(shp_file.crs)

    print(shp_file.columns)

    # Google image tiling
    request1 = cimgt.GoogleTiles(style="satellite")
    request2 = cimgt.GoogleTiles(
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}.jpg"
    )

    # Map projection
    proj = ccrs.AlbersEqualArea(
        central_longitude=np.mean(lon_lim),  # -135.076167,
        central_latitude=np.mean(lat_lim),  # 60.729549,
        false_easting=0.0,
        false_northing=0.0,
        standard_parallels=(50, 70.0),
        globe=None,
    )
    # proj = ccrs.PlateCarree(
    #    # central_longitude=np.mean(lon_lim)  # , central_latitude=np.mean(lat_lim)
    # )
    """
    ax.coastlines(resolution=res, linewidth=0.6, color="black", alpha=0.8, zorder=4)
    ax.add_feature(cpf.BORDERS, linestyle=':', alpha=0.4, zorder=2)
    ax.add_feature(cpf.LAND, color="lightgrey", zorder=2)
    projection = ccrs.PlateCarree(central_longitude=0) 
    ax.add_geometries(shdf.geometry,
                    projection,
                    facecolor="red",
                    edgecolor='k')
    """

    # Create figure and axis (you might want to edit this to focus on station coverage)

    # fig = plt.figure(figsize=(10, 10))
    # fig = plt.figure()
    # ax = fig.add_subplot(projection=proj)

    if plot_type == "scatter":
        # filter to site bounds
        shp_file = shp_file[
            (shp_file[lat_name] >= lat_lim[0])
            & (shp_file[lat_name] <= lat_lim[1])
            & (shp_file[lon_name] >= lon_lim[0])
            & (shp_file[lon_name] <= lon_lim[1])
        ]

        # if is_string_dtype(shp_file[variable]):
        if (
            shp_file[variable].str.contains(">").any()
            or shp_file[variable].str.contains("<").any()
        ):
            exact_ds = shp_file[
                (~shp_file[variable].str.contains(">", na=False))
                & (~shp_file[variable].str.contains("<", na=False))
            ]
            exact_ds[variable] = exact_ds[variable].values.astype(float) * 0.3048

            g_ds = shp_file[shp_file[variable].str.contains(">", na=False)]
            g_ds[variable] = (
                g_ds[variable].str.replace(">", "").values.astype(float) * 0.3048
            )

            l_ds = shp_file[shp_file[variable].str.contains("<", na=False)]
            l_ds[variable] = (
                l_ds[variable].str.replace("<", "").values.astype(float) * 0.3048
            )

            ax = g_ds.plot(
                # ax=ax,
                column=variable,
                # legend=True,
                # categorical=True,
                # legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
                legend_kwds={"label": "depth(m)"},
                edgecolors="red",
                aspect=None,
            )

            ax = exact_ds.plot(
                ax=ax,
                column=variable,
                legend=True,
                # categorical=True,
                # legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
                # legend_kwds={"label": variable},
                legend_kwds={"label": "depth(m)"},
                aspect=None,
            )

            plt.xlim(*lon_lim)
            plt.ylim(*lat_lim)

        else:

            ax = shp_file.plot(
                # ax=ax,
                column=variable,
                legend=True,
                # categorical=True,
                # legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
                # legend_kwds={"label": variable},
                aspect=None,
            )
    elif plot_type == "categorical":
        in_bounds = []
        for p in shp_file["geometry"].values:
            in_b = (
                (lon_lim[0] <= p.bounds[0])
                | (lat_lim[0] <= p.bounds[1])
                | (lon_lim[1] <= p.bounds[2])
                | (lat_lim[1] <= p.bounds[3])
            )
            in_bounds.append(in_b)

        # print(len(in_bounds), np.sum(in_bounds))
        shp_file = shp_file[in_bounds]
        ax = shp_file.plot(
            # ax=ax,
            column=variable,
            legend=True,
            categorical=True,
            # legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
            # legend_kwds={"label": "depth (m)"},
            aspect=None,
        )

        plt.xlim(*lon_lim)
        plt.ylim(*lat_lim)

    # ax.set_extent([-135.25, -134.9, 60.65, 60.81])
    # plt.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])

    # Add background
    # ax.add_image(request2, 13)
    # ax.add_image(request1, 13, alpha=0.5)

    """
    cx.providers.keys()
    dict_keys(['OpenStreetMap', 'OpenSeaMap', 'OpenPtMap', 'OpenTopoMap', 'OpenRailwayMap', 
        'OpenFireMap', 'SafeCast', 'Thunderforest', 'OpenMapSurfer', 'Hydda', 'MapBox', 'Stamen', 
        'Esri', 'OpenWeatherMap', 'HERE', 'FreeMapSK', 'MtbMap', 'CartoDB', 'HikeBike', 'BasemapAT', 
        'nlmaps', 'NASAGIBS', 'NLS', 'JusticeMap', 'Wikimedia', 'GeoportailFrance', 'OneMapSG'])

    cx.providers.CartoDB.keys()

    OpenStreetMap: ['Mapnik', 'DE', 'CH', 'France', 'HOT', 'BZH', 'BlackAndWhite']

    """

    # print(cx.providers.Esri.keys())
    # ['WorldStreetMap', 'WorldTopoMap', 'WorldImagery', 'WorldTerrain',
    # 'WorldShadedRelief', 'WorldPhysical', 'OceanBasemap', 'NatGeoWorldMap',
    # 'WorldGrayCanvas', 'ArcticImagery', 'ArcticOceanBase', 'ArcticOceanReference',
    # 'AntarcticImagery', 'AntarcticBasemap']

    cx.add_basemap(
        ax,
        # source=cx.providers.Esri.WorldStreetMap,
        source=cx.providers.OpenStreetMap.Mapnik,
        # source=cx.providers.CartoDB.Positron,
        # crs="EPSG:3578"
        crs="EPSG:4326",
    )  # , zoom=12)

    # fig.set_title("plot_title")
    # fig.set_xlabel("Longitude")
    # fig.set_ylabel("Latitude")

    """
    # Draw gridlines
    gl1 = ax.gridlines(
        draw_labels=True,
        xlocs=np.arange(-136.0, -134.0, 0.1),
        ylocs=np.arange(60.0, 61.0, 0.1),
        linestyle=":",
        color="w",
        zorder=2,
    )
    """
    # Turn off labels on certain sides of figure
    # gl1.top_labels = False
    # gl1.right_labels = False

    # Update label fontsize
    # gl1.xlabel_style = {"size": 16}
    # gl1.ylabel_style = {"size": 16}

    # Add scalebar
    # scale_bar(ax, proj, 4)

    """
    # add well locations
    df = pd.read_csv("./data/yukon_datasets/Water_wells.csv")

    lons = df["X"]
    lats = df["Y"]

    ax.scatter(lons, lats, c="red")
    """

    # Save figure
    # plt.savefig("./results/figures/site/stations_map.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_drillholes():
    plot_type = "scatter"
    # plot_type = "categorical"

    # lon_lim = [-137.0, -133.0]
    # lat_lim = [60.0, 62.0]

    # lon_lim = [-135.5, -134.5]
    # lat_lim = [60.45, 60.95]

    lon_lim = [-135.3, -134.9]
    lat_lim = [60.65, 60.81]

    # 'TEMP_WATER', 'DRILL_DEPTH', 'BROCKDEPTH', 'THERM_GRAD', 'LAT_DD', 'LONG_DD', 'THERM_COND'
    variable, lat_name, lon_name = "BROCKDEPTH", "LAT_DD", "LONG_DD"
    dir_name, shp_name = "DrillHoles/GeothermalBoreholes", "Geothermal_Boreholes"
    # plot_yukon_datasets(
    #    dir_name, shp_name, variable, lat_lim, lon_lim, plot_type, lat_name, lon_name
    # )

    # 'WELL_DEPTH', 'DEPBEDROCK', 'CAS_ELEV', 'GROUNDELEV', 'LAT_DD', 'LONG_DD'
    variable, lat_name, lon_name = "DEPBEDROCK", "LAT_DD", "LONG_DD"
    dir_name, shp_name = "DrillHoles/WaterWells", "Water_Wells"
    plot_yukon_datasets(
        dir_name, shp_name, variable, lat_lim, lon_lim, plot_type, lat_name, lon_name
    )

    # 'SAMPLE_NUM', 'AGE', 'P_SUITE', 'BATH_PLUT', 'DESC_', 'LATITUDE_D', 'LONGITUDE_', 'LOC_ACC', 'COMP',
    # 'P_kg_m3', 'K2O', 'cTh', 'cU', 'cK', 'A_uW_m3', 'created_da', 'last_edite', 'last_edi_1', 'geometry'
    # variable, lat_name, lon_name = "A_uW_m3", "LATITUDE_D", "LONGITUDE_"
    # dir_name, shp_name = (
    #    "Geothermal/GeothermalDataset/Shapefiles",
    #    "RadiogenicHeatProduction",
    # )

    # 'NAME', 'LATITUDE_D', 'LONGITUDE_', 'LOC_SOURCE', 'WATER_TEMP', 'DISCHARGE', 'GT_SILICA', 'GT_ALKALI',
    # 'PH', 'CONDUCTIVI', 'HARDNESS', 'CA_MG_L', 'MG_MG_L', 'NA_MG_L', 'K_MG_L', 'HCO3_MG_L', 'SO4_MG_L',
    # 'CO3_MG_L', 'CL_MG_L', 'F_MG_L', 'SIO2_MG_L', 'E_ISO_18O', 'E_ISO_2H', 'RAD_ISO_3H', 'RAD_ISO_14',
    # 'GAS_NOBLE', 'GAS_DISS', 'GAS_EMITTE', 'REFERENCE', 'REF_LINK', 'COMMENTS', 'PUBLIC', 'created_da',
    # 'last_edite', 'last_edi_1', 'geometry'
    # variable, lat_name, lon_name = "WATER_TEMP", "LATITUDE_D", "LONGITUDE_"
    # dir_name, shp_name = "Geothermal/GeothermalDataset/Shapefiles", "ThermalSprings"


def plot_datasets():

    # 'TEMP_WATER', 'DRILL_DEPTH', 'BROCKDEPTH', 'THERM_GRAD', 'LAT_DD', 'LONG_DD', 'THERM_COND'
    # variable, lat_name, lon_name = "TEMP_WATER", "LAT_DD", "LONG_DD"
    # dir_name, shp_name = "DrillHoles/GeothermalBoreholes", "Geothermal_Boreholes"

    # 'WELL_DEPTH', 'DEPBEDROCK', 'CAS_ELEV', 'GROUNDELEV', 'LAT_DD', 'LONG_DD'
    # variable, lat_name, lon_name = "DEPBEDROCK", "LAT_DD", "LONG_DD"
    # dir_name, shp_name = "DrillHoles/WaterWells", "Water_Wells"

    # 'UNIT_1M', 'UNIT_250K', 'UNIT_ORIG', 'ASSEMBLAGE', 'SUPERGROUP', 'GP_SUITE',
    # 'FORMATION', 'MEMBER', 'NAME', 'TERRANE', 'AGE_MAX_MA', 'AGE_MIN_MA',
    # 'ROCK_CLASS', 'ROCK_SUBCL', 'ROCK_MAJOR', 'ROCK_MINOR', 'ROCK_NOTES'
    # variable = "AGE_MIN_MA"
    # dir_name, shp_name = (
    #    "Geology/BedrockGeology/Yukon_Bedrock_Geology_Complete.shp",
    #    "Geology/BedrockGeology",
    #    "Bedrock_Geology",
    # )

    #'TERRANE', 'TECT_SETTING', 'AGE_RANGE'
    # variable = "TERRANE"
    # dir_name, shp_name = "Geology/Terranes", "Terranes"

    # 'SAMPLE_NUM', 'AGE', 'P_SUITE', 'BATH_PLUT', 'DESC_', 'LATITUDE_D', 'LONGITUDE_', 'LOC_ACC', 'COMP',
    # 'P_kg_m3', 'K2O', 'cTh', 'cU', 'cK', 'A_uW_m3', 'created_da', 'last_edite', 'last_edi_1', 'geometry'
    # variable, lat_name, lon_name = "A_uW_m3", "LATITUDE_D", "LONGITUDE_"
    # dir_name, shp_name = (
    #    "Geothermal/GeothermalDataset/Shapefiles",
    #    "RadiogenicHeatProduction",
    # )

    # 'NAME', 'LATITUDE_D', 'LONGITUDE_', 'LOC_SOURCE', 'WATER_TEMP', 'DISCHARGE', 'GT_SILICA', 'GT_ALKALI',
    # 'PH', 'CONDUCTIVI', 'HARDNESS', 'CA_MG_L', 'MG_MG_L', 'NA_MG_L', 'K_MG_L', 'HCO3_MG_L', 'SO4_MG_L',
    # 'CO3_MG_L', 'CL_MG_L', 'F_MG_L', 'SIO2_MG_L', 'E_ISO_18O', 'E_ISO_2H', 'RAD_ISO_3H', 'RAD_ISO_14',
    # 'GAS_NOBLE', 'GAS_DISS', 'GAS_EMITTE', 'REFERENCE', 'REF_LINK', 'COMMENTS', 'PUBLIC', 'created_da',
    # 'last_edite', 'last_edi_1', 'geometry'
    variable, lat_name, lon_name = "WATER_TEMP", "LATITUDE_D", "LONGITUDE_"
    dir_name, shp_name = "Geothermal/GeothermalDataset/Shapefiles", "ThermalSprings"

    plot_type = "scatter"
    # plot_type = "categorical"

    # lon_lim = [-137.0, -133.0]
    # lat_lim = [60.0, 62.0]

    # lon_lim = [-135.5, -134.5]
    # lat_lim = [60.45, 60.95]

    lon_lim = [-135.3, -134.9]
    lat_lim = [60.65, 60.81]

    # plot_yukon_datasets(dir_name, shp_name, variable, lat_lim, lon_lim, plot_type)
    plot_yukon_datasets(
        dir_name, shp_name, variable, lat_lim, lon_lim, plot_type, lat_name, lon_name
    )
    # plot_yukon_datasets_plotly(
    #    dir_name, shp_name, variable, lat_name, lon_name, lat_lim, lon_lim, plot_type
    # )
    # plot_yukon_datasets_plotly_subplots(
    #    dir_name, shp_name, variable, lat_name, lon_name, plot_type
    # )
