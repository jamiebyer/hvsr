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

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(
        projection=ccrs.NearsidePerspective(
            satellite_height=2000000.0,
            central_longitude=lon_PR,
            central_latitude=lat_PR,
        )
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
        markerfacecolor="none",
        markersize=8,
        alpha=0.7,
        transform=ccrs.Geodetic(),
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
        "\u25B2\nN",
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


def plot_f_0_map(in_path="./results/raydec/csv/0-2-dfpar/"):

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

    ax.set_extent([-135.3, -134.9, 60.65, 60.81])

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
    gl1.xlabel_style = {"size": 10}
    gl1.ylabel_style = {"size": 10}

    positions_df = pd.read_csv("./data/station_positions.csv")
    mapping_df = pd.read_csv("./data/f_0_mapping.csv")

    f_0_sites = mapping_df["Site"].values

    f_0_site_inds = np.array([c in f_0_sites for c in positions_df["Code"].values])

    # plot blank sites
    ax.scatter(
        positions_df["Lon"][f_0_site_inds == False],
        positions_df["Lat"][f_0_site_inds == False],
        color="k",
        marker="^",
        s=75,
        transform=ccrs.PlateCarree(),
        zorder=9,
        # label=names,
    )

    # plot f_0 sites
    f_0_list = []
    a = 0
    for site in positions_df["Code"][f_0_site_inds == True].values:
        print(a)
        name = mapping_df["Station"][f_0_sites == site].values[0].split("_")

        station, date = name[0], name[1]
        for i in os.listdir(in_path):
            if station in i and date in i:
                file = i
        print(station, date)
        df = pd.read_csv(in_path + file)

        w = np.unique(df["wind"])[0]

        median = df["median"][df["wind"] == w].values
        freqs = df["freqs"][df["wind"] == w].values

        peaks, _ = find_peaks(median, height=0.7 * median.max())

        peak_ind = 0
        f_0 = freqs[peaks[peak_ind]]
        f_0_list.append(f_0)
        a += 1

    cm = plt.cm.get_cmap("RdYlBu")
    sc = ax.scatter(
        positions_df["Lon"][f_0_site_inds == True],
        positions_df["Lat"][f_0_site_inds == True],
        c=f_0_list,
        cmap=cm,
        marker="^",
        s=75,
        transform=ccrs.PlateCarree(),
        zorder=9,
        # label=names,
        norm=colors.LogNorm(),
    )

    plt.colorbar(sc)

    # Add scalebar
    scale_bar(ax, proj, 6)
    # plt.show()

    # Save figure
    plt.savefig("./results/figures/f_0_map.png", dpi=300, bbox_inches="tight")


def plot_stations_map():
    stations_df = pd.read_csv("./results/site/df_mapping.csv")
    
    stations_df = stations_df.drop_duplicates("site")
    lats, lons = stations_df["GNSS_latitude_rounded"], stations_df["GNSS_longitude_rounded"]

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
        lat, lon = stations_df[stations_df["site"] == site]["GNSS_latitude_rounded"].values[0], stations_df[stations_df["site"] == site]["GNSS_longitude_rounded"].values[0]
        print(lat, lon)
        ax.scatter(lon, lat, color='r', marker="o", s=200, facecolors='none', transform=ccrs.PlateCarree())

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
    plt.savefig(
        "./results/figures/site/stations_map.png", dpi=300, bbox_inches="tight"
    )


def plot_map(fig, gs, station=None):

    names, lats, lons = get_station_locations()
    # print(np.unique(names).shape, np.unique(names)[:10])
    # print(np.unique(lats).shape, np.unique(lats)[:10])
    # print(np.unique(lons).shape, np.unique(lons)[:10])

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
    # fig = plt.figure()  # (figsize=(10,10))

    ax = fig.add_subplot(gs[1], projection=proj)
    ax.set_extent([-135.3, -134.9, 60.65, 60.81])

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
    gl1.xlabel_style = {"size": 10}
    gl1.ylabel_style = {"size": 10}

    # for i in range(len(names)):
    ax.scatter(
        lons,
        lats,
        color="k",
        marker="^",
        # markersize=10,
        transform=ccrs.PlateCarree(),
        zorder=9,
        # label=names,
    )

    if station is not None and len(names) > 0:
        ax.scatter(
            np.array(lons)[names == int(station)],
            np.array(lats)[names == int(station)],
            color="red",
            marker="^",
            # markersize=10,
            transform=ccrs.PlateCarree(),
            zorder=9,
            # label=names,
        )

    # Add scalebar
    scale_bar(ax, proj, 6)
    # plt.show()

    # Save figure
    # plt.savefig("test_map.png", dpi=300, bbox_inches="tight")
    return fig


### WELL DATA ###


def read_well_data():
    """
    X,
    Y,
    OBJECTID,
    BOREHOLE_ID,
    WELL_NAME,
    COMMUNITY,
    PURPOSE,
    WELL_DEPTH_FTBGS,
    DEPTH_TO_BEDROCK_FTBGS,
    ESTIMATED_YIELD_GPM,
    YIELD_METHOD,
    STATIC_WATER_LEVEL_FTBTOC,
    DRILL_YEAR,
    DRILL_MONTH,
    DRILL_DAY,
    CASING_OUTSIDE_DIAM_IN,
    TOP_OF_SCREEN_FTBGS,
    BOTTOM_OF_SCREEN_FTBGS,
    TOP_OF_CASING_ELEVATION_MASL,
    GROUND_LEVEL_ELEVATION_MASL,
    WELL_HEAD_STICKUP_M,
    WELL_LOG,
    LINK,
    QUALITY,
    LOCATION_SOURCE,
    LATITUDE_DD,
    LONGITUDE_DD
    """

    # X: longitude
    # Y: latitude
    # WELL_DEPTH_FTBGS:
    # DEPTH_TO_BEDROCK_FTBGS:
    # GROUND_LEVEL_ELEVATION_MASL:

    df = pd.read_csv("./data/yukon_datasets/Water_wells.csv")

    lons = df["X"]
    lats = df["Y"]
    well_depth = df["WELL_DEPTH_FTBGS"]
    depth_to_bedrock = df["DEPTH_TO_BEDROCK_FTBGS"]
    ground_level_elevation = df["GROUND_LEVEL_ELEVATION_MASL"]

    depth_to_bedrock = (
        depth_to_bedrock.str.replace(">", "").str.replace("<", "").values.astype(float)
    )

    inds = (
        (lons > -136)
        & (lons < -134)
        & (lats > 60)
        & (lats < 61)
        & (depth_to_bedrock < 2800)
    )
    plt.scatter(lons[inds], lats[inds], c=depth_to_bedrock[inds])
    plt.colorbar()
    plt.xlim([-135.3, -134.9])
    plt.ylim([60.65, 60.81])

    plt.show()
