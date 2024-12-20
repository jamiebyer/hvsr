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

    lat_PR = 54.72
    lon_PR = -113.29
    lat_DC = 56.0027
    lon_DC = -119.7426

    plt.figure(figsize=(4, 4))
    ax = plt.axes(
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
    plt.savefig("world.pdf")


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
    ax, proj, length, location=(0.5, 0.05), linewidth=3, units="km", m_per_unit=1000
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
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
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
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(
        sbcx,
        sbcy + 1,
        str(length) + " " + units,
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


def get_station_locations_full_xml():
    path = "./data/FDSN_Information.xml"

    with open(path, "r") as f:
        file = f.read()

    soup = BeautifulSoup(file, "xml")
    sites = soup.find_all("Site")
    names = [("".join(filter(str.isdigit, site.text))) for site in sites]

    lats = [float(lat.text) for lat in soup.find_all("Latitude")]
    lons = [float(lon.text) for lon in soup.find_all("Longitude")]

    return names, lats, lons


def get_station_locations():
    path = "./data/parsed_xml.csv"

    df = pd.read_csv(path)
    # sites = soup.find_all("Site")
    names = df["Site"]
    lats = df["Latitude"]
    lons = df["Longitude"]

    return names, lats, lons


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
    fig = plt.figure(figsize=(10,10))
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

        median = df["median"][df["wind"]==w].values
        freqs = df["freqs"][df["wind"]==w].values
        
        peaks, _ = find_peaks(median, height=0.7*median.max())

        peak_ind=0
        f_0 = freqs[peaks[peak_ind]]
        f_0_list.append(f_0)
        a+=1



    cm = plt.cm.get_cmap('RdYlBu')
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
        norm=colors.LogNorm()
    )

    plt.colorbar(sc)

    # Add scalebar
    scale_bar(ax, proj, 6)
    # plt.show()

    # Save figure
    plt.savefig("./results/figures/f_0_map.png", dpi=300, bbox_inches="tight")

    

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


