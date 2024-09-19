from obspy import read, Stream
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
from scipy import fft
import plotly.express as plotly_express
import plotly.graph_objects as go
from utils import make_output_folder

"""
TODO:
- set up linters.
- add unit information
"""

"""
RAW DATA AND STATION INFORMATION
"""

def plot_from_xml():
    path = "./data/FDSN_Information.xml"

    with open(path, 'r') as f:
        file = f.read()

    soup = BeautifulSoup(file, 'xml')
    lats = [float(lat.text) for lat in soup.find_all("Latitude")]
    lons = [float(lat.text) for lat in soup.find_all("Longitude")]

    plt.scatter(lons, lats)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def plot_station_info():
    df = pd.read_csv("./data/parsed_xml.csv")
    path = "./figures/station_locations.png"

    sites = df["Site"]
    lats = df["Latitude"]
    lons = df["Longitude"]
    elevs = df["Elevation"]
    
    plt.subplot(3, 1, 1)
    for ind in range(len(sites)):
        plt.scatter(lons[ind], lats[ind], c="blue")
        plt.text(lons[ind] - 0.01, lats[ind] + 0.003, sites[ind])
        
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    plt.subplot(3, 1, 2)
    for ind in range(len(sites)):
        plt.scatter(lons[ind], elevs[ind], c="blue")
        plt.text(lons[ind], elevs[ind], sites[ind])
        
    plt.xlabel("Longitude")
    plt.ylabel("Elevation")
    
    plt.subplot(3, 1, 3)
    for ind in range(len(sites)):
        plt.scatter(lats[ind], elevs[ind], c="blue")
        plt.text(lats[ind], elevs[ind], sites[ind])
        
    plt.xlabel("Latitude")
    plt.ylabel("Elevation")

    plt.tight_layout()
    plt.savefig(path)


def plot_map_locations():
    df = pd.read_csv("./data/parsed_xml.csv")
    path = "./figures/station_map_locations.html"

    fig = go.Figure(go.Scattermap(
        lat=df["Latitude"],
        lon=df["Longitude"],
        mode='markers',
        text=df["Site"],
    ))
    fig.update_layout(
        map=dict(
            center=dict(
                lat=60.74,
                lon=-135.08
            ),
            zoom=10
        ),
    )
    fig.write_html(path)

def plot_3d_locations():
    df = pd.read_csv("./data/parsed_xml.csv")
    path = "./figures/station_3d_locations.html"

    fig = plotly_express.scatter_3d(
        df,
        x="Longitude", 
        y="Latitude", 
        z="Elevation",
        text="Site", 
        # mode='markers'
    )
    fig.write_html(path)


def plot_station_timeseries(
        start_date, 
        station,
        times, 
        east, 
        north, 
        vert, 
        times_avg=None,
        east_avg=None,
        north_avg=None,
        vert_avg=None,
        dir_path="./figures/"
    ):

    plot_average = times_avg is not None
    make_output_folder(dir_path, str(station) + "_raw")
    
    plt.clf()
    plt.gcf().set_size_inches(7, 10)

    # TIME ZONE
    # plot as time series with average
    # plot day and night separately

    plt.subplot(3, 1, 1)
    plt.plot(times, east, label="east")
    if plot_average:
        plt.plot(times_avg, east_avg, label="east avg")
    plt.title("east")

    plt.subplot(3, 1, 2)
    plt.plot(times, north, label="north")
    if plot_average:
        plt.plot(times_avg, north_avg, label="north avg")
    plt.title("north")
    
    plt.subplot(3, 1, 3)
    plt.plot(times, vert, label="vert")
    if plot_average:
        plt.plot(times_avg, vert_avg, label="vert avg")
    plt.title("vert")

    plt.xlabel("time")
    plt.ylabel("mV")
    
    plt.suptitle("station: " + str(station) + "; start date:" + str(start_date.year) + "-" + str(start_date.month) + "-" + str(start_date.day))
    plt.tight_layout

    path = dir_path + str(station) + "_raw/" + str(start_date.month) + "-" + str(start_date.day) + ".png"
    plt.savefig(path)


"""
PLOTS OF RAYDEC PROCESSING
"""

def plot_timeseries_slice():
    station = 24025
    dir_in = "./timeseries/" + str(station) + "/"
    for file_name in os.listdir(dir_in):
        df = pd.read_csv(dir_in + file_name)
        plt.plot(df["times"], df["east"])
        plt.show()


def plot_raydec():
    station = 24025
    timeseries_dir = "./timeseries/" + str(station) + "/"
    raydec_dir = "./raydec/" + str(station) + "/"
    for file_name in os.listdir(timeseries_dir):
    #for file_name in os.listdir(timeseries_dir):
        timeseries_df = pd.read_csv(timeseries_dir + file_name)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(timeseries_df["times"], timeseries_df["east"], alpha=0.3)
        plt.plot(timeseries_df["times"], timeseries_df["vert"], alpha=0.3)
        plt.plot(timeseries_df["times"], timeseries_df["north"], alpha=0.3)

        raydec_df = pd.read_csv(raydec_dir + file_name)

        plt.subplot(2, 1, 2)
        freqs = pd.to_numeric(raydec_df.columns[1:])
        plt.plot(freqs, raydec_df.iloc[0][1:])

        #plt.show()
        make_output_folder("./figures/" + str(station) + "_raydec/")
        plt.savefig("./figures/" + str(station) + "_raydec/" + file_name.replace("csv", "png"))

def plot_raydec():
    df = pd.read_csv("./raydec/24025/2024-06-08.csv", index_col=0).T
    #df = pd.read_csv("./raydec/24025/2024-06-07.csv")

    plt.plot(df.index, df, c="grey", alpha=0.2)
    plt.plot(df.index, df.mean(axis=1), c="black")
    plt.show()

"""
DOWNSAMPLE FOR PLOTTING.
"""

if __name__ == "__main__":
    """
    run from terminal
    """
    plot_raydec()
