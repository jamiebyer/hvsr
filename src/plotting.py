from obspy import read, Stream
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
from scipy import fft
import plotly.express as plotly_express
import plotly.graph_objects as go

"""
TODO:
- set up linters.
- add unit information
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

def plot_manual_hvsr(
        start_date, 
        station, 
        freqs,
        times_avg,
        east_avg, 
        north_avg, 
        vert_avg, 
        hvsr, 
        dir_path="./figures/"
    ):
    """
    """
    make_output_folder(dir_path, str(station) + "_hvsr")
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(times_avg, east_avg)
    plt.plot(times_avg, north_avg)
    plt.plot(times_avg, vert_avg)
    plt.legend(["east", "north", "vert"])
        
    shifted_freqs = fft.fftshift(freqs)

    plt.subplot(2, 1, 1)
    plt.plot(shifted_freqs[10:], fft.fft(east_avg)[10:])

    plt.subplot(2, 1, 2)
    # get horizontal-vertical ratio
    # would be with the fft transformed components

    plt.plot(shifted_freqs, fft.fftshift(hvsr))

    path = "./figures/" + str(station) + "_hvsr/" + str(start_date.month) + "-" + str(start_date.day) + ".png"
    plt.savefig(path)

def make_output_folder(dir_path, plot_type):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        
    if not os.path.isdir(dir_path + plot_type):
        os.mkdir(dir_path + plot_type)


def plot_raydec():
    pass



"""
DOWNSAMPLE FOR PLOTTING.
"""

if __name__ == "__main__":
    """
    run from terminal
    """
    plot_map_locations()
