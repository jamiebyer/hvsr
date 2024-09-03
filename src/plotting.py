from obspy import read, Stream
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

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

    #print(soup.__dict__.keys())
    #print(soup.tagStack)

    lats = [float(lat.text) for lat in soup.find_all("Latitude")]
    lons = [float(lat.text) for lat in soup.find_all("Longitude")]

    plt.scatter(lons, lats)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()



def plot_station_info():
    df = pd.read_csv("./data/parsed_xml.csv")

    plt.scatter(df["Longitude"], df["Latitude"], label=df["Site"])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()

    
    # Elevation
    # Times


def plot_data():
    """
    save processed data in csv to read in, separated by station.
    can use xml info to get station location info, link it to data.
    """

    # read in data
    stream_east = read("data/453025390.0029.2024.07.04.00.00.00.000.E.miniseed", format="mseed")
    stream_north = read("data/453025390.0029.2024.07.04.00.00.00.000.N.miniseed", format="mseed")
    stream_vert = read("data/453025390.0029.2024.07.04.00.00.00.000.Z.miniseed", format="mseed")
    

    # make sure traces are length 1
    trace_east = stream_east.traces[0]
    trace_north = stream_north.traces[0]
    trace_vert = stream_vert.traces[0]

    stream_plot = Stream(traces=[trace_east, trace_north, trace_vert])
    dt = trace_east.stats.starttime
    
    stream_plot.plot(endtime=dt + 18*60*60)


    plt.show()


if __name__ == "__main__":
    """
    run from terminal
    """
    plot_station_info()
