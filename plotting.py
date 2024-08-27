from obspy import read, Stream
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_station_info():
    df = pd.read_csv("./data/parsed_xml.csv")

    plt.scatter(df["Latitude"], df["Longitude"], label=df["Site"])
    plt.legend()
    plt.show()


def plot_data():
    """
    trace.stats:
    network: SS
    station: 24025
    location: SW
    channel: EPE
    starttime: 2024-06-06T18:04:52.000000Z
    endtime: 2024-06-07T00:00:00.000000Z
    sampling_rate: 100.0
    delta: 0.01
    npts: 2130801
    calib: 1.0
    _format: MSEED
    mseed: AttribDict({'dataquality': 'D', 'number_of_records': 2110, 'encoding': 'FLOAT32', 'byteorder': '>', 'record_length': 4096, 'filesize': 8642560})
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


    # will loop over all files...
    # filename = "453024025.0001.2024.06.06.18.04.52.000.E.miniseed"
    # path = Path("/gilbert_lab/Whitehorse_ANT/")



plot_station_info()