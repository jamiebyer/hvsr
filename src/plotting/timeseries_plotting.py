import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from utils.utils import make_output_folder
import xarray as xr
from processing.timeseries_processing import filter_timeseries, slice_timeseries

import plotly.io as pio


# TIMESERIES PLOT

def plot_station_noise(in_path, out_path):
    times, ts, night_inds, delta = slice_timeseries(in_path)

    #magnitude = magnitude.resample("1s").mean()

    # get noise frequency
    mag = np.sqrt(ts[0][night_inds]**2+ts[1][night_inds]**2+ts[2][night_inds]**2)
    #mags.append(mag)
    
    plt.clf()
    #plt.imshow(freqs)
    plt.plot(times[night_inds], mag)

    #plt.xticks(np.arange(0, 8*60 *60 *100, 8), [22, 23, 0, 1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.savefig(out_path)


def all_station_timeseries(ind):
    in_path = "./results/timeseries/sorted/"
    out_path = "./figures/timeseries/examples/"
    # sites
    sites = ["06", "07A", "17", "23", "24", "25", "32B", "34A", "38B", "41A", "41B", "42B", "47", "50"]
    #sites = ["06"]
    # sites = os.listdir(in_paths)

    in_paths = []
    out_paths = []
    make_folders = True
    for s in sites:
        for f in os.listdir(in_path + s + "/"):
            if ".E." not in f:
                continue
            if make_folders:
                make_output_folder(out_path + s + "/")
            in_paths.append(in_path + s + "/" + f)
            out_paths.append(out_path + s + "/" + f.replace(".E.miniseed", ".png"))
    
    plot_station_noise(in_paths[ind], out_paths[ind])



def plot_timeseries_processing():
    # plot original timeseries
    # plot all components
    # plot full day and show selected subset
    # show filtered timeseries
    # show transform and selection from transform
    # show selected selction from filtered timeseries

    times, ts, ts_filt, night_inds, _ = filter_timeseries()

    ts_mag = np.sqrt(np.nansum(ts**2, axis=0)).astype(float)
    ts_filt_mag = np.sqrt(np.nansum(ts_filt**2, axis=0)).astype(float)

    # magnitude = np.sqrt(np.nansum((ts - ts_filt) ** 2, axis=0)).astype(float)

    # hist before and after
    plt.subplot(2, 2, 1)
    hist1 = plt.hist(ts_mag, bins=20)
    plt.ylabel("counts")
    plt.xlabel("magnitude")
    plt.title("timeseries magnitude")

    plt.subplot(2, 2, 2)
    hist2 = plt.hist(ts_filt_mag, bins=20)
    plt.ylabel("counts")
    plt.xlabel("magnitude")
    plt.title("filtered timeseries magnitude")

    plt.subplot(2, 2, 3)
    plt.yscale("log")
    hist1 = plt.hist(ts_mag, bins=20)
    plt.ylabel("counts")
    plt.xlabel("magnitude")

    plt.subplot(2, 2, 4)
    plt.yscale("log")
    hist2 = plt.hist(ts_filt_mag, bins=20)
    plt.ylabel("counts")
    plt.xlabel("magnitude")

    plt.tight_layout()
    plt.show()



def plot_temperature():
    path = "./data/temperature/"
    # plot average from all stations in the background.
    df = pd.read_csv(
        path + station + ".csv",
        names=["millisecond_since_epoch", "yyyy-MM-ddThh:mm:ss", "data_value"],
        skiprows=[0],
    )
    fig = go.Figure(go.Scatter(x=df["yyyy-MM-ddThh:mm:ss"], y=df["data_value"]))

    return fig
