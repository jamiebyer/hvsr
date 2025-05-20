import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from utils.utils import make_output_folder
import xarray as xr

import plotly.io as pio


# TIMESERIES PLOT
def plot_timeseries(station, date, in_path="./results/timeseries/"):
    dir_in = in_path + str(station) + "/" + date + ".parquet"
    dir_out = "./results/figures/timeseries/" + str(station) + "/" + date + ".png"

    make_output_folder("./results/figures/timeseries/")
    make_output_folder("./results/figures/timeseries/" + str(station) + "/")

    df_timeseries = pd.read_parquet(dir_in, engine="pyarrow")
    df_timeseries.set_index("date", inplace=True)

    df_timeseries.index = pd.to_datetime(
        df_timeseries.index  # , format="ISO8601"  # format="%Y-%m-%d %H:%M:%S.%f%z"
    )

    df_timeseries = df_timeseries[df_timeseries["spikes"] == False]
    magnitude = df_timeseries["magnitude"]
    print(len(magnitude))
    magnitude = magnitude.resample("1s").mean()
    print(len(magnitude), "\n")

    # change to just amplitude...?
    timeseries_fig = px.line(
        magnitude
        # color_discrete_sequence=["rgba(100, 100, 100, 0.1)"],
        # color="spikes",
    )

    # timeseries_fig.write_image(dir_out)
    return timeseries_fig

    # return timeseries_fig


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
    """
    # Plot full timeseries
    plt.subplot(3, 1, 1)
    plt.plot(times, ts_mag, c=(0.35, 0.35, 0.35))
    plt.axvline(x=times[night_inds][0], c="red")
    plt.axvline(x=times[night_inds][-1], c="red")
    plt.title("Full day of ambient noise")
    plt.xlabel("Time")
    plt.ylabel("Magnitude (mV)")

    plt.subplot(3, 1, 2)
    plt.plot(times[night_inds], ts_mag[night_inds], c=(0.35, 0.35, 0.35))
    plt.title("Night section of ambient noise")
    plt.xlabel("Time")
    plt.ylabel("Magnitude (mV)")

    plt.subplot(3, 1, 3)
    plt.plot(times[night_inds], ts_filt_mag, c=(0.35, 0.35, 0.35))
    plt.title("Night section of filtered ambient noise")
    plt.xlabel("Time")
    plt.ylabel("Magnitude (mV)")

    plt.tight_layout()
    plt.show()
    """

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


def save_all_timeseries_plot():
    dir = "./results/timeseries/clipped/"
    for station in os.listdir(dir):
        print("\n", station)
        for file in os.listdir(dir + "/" + station + "/"):
            file = file.replace(".parquet", "")
            print(file)
            timeseries_fig = plot_timeseries(station, file, dir)

            output_dir = "./results/figures/timeseries"
            make_output_folder(output_dir + "/")
            make_output_folder(output_dir + "/" + str(station) + "/")
            pio.write_image(
                timeseries_fig,
                output_dir + "/" + station + "/" + file + ".png",
                engine="kaleido",
            )


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
