from obspy import read
import numpy as np
import datetime
import pandas as pd
from utils.utils import make_output_folder, create_file_list
from dateutil import tz
import sys
import os
import pyarrow.parquet as pq
import obspy
import pytz
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, filtfilt
from obspy.core.utcdatetime import UTCDateTime
import time
from skimage.measure import block_reduce
from datetime import time


###### TIMESERIES PROCESSING ######


def filter_timeseries(in_path):
    """
    Keep in miniseed format.
    """
    # use df_mapping to get stations
    # df_mapping = pd.read_csv("./data/df_mapping.csv")

    st_E = obspy.read(in_path)
    tr_E = st_E.traces[0]
    st_N = obspy.read(in_path.replace(".E.", ".N."))
    tr_N = st_N.traces[0]
    st_Z = obspy.read(in_path.replace(".E.", ".Z."))
    tr_Z = st_Z.traces[0]

    delta = tr_E.stats["delta"]
    start_date = tr_E.stats["starttime"]
    start_date = datetime.datetime(
        year=start_date.year, month=start_date.month, day=start_date.day
    )
    # Change timezone
    utc_tz = pytz.timezone("UTC")
    start_date = utc_tz.localize(start_date)
    start_date = start_date.astimezone(pytz.timezone("Canada/Yukon"))

    delta = tr_E.stats["delta"]
    delta_times = np.arange(0, tr_E.stats["npts"]) * datetime.timedelta(seconds=delta)
    times = start_date + delta_times

    hours = np.array([t.hour for t in times])
    night_inds = (hours >= 22) | (hours <= 6)

    ts = np.array([tr_Z.data, tr_N.data, tr_E.data])

    ft = np.fft.fftn(
        ts[:, night_inds],
        axes=[1],
    )

    freqs = np.fft.fftfreq(len(times[night_inds]), d=delta)
    freqs_shift = np.fft.fftshift(freqs)
    freqs_mag = np.sqrt(
        np.sum(np.array([freqs_shift, freqs_shift, freqs_shift]) ** 2, axis=0)
    )

    ft_filt = ft.copy()
    ft_filt[:, freqs_mag > 2.5] = 0

    ts_filt = np.fft.ifftn(ft_filt, axes=[1]).real

    return times, ts, ts_filt, night_inds, delta


def slice_timeseries(in_path):
    """
    Keep in miniseed format.
    """
    # use df_mapping to get stations
    # df_mapping = pd.read_csv("./data/df_mapping.csv")

    st_E = obspy.read(in_path)
    tr_E = st_E.traces[0]
    st_N = obspy.read(in_path.replace(".E.", ".N."))
    tr_N = st_N.traces[0]
    st_Z = obspy.read(in_path.replace(".E.", ".Z."))
    tr_Z = st_Z.traces[0]

    delta = tr_E.stats["delta"]
    start_date = tr_E.stats["starttime"]
    start_date = datetime.datetime(
        year=start_date.year, month=start_date.month, day=start_date.day
    )

    # Change timezone
    utc_tz = pytz.timezone("UTC")
    start_date = utc_tz.localize(start_date)
    start_date = start_date.astimezone(pytz.timezone("Canada/Yukon"))

    delta = tr_E.stats["delta"]
    delta_times = np.arange(0, tr_E.stats["npts"]) * datetime.timedelta(seconds=delta)
    times = start_date + delta_times

    hours = np.array([t.hour for t in times])
    night_inds = (hours >= 22) | (hours < 6)

    ts = np.array([tr_Z.data, tr_N.data, tr_E.data])

    return times, ts, night_inds, delta


def plot_station_noise():
    in_path = "./data/example_site/"
    site = "06"

    mags = []
    for f in os.listdir(in_path + site + "/"):
        if ".E." not in f:
            continue

        p = in_path + site + "/" + f
        times, ts, night_inds, delta = slice_timeseries(p)

        # make_output_folder("./figures/timeseries/" + site + "/")

        mag = np.sqrt(ts[0] ** 2 + ts[1] ** 2 + ts[2] ** 2)
        mags.append(mag)

    mags = np.array(mags)
    # downsample
    mags_downsampled = block_reduce(
        mags, block_size=(1, 5), func=np.mean, cval=np.mean(mags)
    )

    fig, ax = plt.subplots(1, 1)

    img = plt.imshow(mags_downsampled, interpolation="none", aspect="auto", norm="log")
    # ax.xaxis_date()
    fig.colorbar(img, label="Magnitude (mV)")

    # times_downsampled = block_reduce(
    #    times, block_size=(0, 5), func=np.mean, cval=np.mean(times)
    # )

    x_ticks = np.arange(0, mags_downsampled.shape[1] + 1, 200000)
    x_label_list = [times[x * 5].time() for x in x_ticks]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_label_list)

    # plt.axvline(times[int(5 * 60 * 60 * 20 / 5)], c="red")
    # plt.axvline(times[int(13 * 60 * 60 * 20 / 5)], c="red")

    plt.axvline(5 * 60 * 60 * 20, c="red")
    plt.axvline(13 * 60 * 60 * 20, c="red")
    # plt.xlim()
    plt.xlabel("Time")
    plt.ylabel("Day number")
    plt.title("Daily ambient seismic noise for a single station")

    plt.tight_layout()
    # plt.savefig(
    #    "./figures/timeseries/" + site + "/" + f.replace(".E.miniseed", ".png")
    # )
    plt.show()
