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


###### TIMESERIES PROCESSING ######


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

    '''
    st_E.slice(
        UTCDateTime(times[night_inds][0]), UTCDateTime(times[night_inds][-1])
    ).write("./results/timeseries/example_timeseries_slice_E.miniseed", format="MSEED")
    st_N.slice(
        UTCDateTime(times[night_inds][0]), UTCDateTime(times[night_inds][-1])
    ).write("./results/timeseries/example_timeseries_slice_N.miniseed", format="MSEED")
    st_Z.slice(
        UTCDateTime(times[night_inds][0]), UTCDateTime(times[night_inds][-1])
    ).write("./results/timeseries/example_timeseries_slice_Z.miniseed", format="MSEED")
    '''
    return times, ts, night_inds, delta


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



def get_clean_timeseries_slice(
    include_outliers,
    in_path="./results/timeseries/raw/",
    out_path="./results/timeseries/clipped/",
):
    """
    Remove outliers/spikes and slice timeseries. Save timeseries stats.

    :param in_path: path to full timeseries parquet files.
    """

    timeseries_stats = {}
    for station in os.listdir(in_path):
        timeseries_stats[station] = {}
        for file in os.listdir(in_path + station):
            date = file.replace(".parquet", "")
            df_timeseries = pd.read_parquet(
                in_path + station + "/" + file, engine="pyarrow"
            )

            if include_outliers == False:
                # subset timeseries so only valid points are included
                df_timeseries = label_spikes(df_timeseries, station, date)

                df_timeseries = df_timeseries[df_timeseries["spikes"] == 0]

            timeseries_stats[station][date] = get_timeseries_stats(
                df_timeseries, hourly=True
            )

            # slice time
            df_timeseries = get_time_slice(df_timeseries)
            # making output paths
            make_output_folder(out_path + station)
            df_timeseries.to_parquet(out_path + station + "/" + date + ".parquet")

    # save stats to df
    df_stats = pd.DataFrame(timeseries_stats)
    df_stats.to_csv("./results/timeseries/stats/timeseries_slice.csv")



        

