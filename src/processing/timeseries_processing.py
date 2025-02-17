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


###### TIMESERIES PROCESSING ######


def rolling_median(data, window):
    if len(data) < window:
        subject = data[:]
    else:
        subject = data[-30:]
    return sorted(subject)[len(subject) / 2]


def filter_timeseries():
    """
    Keep in miniseed format.
    """
    # use df_mapping to get stations
    # df_mapping = pd.read_csv("./data/df_mapping.csv")

    in_path = "./data/example_site/453024237.0005.2024.06.09.00.00.00.000.E.miniseed"

    st_E = obspy.read(in_path)
    tr_E = st_E.traces[0]
    st_N = obspy.read(in_path.replace(".E.", ".N."))
    tr_N = st_N.traces[0]
    st_Z = obspy.read(in_path.replace(".E.", ".Z."))
    tr_Z = st_Z.traces[0]

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

    # pandas...
    df = pd.DataFrame(
        {
            # "time": times,
            "vert": tr_Z.data,
            "north": tr_N.data,
            "east": tr_E.data,
        }
    )

    ft = np.fft.fftn(
        [df["vert"][night_inds], df["east"][night_inds], df["north"][night_inds]],
        axes=[1],
    )

    freqs = np.fft.fftfreq(len(times[night_inds]), d=delta)
    freqs_shift = np.fft.fftshift(freqs)
    freqs_mag = np.sqrt(
        np.sum(np.array([freqs_shift, freqs_shift, freqs_shift]) ** 2, axis=0)
    )

    ft_filt = ft.copy()
    ft_filt[:, freqs_mag > 2.5] = 0

    ts = np.fft.ifftn(ft_filt, axes=[1])


def select_time_slice():
    # get fourier transform
    # stats for spike distribution
    pass


def convert_miniseed_to_parquet(
    in_path=r"/home/gilbert_lab/Whitehorse_ANT/Whitehorse_ANT/",
    out_path=r"./results/timeseries/raw/",
):
    """
    Loop over raw timeseries miniseed files from smart solo geophone.
    Consolodate 3 components (vert, north, east) into one file.
    Determine station and date from miniseed and save to corresponding path.

    :param in_path: path of directory containing miniseed files
    """
    # Loop over raw miniseed files
    make_output_folder(out_path)
    full_timeseries_stats = {}
    for file_name in os.listdir(in_path):
        print(file_name)
        if ".E." not in file_name:
            continue
        # read in miniseed
        stream_east = read(in_path + file_name, format="mseed")
        stream_north = read(in_path + file_name.replace(".E.", ".N."), format="mseed")
        stream_vert = read(in_path + file_name.replace(".E.", ".Z."), format="mseed")

        # validate input file

        if not np.all(
            np.array([len(stream_east), len(stream_north), len(stream_vert)]) == 1
        ):
            raise ValueError

        trace_east = stream_east.traces[0]
        trace_north = stream_north.traces[0]
        trace_vert = stream_vert.traces[0]

        # using times from only east component... validate somehow?

        dates = [d.datetime for d in trace_east.times(type="utcdatetime")]
        # time passed, used in raydec
        times = trace_east.times()
        times -= times[0]

        # get station
        station = trace_east.stats["station"]

        # also get station
        if station not in full_timeseries_stats:
            full_timeseries_stats[station] = {}
        # get stats for whole timeseries
        full_timeseries_stats[station][date] = get_timeseries_stats(vert, north, east)

        # save miniseed file info
        east, north, vert = trace_east.data, trace_north.data, trace_vert.data
        # could save these stats later
        """
        # get serial number and day number from miniseed filename
        start_date, sampling_rate, delta, npts = (
            trace_east.stats["starttime"],
            trace_east.stats["sampling_rate"],
            trace_east.stats["delta"],
            trace_east.stats["npts"],            
        )
        """
        magnitude = np.sqrt(vert**2 + north**2 + east**2)

        df_timeseries = pd.DataFrame(
            {
                "date": dates,
                "time": times,
                "vert": vert,
                "north": north,
                "east": east,
                "magnitude": magnitude,
            },
        )

        # write to parquet

        # make output folders for diff stations, with date as filename

        start_date = str(trace_east.stats["starttime"]).split("T")[0]
        # save with parquet to compress
        make_output_folder(out_path + station)
        df_timeseries.to_parquet(out_path + station + "/" + start_date + ".parquet")

        # compare/save file size difference
        # 212 GB for miniseed


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
