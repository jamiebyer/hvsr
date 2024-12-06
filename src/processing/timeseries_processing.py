from obspy import read
import numpy as np
import datetime
import pandas as pd
from utils.utils import make_output_folder, create_file_list
from dateutil import tz
import sys
import os
import pyarrow.parquet as pq


###### TIMESERIES PROCESSING ######

def convert_miniseed_to_parquet(in_path=r"./", out_path=r"./"):
    """
    Loop over raw timeseries miniseed files from smart solo geophone.
    Consolodate 3 components (vert, north, east) into one file.
    Determine station and date from miniseed and save to corresponding path.

    :param in_path: path of directory containing miniseed files
    """
    # Loop over raw miniseed files
    full_timeseries_stats = {}

    for file_name in os.listdir(in_path):
        if ".E." not in file:
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

        dates = trace_east.times(type="utcdatetime")
        # time passed, used in raydec
        times = trace_east.times()
        times -= times[0]

        # also get station
        if station not in full_timeseries_stats:
            full_timeseries_stats[station] = {}
        # get stats for whole timeseries
        full_timeseries_stats[station][date] = get_timeseries_stats(vert, north, east)

        # save miniseed file info
        east, north, vert = trace_east.data, trace_north.data, trace_vert.data
        start_date, sampling_rate, sample_spacing = (
            trace_east.stats["starttime"],
            trace_east.stats["sampling_rate"],
            trace_east.stats["delta"],
        )

        df = pd.DataFrame(
            {
                "date": dates,
                "time": times,
                "vert": vert,
                "north": north,
                "east": east,
            },
        )

        # compare/save file size difference
        # write to parquet

        # folders for diff stations, with date as filename.
        # save with parquet to compress
        df_timeseries.to_parquet(output_dir + "/" + station + "/" + date + ".parquet")
    # save stats to df
    df = pd.DataFrame(full_timeseries_stats)
    df.to_csv(output_dir + "/full_timeseries_stats.csv")


def get_timeseries_stats(include_outliers, in_path=r"./results/timeseries/raw/", out_path=r"./results/timeseries/stats/", out_file_name="timeseries_stats"):
    """
    """

    timeseries_stats = {}
    for station in os.listdir(in_path):
        timeseries_stats[station] = {}
        for file in os.listdir(in_path + station):
            date = file.replace(".parquet", "")
            df_timeseries = pd.read_parquet(in_path + station + "/" + file, engine="pyarrow")
            if include_outliers == False:
                # subset timeseries so only valid points are included
                df_timeseries = df_timeseries[df_timeseries["outliers"] == 0]

            # get stats
            magnitude, vert, north, east = df_timeseries["magnitude"], df_timeseries["vert"], df_timeseries["north"], df_timeseries["east"]
            timeseries_stats[station][date] = {
                "length": np.size(magnitude),
                "full_mean": magnitude.mean(),
                "full_std": magnitude.std(),
                "vert_mean": np.mean(vert),
                "vert_std": np.std(vert),
                "north_mean": np.mean(north),
                "north_std": np.std(north),
                "east_mean": np.mean(east),
                "east_std": np.std(east),
            }

    # save stats to df
    df = pd.DataFrame(timeseries_stats)
    df.to_csv(output_dir + "/" + out_file_name + ".csv")


# move to main...
def save_full_timeseries_stats():
    get_timeseries_stats(include_outliers=False, in_path=r"./results/timeseries/raw/", out_path=r"./results/timeseries/stats/", out_file_name="full_timeseries_cleaned")
    get_timeseries_stats(include_outliers=True, in_path=r"./results/timeseries/raw/", out_path=r"./results/timeseries/stats/", out_file_name="full_timeseries")

def save_timeseries_slice_stats():
    # hourly for time slice?
    get_timeseries_stats(include_outliers, in_path=r"./results/timeseries/clipped/", out_path=r"./results/timeseries/stats/", out_file_name="sliced_timeseries_cleaned")


def label_spikes(ind, std, in_path=r"./results/timeseries/raw/"):
    """
    remove spikes from timeseries data.

    *** later may do LTA/STA ***
    """

    df_stats = pd.read_csv(r"./results/timeseries/stats/full_timeseries_cleaned", index_col=0)
    
    # read in timeseries slice
    station, date = create_file_list(ind, in_path)
    std = df_stats[station][date]["full_std"]

    if not np.all(np.isnan(std)):
        df["magnitude"] = magnitude
        df["spikes"] = magnitude >= 3 * std

        df.to_parquet(path + "/" + station + "/" + date)



def get_time_slice(df):
    """
    df: timeseries df

    convert to correct time zone.
    get slice between hour limits
    """
    # *** can probably make this more efficient... ***
    df["dates"] = df["dates"].apply(lambda d: d.datetime)
    df["dates"] = df["dates"].dt.tz_localize(datetime.timezone.utc)
    df["dates"] = df["dates"].dt.tz_convert(tz.gettz("Canada/Yukon"))

    hours = np.array([d.hour for d in df["dates"]])

    df = df[np.any(np.array([hours >= 20, hours <= 8]), axis=0)]
    return df


def get_clean_timeseries_slice(in_path, out_path):
    """
    Remove outliers/spikes and slice timeseries. Save timeseries stats.

    :param in_path: path to full timeseries parquet files.
    """

    # remove outliers

    # read in stats for full timeseries

    label_spikes(ind, std, in_path=r"./results/timeseries/raw/")


    # slice time

    # stats for subsection (hourly? so different times can be picked later)


    df = get_time_slice(df)

    # making output paths
    name = str(start_date).split("T")[0] + ".csv"
    make_output_folder(output_dir)
    make_output_folder(output_dir + "/" + str(station) + "/")
    # save station timeseries to csv
    df.to_csv(output_dir + "/" + str(station) + "/" + name)



