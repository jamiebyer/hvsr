from obspy import read
import numpy as np
import datetime
import pandas as pd
from src.utils.utils import make_output_folder
from dateutil import tz
import sys
import os
import pyarrow.parquet as pq


###### TIMESERIES PROCESSING ######


def get_full_timeseries_stats():

    stats = {}
    path = "/run/media/jbyer/Backup/raw/"
    for station in os.listdir(path):
        for date in os.listdir(path + station):
            df = pd.read_csv(path + station + "/" + date)
            magnitude = np.sqrt(df["vert"] ** 2 + df["north"] ** 2 + df["east"])
            stats[station + "/" + date.replace(".csv", "")] = {
                "full_mean": magnitude.mean(),
                "full_std": magnitude.std(),
                "vert_mean": df["vert"].mean(),
                "vert_std": df["vert"].std(),
                "north_mean": df["north"].mean(),
                "north_std": df["north"].std(),
                "east_mean": df["east"].mean(),
                "east_std": df["east"].std(),
            }

    pd.DataFrame(stats).to_csv("./results/timeseries/stats.csv")


def label_spikes():
    path = "./results/timeseries/"
    # read in timeseries stats
    stats = pd.read_csv(path + "stats.csv", index_col=0)
    # read in timeseries slice
    for station in os.listdir(path):
        for date in os.listdir(path + "/" + station):
            if not os.path.isfile(path + "/" + station + "/" + date):
                continue
            df = pd.read_parquet(path + "/" + station + "/" + date)
            magnitude = np.sqrt(df["vert"] ** 2 + df["north"] ** 2 + df["east"] ** 2)
            # print(stats[station + "/" + date.replace(".parquet", "")])
            std = stats[station + "/" + date.replace(".parquet", "")]["full_std"]
            if np.all(np.isnan(std)):
                continue

            df["magnitude"] = magnitude
            df["spikes"] = magnitude >= (3 * std)

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


def slice_station_data(station, input_dir, output_dir="./timeseries/"):
    """
    station: station to save timeseries
    file_names: list of files for station to save
    input_dir
    output_dir: where to save timeseries

    *** needs file_information file ***
    """

    # get mapping between station and files with station info
    file_mapping = pd.read_csv("./data/file_information.csv", index_col=0).T

    file_names = file_mapping[file_mapping["station"] == s].index

    # iterate over station files
    for file_name in file_names:
        # read in data
        stream_east = read(input_dir + file_name, format="mseed")
        stream_north = read(input_dir + file_name.replace(".E.", ".N."), format="mseed")
        stream_vert = read(input_dir + file_name.replace(".E.", ".Z."), format="mseed")

        if not np.all(
            np.array([len(stream_east), len(stream_north), len(stream_vert)]) == 1
        ):
            raise ValueError

        trace_east = stream_east.traces[0]
        trace_north = stream_north.traces[0]
        trace_vert = stream_vert.traces[0]

        dates = trace_east.times(type="utcdatetime")
        # time passed, used in raydec
        times = trace_east.times()
        times -= times[0]

        east, north, vert = trace_east.data, trace_north.data, trace_vert.data
        start_date, sampling_rate, sample_spacing = (
            trace_east.stats["starttime"],
            trace_east.stats["sampling_rate"],
            trace_east.stats["delta"],
        )

        df = pd.DataFrame(
            {
                "dates": dates,
                "times": times,
                "vert": vert,
                "north": north,
                "east": east,
            },
        )

        df = get_time_slice(df)

        # making output paths
        name = str(start_date).split("T")[0] + ".csv"
        make_output_folder(output_dir)
        make_output_folder(output_dir + "/" + str(station) + "/")
        # save station timeseries to csv
        df.to_csv(output_dir + "/" + str(station) + "/" + name)


def label_spikes_og(df_timeseries, spike_quartile=0.95):
    """
    remove spikes from timeseries data.
    values over a certain quartile threshold


    *** later may do LTA/STA ***
    """

    # for comp in ["vert", "north", "east"]:
    #    df_timeseries.clip(upper=df_timeseries[comp].quantile(spike_quartile), axis=1)

    df_timeseries["outliers"] = np.any(
        df_timeseries[["vert", "north", "east"]].quantile(spike_quartile), axis=0
    )

    # df_timeseries["outliers"] = (
    #     np.abs(df_timeseries["vert"].values) >= max_amplitude
    # ).astype(int)

    return df_timeseries


def clean_timeseries_files(ind, in_dir=r"./results/timeseries/"):
    """
    ind:
    in_dir: input directory with saved timeseries csvs

    read in saved timeseries csv.
    label spikes based on full timeseries.
    slice a subset of night timeseries.
    saved with parquet for compression.
    """

    # loop over stations and files to make an indexable list
    file_path = []
    for station in os.listdir(in_dir):
        for file in os.listdir(in_dir + station):
            date = file.replace(".csv", "")
            file_path.append([station, date])

    # read in full night timeseries
    df_timeseries = pd.read_csv(
        in_dir + file_path[ind][0] + "/" + file_path[ind][1] + ".csv"
    )

    df_timeseries = df_timeseries.set_index(
        pd.DatetimeIndex(pd.to_datetime(df_timeseries["dates"], format="ISO8601"))
    )

    # label spikes (on full night(?), 20:00-8:00)
    df_timeseries = label_spikes(df_timeseries, spike_quartile=0.95)

    # clip to  22:00 - 5:00
    hours = df_timeseries.index.hour
    df_timeseries = df_timeseries[np.any(np.array([hours >= 22, hours <= 5]), axis=0)]

    # creating output folders
    f = file_path[ind]
    output_dir = in_dir + "clipped/"
    make_output_folder(output_dir)
    make_output_folder(output_dir + "/" + f[0] + "/")

    # save with parquet to compress
    df_timeseries.to_parquet(output_dir + "/" + f[0] + "/" + f[1] + ".parquet")


def process_station_timeseries(ind, in_dir=r"./data/Whitehorse_ANT/"):
    """
    save a station
    """
    # save each station to a separate folder...
    # input station list and file list to save

    file_mapping = pd.read_csv("./data/file_information.csv", index_col=0).T
    stations = file_mapping["station"].values
    unique_stations = np.unique(stations)
    s = unique_stations[ind]

    slice_station_data(s, in_dir)


if __name__ == "__main__":
    """
    run from terminal
    """

    # get_full_timeseries_stats()
    label_spikes()
