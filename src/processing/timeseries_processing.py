from obspy import read
import numpy as np
import datetime
import pandas as pd
from utils.utils import make_output_folder
from dateutil import tz
import sys
import os
import pyarrow.parquet as pq


###### TIMESERIES PROCESSING ######


# public method
def convert_miniseed_to_parquet(in_path=r"/home/gilbert_lab/Whitehorse_ANT/Whitehorse_ANT/", out_path=r"./results/timeseries/raw/"):
    """
    Loop over raw timeseries miniseed files from smart solo geophone.
    Consolodate 3 components (vert, north, east) into one file.
    Determine station and date from miniseed and save to corresponding path.

    :param in_path: path of directory containing miniseed files
    """
    # Loop over raw miniseed files
    make_output_folder(out_path)
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



# private method
def get_full_timeseries_stats(in_path=r"/run/media/jbyer/Backup/raw/"):
    """
    in_path: path with timeseries data
    """

    stats = {}
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

# public method
def get_clean_timeseries_slice(in_path):
    """
    :param in_path: path to full timeseries parquet files.
    """

    # get stats for whole timeseries

    # remove outliers
    # get stats for cleaned timeseries
    # slice time
    # stats for subsection (hourly? so different times can be picked later)




    df = get_time_slice(df)

    # making output paths
    name = str(start_date).split("T")[0] + ".csv"
    make_output_folder(output_dir)
    make_output_folder(output_dir + "/" + str(station) + "/")
    # save station timeseries to csv
    df.to_csv(output_dir + "/" + str(station) + "/" + name)





def clean_timeseries_files(ind, in_dir=r"./results/timeseries/"):
    """
    ind:
    in_dir: input directory with saved timeseries csvs

    read in saved timeseries csv.
    label spikes based on full timeseries.
    slice a subset of night timeseries.
    saved with parquet for compression.
    """

    # read in full night timeseries
    df_timeseries = pd.read_csv(
        in_dir + file_path[ind][0] + "/" + file_path[ind][1] + ".csv"
    )

    df_timeseries = df_timeseries.set_index(
        pd.DatetimeIndex(pd.to_datetime(df_timeseries["dates"], format="ISO8601"))
    )

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


def label_spikes(ind):
    """
    remove spikes from timeseries data.
    values over a certain quartile threshold


    *** later may do LTA/STA ***
    """
    # originially using quantile:
    """
    df_timeseries["outliers"] = np.any(
        df_timeseries[["vert", "north", "east"]].quantile(spike_quartile), axis=0
    )
    """


    path = "./results/timeseries/clipped/"
    # read in timeseries stats
    stats = pd.read_csv("./results/timeseries/stats.csv", index_col=0)
    # read in timeseries slice
    df, station, date = create_file_list(ind)
    magnitude = np.sqrt(df["vert"] ** 2 + df["north"] ** 2 + df["east"] ** 2)
    # print(stats[station + "/" + date.replace(".parquet", "")])
    std = stats[station + "/" + date.replace(".parquet", "")]["full_std"]
    if not np.all(np.isnan(std)):
        print("\nstd", std)
        df["magnitude"] = magnitude
        df["spikes"] = magnitude >= 3 * std
        print("spikes", np.sum(df["spikes"]), "/", len(df["spikes"]))

        df.to_parquet(path + "/" + station + "/" + date)









# dont think i need this anymore:

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


