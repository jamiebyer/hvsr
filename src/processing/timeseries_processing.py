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



def get_timeseries_stats(df_timeseries, hourly):
    """
    """
    # stats for subsection (hourly? so different times can be picked later)
    # get stats
    magnitude, vert, north, east = df_timeseries["magnitude"], df_timeseries["vert"], df_timeseries["north"], df_timeseries["east"]
    if hourly:
        hours = np.array([d.hour for d in df_timeseries["date"]])

        length, full_mean, full_std, vert_mean, vert_std, north_mean, north_std, east_mean, east_std = [], [], [], [], [], [], [], [], []

        for h in np.unique(hours):
            df = df_timeseries[np.any(np.array([hours == h]), axis=0)]

            length.append(len(magnitude))
            full_mean.append(float(magnitude.mean()))
            full_std.append(float(magnitude.std()))
            vert_mean.append(float(vert.mean()))
            vert_std.append(float(vert.std()))
            north_mean.append(float(north.mean()))
            north_std.append(float(north.std()))
            east_mean.append(float(east.mean()))
            east_std.append(float(east.std()))
    else:
        length = len(magnitude)
        full_mean = float(magnitude.mean())
        full_std = float(magnitude.std())
        vert_mean = float(vert.mean())
        vert_std = float(vert.std())
        north_mean = float(north.mean())
        north_std = float(north.std())
        east_mean = float(east.mean())
        east_std = float(east.std())

    stats = {
        "length": length,
        "full_mean": full_mean,
        "full_std": full_std,
        "vert_mean": vert_mean,
        "vert_std": vert_std,
        "north_mean": north_mean,
        "north_std": north_std,
        "east_mean": east_mean,
        "east_std": east_std,
    }

    return stats



def label_spikes(df_timeseries, station, date):
    """
    remove spikes from timeseries data.

    *** later may do LTA/STA ***
    """

    df_stats = pd.read_csv(r"./results/timeseries/stats/full_timeseries.csv", index_col=0)
    
    stats_string = df_stats.loc[[date],[station]].values[0][0]
    stats_string = stats_string.replace("(", "").replace(")", "").replace("np.float32", "").replace("{", "").replace("}", "").replace("'", "").replace(" ", "")
    
    stats = {}
    for s in stats_string.split(","):
        stats[s.split(":")[0]]= float(s.split(":")[1])

    if len(stats) > 0:
        df_timeseries["spikes"] = df_timeseries["magnitude"] >= 3 * stats["full_std"]
    else:
        df_timeseries["spikes"] = None
    # save labeled timeseries

    return df_timeseries


def get_time_slice(df):
    """
    df: timeseries df

    convert to correct time zone.
    get slice between hour limits  
    """
    # *** can probably make this more efficient... ***
    #df["date"] = df["date"].apply(lambda d: d.datetime)
    df["date"] = df["date"].apply(lambda d: pd.to_datetime(d))

    df["date"] = df["date"].dt.tz_localize(datetime.timezone.utc)
    df["date"] = df["date"].dt.tz_convert(tz.gettz("Canada/Yukon"))

    hours = np.array([d.hour for d in df["date"]])

    df = df[np.any(np.array([hours >= 20, hours <= 8]), axis=0)]
    return df


def get_clean_timeseries_slice(
        include_outliers, 
        in_path="./results/timeseries/raw/", 
        out_path="./results/timeseries/clipped/"
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
            df_timeseries = pd.read_parquet(in_path + station + "/" + file, engine="pyarrow")

            if include_outliers == False:
                # subset timeseries so only valid points are included
                df_timeseries = label_spikes(df_timeseries, station, date)

                df_timeseries = df_timeseries[df_timeseries["spikes"] == 0]
            
            timeseries_stats[station][date] = get_timeseries_stats(df_timeseries, hourly=True)

            # slice time
            df_timeseries = get_time_slice(df_timeseries)
            # making output paths
            make_output_folder(out_path + station)
            df_timeseries.to_parquet(out_path + station + "/" + date + ".parquet")

    
    # save stats to df
    df_stats = pd.DataFrame(timeseries_stats)
    df_stats.to_csv("./results/timeseries/stats/timeseries_slice.csv")


    





