from obspy import read
import numpy as np
import datetime
import pandas as pd
from utils import make_output_folder
from dateutil import tz
import sys


###### TIMESERIES PROCESSING ######


def get_time_slice(start_date, time_passed):
    # shift to be in correct time zone
    # Convert time zone
    start_date = start_date.datetime.astimezone(tz.gettz("Canada/Yukon"))
    dates = [start_date + datetime.timedelta(seconds=s) for s in time_passed]
    hours = np.array([d.hour for d in dates])

    inds = (hours >= 2) and (hours <= 4)

    print(inds.shape, np.sum(inds))

    # look at night-time hours and find quietist(?) (would we always want quietist...?) consecutive 3h?

    # check spacing and nans

    return inds


def slice_station_data(station, file_names, input_dir, output_dir="./timeseries/"):
    """"""
    # iterate over files in directory
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

        # *** can probably make this more efficient... ***
        df["dates"] = df["dates"].apply(lambda d: d.datetime)
        df["dates"] = df["dates"].dt.tz_localize(datetime.timezone.utc)
        df["dates"] = df["dates"].dt.tz_convert(tz.gettz("Canada/Yukon"))

        hours = np.array([d.hour for d in df["dates"]])

        df = df[np.any(np.array([hours >= 20, hours <= 8]), axis=0)]

        # *** make sure the spacing is correct and gaps have nans
        name = str(start_date).split("T")[0] + ".csv"
        make_output_folder(output_dir)
        make_output_folder(output_dir + "/" + str(station) + "/")
        # write station df to csv
        df.to_csv(output_dir + "/" + str(station) + "/" + name)


def remove_spikes(df_timeseries, max_amplitude):
    """
    remove spikes from timeseries data.
    values over a certain amplitude
    or
    LTA/STA
    """

    df_timeseries["outliers"] = (
        np.abs(df_timeseries["vert"].values) >= max_amplitude
    ).astype(int)
    return df_timeseries


def find_quiet_times():
    pass


def remove_timeseries_edges():
    pass


def process_station_timeseries(ind, directory=r"./data/Whitehorse_ANT/"):
    # save each station to a separate folder...
    # input station list and file list to save

    file_mapping = pd.read_csv("./data/file_information.csv", index_col=0).T
    stations = file_mapping["station"].values
    unique_stations = np.unique(stations)

    s = unique_stations[ind]
    file_names = file_mapping[file_mapping["station"] == s].index
    slice_station_data(s, file_names, directory)

    print("done")


if __name__ == "__main__":
    """
    run from terminal
    """

    # #SBATCH --array=1-32 #838
    # python src/process_data.py $SLURM_ARRAY_TASK_ID
    # sbatch slice_timeseries_job.slurm

    ind = int(sys.argv[1])

    # process_station_ellipticity(ind)
    sensitivity_test(ind)
