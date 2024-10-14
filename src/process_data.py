from obspy import read
import numpy as np
from bs4 import BeautifulSoup
import datetime
import pandas as pd
import os
from raydec import raydec
from utils import make_output_folder
from dateutil import tz


def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False


def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def is_date(val):
    try:
        datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
        return True
    except ValueError:
        return False


# PARSING XML


def xml_to_dict(contents, include):
    # recursively loop over xml to make dictionary.
    # parse station data

    results_dict = {}
    for c in contents:
        if (
            hasattr(c, "name")
            and c.name is not None
            and c.name in include
            and c.name not in results_dict
        ):
            results_dict[c.name] = []
        else:
            continue
        if not hasattr(c, "contents"):
            continue

        if len(c.contents) == 1:
            result = c.contents[0]
            if is_int(result):
                result = int(result)
            elif is_float(result):
                result = float(result)
            elif is_date(result):
                result = datetime.datetime.strptime(result, "%Y-%m-%dT%H:%M:%S")
        elif c.contents is not None:
            result = xml_to_dict(c.contents, include)

        if c.name == "Channel":
            results_dict[c.name].append(result)
        else:
            results_dict[c.name] = result

    return results_dict


def parse_xml(save=True):
    """
    loop over all tags,
    save unique tags as a single value
    recuresively loop over tags with multiple values and add to dictionary
    figure out which values change between stations
    save stations to dataframe -> csv
    """
    path = "./data/FDSN_Information.xml"

    with open(path, "r") as f:
        file = f.read()

    soup = BeautifulSoup(file, "xml")

    # get unit information from Network

    results = {}
    for d in soup.descendants:
        if hasattr(d, "name") and d.name not in ["FDSNStationXML", "Network"]:
            if d.name not in results:
                results[d.name] = []
            results[d.name].append(d.text)

    all_stations = {}
    for k, v in results.items():
        unique_vals = np.unique(v)
        if len(unique_vals) == 1:
            all_stations[k] = unique_vals[0]

    remaining_vars = set(results.keys()) - set(all_stations.keys())
    remaining_vars.remove("Site")
    remaining_vars.remove("Channel")

    stations = {}
    for s in soup.find_all("Station"):
        if s is not None and s.find("Site") is not None:
            site = s.find("Site").find("Name").text
            # maybe save serial number from channels
            # channels = s.find_all("Channel")
            # channels = [xml_to_dict(c.contents, remaining_vars) for c in channels]

            stations[site] = xml_to_dict(s.contents, remaining_vars)
            # stations[site]["Channels"] = channels

    # convert dictionary to dataframe and save stations as csv
    stations_dict = {
        "Site": [],
        "Latitude": [],
        "Longitude": [],
        "Elevation": [],
        "CreationDate": [],
    }

    for site, attrib in stations.items():
        stations_dict["Site"].append(site)
        for key, value in attrib.items():
            stations_dict[key].append(value)

    if save:
        pd.DataFrame(stations_dict).to_csv("./data/parsed_xml.csv")


#
# PARSING STATION DATA
#


def get_file_information():
    directory = r"./../../gilbert_lab/Whitehorse_ANT/"

    # iterate over files in directory
    data_dict = {}
    for file_name in os.listdir(directory):
        if not file_name.endswith(".E.miniseed"):
            continue

        # read in data
        stream_east = read(directory + file_name, format="mseed")
        trace_east = stream_east.traces[0]
        station = int(trace_east.stats["station"])

        if len(stream_east) != 1:
            raise ValueError

        data_dict[file_name] = {
            "station": station,
        }
    df = pd.DataFrame(data_dict)
    df.to_csv("./data/file_information.csv")


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

        # dates = trace_east.times(type="matplotlib")
        dates = trace_east.times(type="utcdatetime")
        times = trace_east.times()
        times -= times[0]

        east, north, vert = trace_east.data, trace_north.data, trace_vert.data
        start_date, sampling_rate, sample_spacing = (
            trace_east.stats["starttime"],
            trace_east.stats["sampling_rate"],
            trace_east.stats["delta"],
        )

        # time_slice_inds = get_time_slice(start_date, times, east, north, vert)

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

        df = df[np.all(np.array([hours >= 20, hours <= 8]), axis=0)]

        # *** make sure the spacing is correct and gaps have nans
        name = str(start_date).split("T")[0] + ".csv"
        make_output_folder(output_dir)
        make_output_folder(output_dir + "/" + str(station) + "/")
        # write station df to csv
        df.to_csv(output_dir + "/" + str(station) + "/" + name)


def process_stations(directory=r"./../../gilbert_lab/Whitehorse_ANT/"):
    # save each station to a separate folder...
    # input station list and file list to save

    file_mapping = pd.read_csv("./data/file_information.csv", index_col=0).T

    # file_mapping.drop(0, axis=1)
    # file_mapping.drop("Unnamed: 0", axis=1)
    # file_mapping = file_mapping.T
    # file_names = file_mapping.iloc[0]
    # stations = file_mapping.iloc[1]
    stations = file_mapping["station"].values
    unique_stations = np.unique(stations)

    s = unique_stations[0]
    file_names = [file_mapping[file_mapping["station"] == s].index for s in stations]
    print(s, file_names[0])
    slice_station_data(s, file_names[0], directory)

    print("done")


def get_ellipticity(
    station, date, fmin=0.8, fmax=40, fsteps=300, cycles=10, dfpar=0.1, len_wind=60
):
    # loop over saved time series files
    # raydec
    # number of windows based on size of slice
    dir_in = "./results/timeseries/" + str(station) + "/" + date + ".csv"
    df_in = pd.read_csv(dir_in).dropna()
    if len(df_in["times"]) < 1:
        return None

    times = df_in["times"].dropna().values
    times -= times[0]
    n_wind = int(np.round(times[-1] / len_wind))

    freqs, ellips = raydec(
        vert=df_in["vert"],
        north=df_in["north"],
        east=df_in["east"],
        time=times,
        fmin=fmin,
        fmax=fmax,
        fsteps=fsteps,
        cycles=cycles,
        dfpar=dfpar,
        nwind=n_wind,
    )

    df = pd.DataFrame(ellips.T, columns=freqs[:, 0])
    return df


def remove_spikes(df_timeseries, max_amplitude):
    """
    remove spikes from timeseries data.
    values over a certain amplitude
    or
    LTA/STA
    """

    df_timeseries["outliers"] = (np.abs(df_timeseries["vert"]) >= max_amplitude).astype(
        int
    )
    return df_timeseries


def remove_outliers(df_raydec, scale_factor):
    """
    remove outliers from phase dispersion. windows with values further than 3 std from mean
    """
    mean = np.mean(df_raydec, axis=1)
    std = np.std(df_raydec, axis=1)

    # diff_from_mean = np.abs(mean - df_raydec)
    diff_from_mean = df_raydec.sub(df_raydec.mean(axis=1), axis=0)

    outlier_inds = np.any(diff_from_mean.T > scale_factor * std, axis=1)
    outlier_inds = outlier_inds.astype(int).rename("outliers").to_frame().T
    df_raydec = pd.concat([df_raydec, outlier_inds])

    mean_inds = df_raydec.loc["outliers"] == 0
    mean = np.nanmean(df_raydec.loc[:, mean_inds], axis=1)
    df_raydec["mean"] = mean

    return df_raydec


if __name__ == "__main__":
    """
    run from terminal
    """
    process_stations()
    # parse_xml()
    # get_file_information()
    # get_ellipticity(24952)
    process_stations()
