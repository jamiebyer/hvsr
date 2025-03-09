from obspy import read
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import os
import sys
from utils.utils import is_int, is_date, is_float
import matplotlib.pyplot as plt
import shutil
import string


####### PARSING XML ######


def xml_to_dict(contents, include):
    """
    contents:
    include: names of data to save

    recursively loop over xml to make dictionary.
    parse station data.
    """

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
                result = datetime.strptime(result, "%Y-%m-%dT%H:%M:%S")
        elif c.contents is not None:
            result = xml_to_dict(c.contents, include)

        if c.name == "Channel":
            results_dict[c.name].append(result)
        else:
            results_dict[c.name] = result

    return results_dict


def change_coords(df):
    coords_list = []
    for d in df:
        spl = d.split(" ")
        deg = spl[0]
        min = spl[1]
        sec = spl[2]
        dir = spl[3]

        coords = (float(deg) + float(min) / 60 + float(sec) / (60 * 60)) * (
            -1 if dir in ["W", "S"] else 1
        )
        coords_list.append(coords)
    return coords_list


def get_station_coords():
    df = pd.read_csv("./data/spreadsheets/stations_coords.csv")

    # read file names
    in_dir = r"./../../gilbert_lab/Whitehorse_ANT/Whitehorse_ANT/"
    # get serial and lat, lon for each file
    files = os.listdir(in_dir)

    out_files = []
    lats = []
    lons = []
    elevs = []
    complete = []

    df["Start_time (UTC)"] = pd.to_datetime(df["Start_time (UTC)"])
    df["End_time (UTC)"] = pd.to_datetime(df["End_time (UTC)"])

    # Whitehorse_ANT/453024025.0001.2024.06.06.18.04.52.000.E.miniseed
    file_split = [f.split(".") for f in files]

    serials, start_dates, end_dates = [], [], []
    for ind, f in enumerate(file_split):
        if "FDSN Information" in files[ind]:
            continue
        serial = int(f[0].replace("4530", ""))
        start_date = datetime(
            int(f[2]),
            int(f[3]),
            int(f[4]),
            int(f[5]),
            int(f[6]),
            int(f[7]),
        )
        end_date = datetime(
            int(f[2]),
            int(f[3]),
            int(f[4]),
            0,
            0,
            0,
        ) + timedelta(days=1)

        # select any recordings from that day
        rows = df.loc[
            (start_date < df["End_time (UTC)"])
            & (df["Start_time (UTC)"] < end_date)
            & (df["Serial"] == serial)
        ]
        if len(rows) < 1:
            continue

        lat = rows["GNSS_latitude"].values
        lon = rows["GNSS_longitude"].values
        elev = rows["GNSS_elevation"].values

        # mark the files with incomplete data
        # sections less than 24h
        recording_time = rows["End_time (UTC)"].values - rows["Start_time (UTC)"].values

        complete += list(recording_time >= np.timedelta64(24, "h")) and len(rows) * [
            start_date.hour == 0 & start_date.minute == 0 & start_date.second == 0
        ]

        serials += len(rows) * [serial]
        start_dates += len(rows) * [start_date]
        end_dates += len(rows) * [end_date]
        lats += list(lat)
        lons += list(lon)
        elevs += list(elev)
        out_files += len(rows) * [files[ind]]

    # convert coords to lats/lons
    new_lats = change_coords(lats)
    new_lons = change_coords(lons)

    df_file_mapping = pd.DataFrame(
        {
            "file": out_files,
            "serial": serials,
            "start_date": start_dates,
            "end_date": end_dates,
            "lat": new_lats,
            "lon": new_lons,
            "elev": elevs,
            "complete": complete,
        }
    )

    # give site name to files with same lat, lon, elev (within threshold?)

    # group rows by coords/site-- within threshold?
    # where lat and lon are unique

    # df_file_mapping["round_lat"] = df_file_mapping["lat"].round(3)
    # df_file_mapping["round_lon"] = df_file_mapping["lon"].round(3)

    df_file_mapping["site"] = df_file_mapping.groupby(["lat", "lon"]).ngroup() + 1

    lats, lons = [], []
    for i in np.unique(df_file_mapping["site"]):
        d = df_file_mapping[df_file_mapping["site"] == i]
        lats.append(d["lat"].values[0])
        lons.append(d["lon"].values[0])

    plt.rcParams["figure.figsize"] = (20, 15)

    # y: site, x: date, lable:serial
    # for each serial, get full x and y
    serials = np.unique(df_file_mapping["serial"].values)

    [plt.axhline(y, c="grey", alpha=0.2) for y in np.arange(0, 80, 1)]
    for s in serials:
        subset = df_file_mapping[df_file_mapping["serial"] == s]
        subset = subset.sort_values(["start_date"])
        dates, sites = [], []
        for ind in range(len(subset)):
            s_date = subset["start_date"].values[ind]
            site = subset["site"].values[ind]

            if len(sites) > 0 and sites[-1] != site:
                if len(sites) > 2:
                    plt.text(dates[0], sites[0] + 0.1, s)
                    plt.plot(dates, sites, linestyle="solid", color="blue")

                dates, sites = [], []

            dates.append(s_date)
            # dates = [subset["start_date"].values[ind], subset["end_date"].values[ind]]
            # d_range = np.arange(subset["start_date"].values[ind], subset["end_date"].values[ind], timedelta(hours=2)).astype(datetime)
            # dates += list(d_range)
            sites += [site]

    plt.yticks(np.arange(0, 80, 1))
    # plt.legend()
    plt.savefig("./results/sites_timeseries.png")

    plt.clf()
    plt.scatter(lons, lats)
    for ind, s in enumerate(np.unique(df_file_mapping["site"])):
        plt.text(lons[ind], lats[ind], s)

    plt.xlim([-135.3, -134.9])
    plt.ylim([60.635, 60.84])

    plt.savefig("./results/sites.png")

    # remove where geophones were moved. (interval around if amplitude reaches max?)

    # save file mapping df

    # copy files to a new directory, sorted by site/coords


def get_station_file_mapping():
    in_df = pd.read_csv("./data/spreadsheets/stations_coords.csv")

    # read file names
    in_dir = r"./../../gilbert_lab/Whitehorse_ANT/Whitehorse_ANT/"
    # get serial and lat, lon for each file
    files = np.array(os.listdir(in_dir))

    in_df["Start_time (UTC)"] = pd.to_datetime(in_df["Start_time (UTC)"])
    in_df["End_time (UTC)"] = pd.to_datetime(in_df["End_time (UTC)"])

    # remove recordings shorter than two days
    site_inds = in_df["End_time (UTC)"] - in_df["Start_time (UTC)"] > np.timedelta64(
        72, "h"
    )
    out_df = in_df[site_inds]

    # convert coords to lats/lons
    out_df.loc[:, "GNSS_latitude"] = change_coords(out_df.loc[:, "GNSS_latitude"])
    out_df.loc[:, "GNSS_longitude"] = change_coords(out_df.loc[:, "GNSS_longitude"])

    # rounding nearby stations
    out_df.loc[:, "GNSS_latitude_rounded"] = out_df.loc[:, "GNSS_latitude"].astype(float).round(3)
    out_df.loc[:, "GNSS_longitude_rounded"] = out_df.loc[:, "GNSS_longitude"].astype(float).round(3)
    #out_df.loc[:, "GNSS_elevation"] = out_df.loc[:, "GNSS_elevation"].round(3)
    
    out_df.loc[:, "site"] = (out_df.groupby(["GNSS_latitude_rounded", "GNSS_longitude_rounded"]).ngroup()+1).values
    out_df.reset_index(inplace=True, drop=True)
    
    files = np.array(os.listdir(in_dir))
    values, counts = np.unique(out_df["site"], return_counts=True)
    for ind, v in enumerate(values):
        df_inds = out_df.index[out_df["site"] == v].values
        #print(df_inds)
        #print(out_df)

        site_name = str(v).rjust(2,"0")

        for c in range(counts[ind]):
            #site_path = str(v) + "." + str(c)
            if c == 0:
                suffix = ""
            else:
                suffix = string.ascii_uppercase[c]
            out_df.loc[df_inds[c], "site"] = site_name + suffix

    # map files to recordings
    # Whitehorse_ANT/453024025.0001.2024.06.06.18.04.52.000.E.miniseed
    start_dates, end_dates = [], []
    paths = []
    for _, recording in out_df.iterrows():
        # get files with this serial number, and in the correct date range
        serial = recording["Serial"]
        file_inds = np.char.startswith(files, "4530" + str(serial))

        # add all file paths between the start and ending date (remove first and last day/file)
        # update the start and end date in the df

        d = recording["Start_time (UTC)"]
        start_date = datetime(2024, d.month, d.day) + timedelta(days=1)
        d = recording["End_time (UTC)"]
        end_date = datetime(2024, d.month, d.day) - timedelta(days=1)

        start_dates.append(start_date)
        end_dates.append(end_date)

    out_df["Start_time (UTC)"] = start_dates
    out_df["End_time (UTC)"] = end_dates
    # out_df.loc[:, "paths"] = paths

    out_df.to_csv("./data/df_mapping.csv", index=False)


    
def organize_station_files():    
    mapping_dir = "./data/df_mapping.csv"
    df_mapping = pd.read_csv(mapping_dir)
    df_mapping["Start_time (UTC)"] = pd.to_datetime(df_mapping["Start_time (UTC)"])
    df_mapping["End_time (UTC)"] = pd.to_datetime(df_mapping["End_time (UTC)"])

    in_dir = r"./../../gilbert_lab/Whitehorse_ANT/Whitehorse_ANT/"
    # create folder for site
    out_dir = "./results/timeseries/sorted/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    files = np.array(os.listdir(in_dir))
    values, counts = np.unique(df_mapping["site"], return_counts=True)
    for ind, v in enumerate(values):
        #if v <= 50:
        #    continue
        site_list = df_mapping[df_mapping["site"] == v]
        site_name = str(v).rjust(2,"0")
        for c in range(counts[ind]):
            site_path = str(v) + "." + str(c)
            if len(site_list) == 1:
                suffix = ""
            else:
                suffix = string.ascii_uppercase[c]
            site_path = out_dir + site_name + suffix + "/"
        
            if not os.path.isdir(site_path):
                os.mkdir(site_path)

            site = site_list.iloc[c]
            serial = site["Serial"]
            site_inds = np.char.startswith(files, "4530" + str(serial))

            dates = pd.date_range(start=site["Start_time (UTC)"], end=site["End_time (UTC)"], inclusive="both")
            for d in dates:            
                # get filename that starts with serial and ends with date, all directions
                date = "2024."+ str(d.month).rjust(2,"0") + "." + str(d.day).rjust(2,"0")
                file_inds = site_inds & [date in f for f in files]
                for f in files[file_inds]:
                    # move 3 components to folder
                    #if not os.path.exists(site_path + f):
                    coord = f.split(".000.")[1]
                    shutil.copyfile(in_dir + f, site_path + site_name + suffix + "_" + date.replace(".", "-") + "." + coord)


def plot_station_schedule():
    in_dir = "./data/df_mapping.csv"
    df_mapping = pd.read_csv(in_dir)

    plt.rcParams["figure.figsize"] = (20, 15)

    df_mapping["Start_time (UTC)"] = pd.to_datetime(df_mapping["Start_time (UTC)"])
    df_mapping["End_time (UTC)"] = pd.to_datetime(df_mapping["End_time (UTC)"])

    #[plt.axhline(y, c="grey", alpha=0.2) for y in np.arange(70)]
    for ind, recording in df_mapping.iterrows():
        dates = pd.date_range(start=recording["Start_time (UTC)"], end=recording["End_time (UTC)"], inclusive="both")
        plt.plot(dates, len(dates)*[recording["site"]], linestyle = 'solid', color="blue")
    
    #plt.legend()
    plt.xlabel("date")
    plt.ylabel("site")
    plt.yticks(np.arange(0, 71, int(70/14)))
    plt.ylim([0, 70])

    xticks=pd.date_range(start=df_mapping["Start_time (UTC)"].min(), end=df_mapping["End_time (UTC)"].max(), periods=15, inclusive="both")

    #print(yticks.values)
    plt.xticks(xticks.values)
    
    plt.grid(True, color="grey", linestyle="-", alpha=0.2)

    plt.savefig("./results/sites_timeseries.png")

    df_mapping = df_mapping.drop_duplicates(subset=["site"])
    lons = df_mapping["GNSS_longitude"]
    lats = df_mapping["GNSS_latitude"]

    plt.clf()
    plt.scatter(lons, lats)
    for ind, recording in df_mapping.iterrows():
        plt.text(
            recording["GNSS_longitude"], recording["GNSS_latitude"], recording["site"]
        )

    plt.xlim([-135.3, -134.9])
    plt.ylim([60.635, 60.84])
    plt.xlabel("longitude")
    plt.ylabel("latitude")

    plt.savefig("./results/sites.png")

    # remove where geophones were moved. (interval around if amplitude reaches max?)

    # save file mapping df

    # copy files to a new directory, sorted by site/coords


def parse_xml():
    """
    Loop over station tags.
        - Save Site/Name
        - From each channel, save DataAvalibility, Latitude, Longitude, Elevation
        and Description/SerialNumber
        - Confirm the information is the same for all three channels


    for item in station:
        # contains: Latitude, Longitude, Elevation, Site, CreationDate, TotalNumberChannels, SelectedNumberChannels
        # Channels: EPZ, EPN, EPE
        for i in item:
            # contains: DataAvalibility, Latitude, Longitude, Elevation, Depth, Azimuth, Dip, Type, SampleRate, ClockDrift,
            # Sensor information
            for j in i:
                # contains: Description, Manufacturer, SerialNumber, InstrumentSensitivity, Stage
    """
    path = "./data/FDSN_Information.xml"

    with open(path, "r") as f:
        file = f.read()

    soup = BeautifulSoup(file, "xml")

    # ["SerialNumber", "Latitude", "Longitude", "Site"]

    results = {
        "site": [],
        "start_date": [],
        "end_date": [],
        "lat": [],
        "lon": [],
        "elev": [],
        "serial": [],
    }
    for station in soup.find_all("Station"):
        site = station.find("Site").find("Name").text
        start_dates = [d["start"] for d in station.find_all("Extent")]
        end_dates = [d["end"] for d in station.find_all("Extent")]
        lats = [l.text for l in station.find_all("Latitude")]
        lons = [l.text for l in station.find_all("Longitude")]
        elevs = [e.text for e in station.find_all("Elevation")]
        serials = [s.text for s in station.find_all("SerialNumber")]

        for prop in [start_dates, end_dates, lats, lons, elevs, serials]:
            if len(np.unique(prop, axis=0)) != 1:
                raise ValueError

        results["site"].append(site)
        results["start_date"].append(start_dates[0])
        results["end_date"].append(end_dates[0])
        results["lat"].append(lats[0])
        results["lon"].append(lons[0])
        results["elev"].append(elevs[0])
        results["serial"].append(serials[0])

    df = pd.DataFrame(results)
    df.to_csv("./results/xml_info.csv")
    print(df)


### STATION COORDS ###


def change_coords(df):
    coords_list = []
    for d in df:
        spl = d.split(" ")
        deg = spl[0]
        min = spl[1]
        sec = spl[2]
        dir = spl[3]

        coords = (float(deg) + float(min) / 60 + float(sec) / (60 * 60)) * (
            -1 if dir in ["W", "S"] else 1
        )
        coords_list.append(coords)

    return coords_list


def get_station_coords():
    df = pd.read_csv("./data/spreadsheets/stations_coords.csv")
    planned_df = pd.read_csv("./data/spreadsheets/deployment_coords.csv")

    # read file names
    in_dir = r"./../../gilbert_lab/Whitehorse_ANT/Whitehorse_ANT/"
    # get serial and lat, lon for each file
    files = os.listdir(in_dir)

    out_files = []
    lats = []
    lons = []
    elevs = []
    complete = []

    df["Start_time (UTC)"] = pd.to_datetime(df["Start_time (UTC)"])
    df["End_time (UTC)"] = pd.to_datetime(df["End_time (UTC)"])

    # Whitehorse_ANT/453024025.0001.2024.06.06.18.04.52.000.E.miniseed
    file_split = [f.split(".") for f in files]

    serials, start_dates, end_dates = [], [], []
    for ind, f in enumerate(file_split):
        if "FDSN Information" in files[ind]:
            continue
        serial = int(f[0].replace("4530", ""))
        start_date = datetime(
            int(f[2]),
            int(f[3]),
            int(f[4]),
            int(f[5]),
            int(f[6]),
            int(f[7]),
        )
        end_date = datetime(
            int(f[2]),
            int(f[3]),
            int(f[4]),
            0,
            0,
            0,
        ) + timedelta(days=1)

        rows = df.loc[
            (start_date < df["End_time (UTC)"])
            & (df["Start_time (UTC)"] < end_date)
            & (df["Serial"] == serial)
        ]
        if len(rows) < 1:
            continue

        lat = rows["GNSS_latitude"].values
        lon = rows["GNSS_longitude"].values
        elev = rows["GNSS_elevation"].values

        # mark the files with incomplete data
        # sections less than 24h
        recording_time = rows["End_time (UTC)"].values - rows["Start_time (UTC)"].values

        complete += list(recording_time >= np.timedelta64(24, "h")) and len(rows) * [
            start_date.hour == 0 & start_date.minute == 0 & start_date.second == 0
        ]

        serials += len(rows) * [serial]
        start_dates += len(rows) * [start_date]
        end_dates += len(rows) * [end_date]
        lats += list(lat)
        lons += list(lon)
        elevs += list(elev)
        out_files += len(rows) * [files[ind]]

    # convert coords to lats/lons
    new_lats = change_coords(lats)
    new_lons = change_coords(lons)

    df_file_mapping = pd.DataFrame(
        {
            "file": out_files,
            "serial": serials,
            "start_date": start_dates,
            "end_date": end_dates,
            "lat": new_lats,
            "lon": new_lons,
            "elev": elevs,
            "complete": complete,
        }
    )

    # give site name to files with same lat, lon, elev (within threshold?)

    # group rows by coords/site-- within threshold?
    # where lat and lon are unique

    df_file_mapping["round_lat"] = df_file_mapping["lat"].round(3)
    df_file_mapping["round_lon"] = df_file_mapping["lon"].round(3)

    # df_file_mapping["round_elev"] = df_file_mapping["elev"].round(0)

    df_file_mapping["site"] = (
        df_file_mapping.groupby(["round_lat", "round_lon"]).ngroup() + 1
    )

    # print(len(np.unique(df_file_mapping["site"])))
    # print(len(np.unique(df_file_mapping.loc[df_file_mapping["complete"] == True]["site"])))
    lats, lons = [], []
    for i in np.unique(df_file_mapping["site"]):
        d = df_file_mapping[df_file_mapping["site"] == i]
        lats.append(d["lat"].values[0])
        lons.append(d["lon"].values[0])

    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (20, 15)

    # y: site, x: date, lable:serial
    # for each serial, get full x and y
    serials = np.unique(df_file_mapping["serial"].values)

    [plt.axhline(y, c="grey", alpha=0.2) for y in np.arange(0, 80, 1)]
    for s in serials:
        subset = df_file_mapping[df_file_mapping["serial"] == s]
        subset = subset.sort_values(["start_date"])
        dates, sites = [], []
        for ind in range(len(subset)):
            s_date = subset["start_date"].values[ind]
            site = subset["site"].values[ind]

            if len(sites) > 0 and sites[-1] != site:
                if len(sites) > 2:
                    plt.text(dates[0], sites[0] + 0.1, s)
                    plt.plot(dates, sites, linestyle="solid", color="blue")

                dates, sites = [], []

            dates.append(s_date)
            # dates = [subset["start_date"].values[ind], subset["end_date"].values[ind]]
            # d_range = np.arange(subset["start_date"].values[ind], subset["end_date"].values[ind], timedelta(hours=2)).astype(datetime)
            # dates += list(d_range)
            sites += [site]

    plt.yticks(np.arange(0, 80, 1))
    # plt.legend()
    plt.savefig("./results/sites_timeseries.png")

    plt.clf()
    plt.scatter(lons, lats)
    for ind, s in enumerate(np.unique(df_file_mapping["site"])):
        plt.text(lons[ind], lats[ind], s)

    plt.xlim([-135.3, -134.9])
    plt.ylim([60.635, 60.84])

    plt.savefig("./results/sites.png")


def get_file_information():
    """
    Iterate over files in data directory.
    Create mapping between files and the station they have data for.
    """
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


def get_station_positions(
    ind, in_path=r"./data/Whitehorse_ANT/", out_path=r"./results/timeseries/raw/"
):
    """
    Loop over raw timeseries miniseed files from smart solo geophone.
    Consolodate 3 components (vert, north, east) into one file.
    Determine station and date from miniseed and save to corresponding path.

    :param in_path: path of directory containing miniseed files
    """
    # Loop over raw miniseed files
    # make_output_folder(out_path)

    file_name = os.listdir(in_path)[ind]

    if ".E." not in file_name:
        return
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
    print(trace_east.stats)

    # using times from only east component... validate somehow?

    dates = [d.datetime for d in trace_east.times(type="utcdatetime")]
    # time passed, used in raydec
    times = trace_east.times()
    times -= times[0]

    # get station
    station = trace_east.stats["station"]
