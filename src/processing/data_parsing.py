from obspy import read
import numpy as np
from bs4 import BeautifulSoup
import datetime
import pandas as pd
import os
import sys
from utils.utils import is_int, is_date, is_float


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
                result = datetime.datetime.strptime(result, "%Y-%m-%dT%H:%M:%S")
        elif c.contents is not None:
            result = xml_to_dict(c.contents, include)

        if c.name == "Channel":
            results_dict[c.name].append(result)
        else:
            results_dict[c.name] = result

    return results_dict


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


###### GET MAPPING OF STATION AND FILES ######


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


def split_temperature_csv():
    path = "./data/other_data/Temperature_20240828163333.csv"
    station_rows = {}
    header, station = None, None
    # determine which rows to read for each station
    with open(path, "r") as file:
        for line_number, line in enumerate(file.readlines()):
            if line.startswith("#"):
                if line.startswith("#Format: "):
                    header = line.removeprefix("#Format: ")
                elif header is not None and line.startswith("#4530"):
                    if station is not None and station in station_rows:
                        station_rows[station][-1].append(line_number)

                    station = line.removeprefix("#4530").removesuffix("\n")
                    if station not in station_rows:
                        station_rows[station] = []

                    station_rows[station].append([int(line_number) + 1])

    file_length = int(line_number)
    station_rows[station][-1].append(file_length)
    for station, rows in station_rows.items():
        inds = []
        print(rows)
        for r in rows:
            inds += list(np.arange(r[0], r[1] + 1))
        df = pd.read_csv(
            path,
            names=header.split(", "),
            skiprows=list(set(np.arange(file_length)) - set(inds)),
        )
        print(df)
        df.to_csv("./data/temperature/" + station + ".csv")
