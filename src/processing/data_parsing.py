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
    #path1 = "./data/FDSN Information/FDSN_Information_453024025_1.xml"
    #path2 = "./data/FDSN Information/FDSN_Information_453025229_1.xml"
    path2 = "./data/FDSN_Information.xml"

    with open(path2, "r") as f:
        file = f.read()

    soup = BeautifulSoup(file, "xml")


    #["SerialNumber", "Latitude", "Longitude", "Site"]


    results = {}
    for station in soup.find_all("Station"):
        site = station.find("Site").find("Name").text
        if site not in results:
            results[site] = []
        location = {
                "lon": float(station.find("Longitude").text),
                "lat": float(station.find("Latitude").text),
                "elev": float(station.find("Elevation").text)
            }
        if location not in results[site]:
            results[site].append(location)
    
    return results




    


def parse_xml_og(save=True):
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



def get_station_positions(ind, in_path=r"./data/Whitehorse_ANT/", out_path=r"./results/timeseries/raw/"):
    """
    Loop over raw timeseries miniseed files from smart solo geophone.
    Consolodate 3 components (vert, north, east) into one file.
    Determine station and date from miniseed and save to corresponding path.

    :param in_path: path of directory containing miniseed files
    """
    # Loop over raw miniseed files
    #make_output_folder(out_path)
    
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
                    
                    station_rows[station].append([int(line_number)+1])
        
    
    file_length = int(line_number)
    station_rows[station][-1].append(file_length)
    for station, rows in station_rows.items():
        inds = []
        print(rows)
        for r in rows:
            inds += list(np.arange(r[0], r[1]+1))
        df = pd.read_csv(path, names=header.split(", "), skiprows=list(set(np.arange(file_length)) - set(inds)))
        print(df)
        df.to_csv("./data/temperature/" + station + ".csv")

