from obspy import read
import numpy as np
from bs4 import BeautifulSoup
import datetime
import pandas as pd
import os
import sys
from utils import is_int, is_date, is_float


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


if __name__ == "__main__":
    """
    run from terminal
    """
    # ind = int(sys.argv[1])
    pass
