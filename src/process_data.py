# *** can this be in init?
#import sys
#import os
#sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from obspy import read
import xml.etree.ElementTree as ET
import numpy as np
from bs4 import BeautifulSoup
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
from raydec import raydec
import time
from scipy import fft
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
        datetime.datetime.strptime(val, '%Y-%m-%dT%H:%M:%S')
        return True
    except ValueError:
        return False


#
# PARSING XML
#


def xml_to_dict(contents, include):
    # recursively loop over xml to make dictionary.
    # parse station data
    
    results_dict = {}
    for c in contents:
        if hasattr(c, "name") and c.name is not None and c.name in include and c.name not in results_dict:
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
                result = datetime.datetime.strptime(result, '%Y-%m-%dT%H:%M:%S')
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

    with open(path, 'r') as f:
        file = f.read()

    soup = BeautifulSoup(file, 'xml')

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
            #channels = s.find_all("Channel")
            #channels = [xml_to_dict(c.contents, remaining_vars) for c in channels]

            stations[site] = xml_to_dict(s.contents, remaining_vars)
            #stations[site]["Channels"] = channels
    

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
    directory = r"../../gilbert_lab/Whitehorse_ANT/"

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


def get_time_slice(start_date, time_passed, east, north, vert):
    # shift to be in correct time zone
    # Convert time zone
    start_date = start_date.datetime.astimezone(tz.gettz('Canada/Yukon'))
    dates = [start_date + datetime.timedelta(seconds=s) for s in time_passed]
    hours = np.array([d.hour for d in dates])
    
    inds = (hours >= 2) and (hours <= 4)
    
    print(inds.shape, np.sum(inds))



    
    # look at night-time hours and find quietist(?) (would we always want quietist...?) consecutive 3h?

    # check spacing and nans

    return inds
    

def slice_station_data(
        stations, 
        file_names, 
        input_dir,
        output_dir="./timeseries/"
    ):
    """"""
    # iterate over files in directory
    for ind in range(len(stations)):
        print(stations[ind])
        file_names = file_names[ind]

        for file_name in file_names:
            # read in data
            stream_east = read(input_dir + file_name, format="mseed")
            stream_north = read(input_dir + file_name.replace(".E.", ".N."), format="mseed")
            stream_vert = read(input_dir + file_name.replace(".E.", ".Z."), format="mseed")

            if not np.all(np.array([len(stream_east), len(stream_north), len(stream_vert)]) == 1):
                raise ValueError

            trace_east = stream_east.traces[0]
            trace_north = stream_north.traces[0]
            trace_vert = stream_vert.traces[0]
            
            #dates = trace_east.times(type="matplotlib")
            dates = trace_east.times(type="utcdatetime")
            times = trace_east.times()

            east, north, vert = trace_east.data, trace_north.data, trace_vert.data
            start_date, sampling_rate, sample_spacing = trace_east.stats["starttime"], trace_east.stats["sampling_rate"], trace_east.stats["delta"]

            #time_slice_inds = get_time_slice(start_date, times, east, north, vert)

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
            df["dates"]= df["dates"].dt.tz_localize(datetime.timezone.utc)
            df["dates"] = df["dates"].dt.tz_convert(tz.gettz('Canada/Yukon'))

            hours = np.array([d.hour for d in df["dates"]])

            df = df[np.all(np.array([hours >= 2, hours <= 4]), axis=0)]
            
            df["times"] -= df["times"][0]

            # *** make sure the spacing is correct and gaps have nans
            name = str(start_date).split("T")[0] + ".csv"
            make_output_folder(output_dir)
            make_output_folder(output_dir + "/" + str(stations[ind]) + "/")
            # write station df to csv
            df.to_csv(output_dir + "/" + str(stations[ind]) + "/" + name)

def process_stations(
    directory=r"./../../gilbert_lab/Whitehorse_ANT/"
):
    # save each station to a separate folder...
    # input station list and file list to save

    file_mapping = pd.read_csv("./data/file_information.csv", index_col=0)
    file_names = file_mapping.columns
    stations = file_mapping.loc["station"]

    #unique_stations = np.unique(stations)
    #slice_station_data([station], [file_names[stations == station]], directory)

    
    stations = [24614, 24718, 24952]
    file_names = [file_names[s] for s in stations]
    slice_station_data(stations, file_names, directory)
    
    print("done")


def get_ellipticity(
        station,
        fmin=0.0001,
        fmax=50,
        fsteps=1000,
        cycles=10,
        dfpar=0.1,
    ):
    # loop over saved time series files
    # raydec
    # number of windows based on size of slice
    dir_in = "./timeseries/" + str(station) + "/"
    for file_name in os.listdir(dir_in):
        df_in = pd.read_csv(dir_in + file_name)

        df_in["times"] -= df_in["times"][0]
        n_wind=int(np.round(df_in["times"].iloc[-1])/30) # 30 second windows

        freqs, ellips = raydec(
            vert=df_in["vert"],
            north=df_in["north"],
            east=df_in["east"],
            time=df_in["times"],
            fmin=fmin,
            fmax=fmax,
            fsteps=fsteps,
            cycles=cycles,
            dfpar=dfpar,
            nwind=n_wind
        )

        df_out = pd.DataFrame(ellips.T, columns=freqs[:, 0])

        make_output_folder("./raydec/")
        make_output_folder("./raydec/" + str(station) + "/")
        # write station df to csv
        df_out.to_csv("./raydec/" + str(station) + "/" + file_name)


if __name__ == "__main__":
    """
    run from terminal
    """
    # parse_xml()
    # get_file_information()
    get_ellipticity(24025)
    #process_stations()
