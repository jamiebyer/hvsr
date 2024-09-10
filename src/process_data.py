# *** can this be in init?
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from obspy import read
import xml.etree.ElementTree as ET
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
from raydec import raydec
import time
from scipy import fft


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
        datetime.strptime(val, '%Y-%m-%dT%H:%M:%S')
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
                result = datetime.strptime(result, '%Y-%m-%dT%H:%M:%S')
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


def process_stations():
    # save each station to a separate folder...
    # input station list and file list to save

    file_mapping = pd.read_csv("./data/file_information.csv", index_col=0)
    file_names = file_mapping.columns
    stations = file_mapping.loc["station"]
    unique_stations = np.unique(stations)

    directory = r"./../../gilbert_lab/Whitehorse_ANT/"

    station = unique_stations[0]
    slice_station_data([station], [file_names[stations == station]], directory)
    
    print("done")

def slice_station_data(
        stations, 
        file_names, 
        input_dir,
        output_dir="./figures/"
    ):
    """"""
    # iterate over files in directory
    for ind in range(len(stations)):
        print(stations[ind])
        file_names = file_names[ind]

        station_data = {
            "time": [],
            "vert": [],
            "north": [],
            "east": [],
        }

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
            
            dates = trace_east.times(type="matplotlib")
            times = trace_east.times()

            east, north, vert = trace_east.data, trace_north.data, trace_vert.data
            start_date, sampling_rate, sample_spacing = trace_east.stats["starttime"], trace_east.stats["sampling_rate"], trace_east.stats["delta"]

            time_slice_inds = get_time_slice(dates, times, east, north, vert)

            station_data["time"].append(times[time_slice_inds])
            station_data["vert"].append(vert[time_slice_inds])
            station_data["north"].append(north[time_slice_inds])
            station_data["east"].append(east[time_slice_inds])
        
        df = pd.DataFrame(station_data)
        df.sort_values(by="time")
        # sort by dates
        # write station df to csv
        df.to_csv(directory + "timeseries/" + str(stations[ind]))

                
    
def merge_time_data_files():
    pass


def get_ellipticity(
        station,
        fmin=1,
        fmax=20,
        fsteps=100,
        cycles=10,
        dfpar=0.1,
    ):
    # loop over saved time series files
    # raydec
    # number of windows based on size of slice
    n_wind=24*60

    raydec(
        vert=vert[time_slice_inds],
        north=north[time_slice_inds],
        east=east[time_slice_inds],
        time=times[time_slice_inds],
        fmin=fmin,
        fmax=fmax,
        fsteps=fsteps,
        cycles=cycles,
        dfpar=dfpar,
        nwind=n_wind
    )
    # save raydec
    path = ""

    np.save(path + "_freqs", V)
    np.save(path + "_ellips", W)

def moving_average():
    # moving averaged
    window_size = int(10*60 / sample_spacing) # 30 m
    convolution_kernel = np.ones(window_size)/window_size
    times_avg = np.convolve(times, convolution_kernel, mode='valid') 
    east_avg = np.convolve(east, convolution_kernel, mode='valid')
    north_avg = np.convolve(north, convolution_kernel, mode='valid')
    vert_avg = np.convolve(vert, convolution_kernel, mode='valid')

def calc_hvsr(times, east, north, vert, sample_spacing):
    n_samples = len(times)
    freqs = fft.fftfreq(n_samples, sample_spacing)
    east_fft = fft.fft(east)
    north_fft = fft.fft(north)
    vert_fft = fft.fft(vert)

    hvsr = np.sqrt(east_fft**2 + north_fft**2)/(np.sqrt(2)*np.abs(vert_fft))
    return freqs, hvsr


def get_time_slice():
    pass


def window_data():
    # window data
    # try diff values for all of these parameters
    window_length = 20 * 60  # 20m in s
    step = window_length
    offset = 0
    include_partial_windows = False
    windows = stream.slide(window_length, step, offset, include_partial_windows)


if __name__ == "__main__":
    """
    run from terminal
    """
    # parse_xml()
    # get_file_information()
    process_stations()
