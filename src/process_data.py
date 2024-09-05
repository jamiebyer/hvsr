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
from plotting import plot_station_timeseries, plot_station_hvsr


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

def parse_data():
    # sensor specs: https://smartsolo.com/cp-3.html
    # parse reading in the data file name
    # 453025390.0029.2024.07.04.00.00.00.000.E.miniseed
    # save each station to a separate file

    file_mapping = pd.read_csv("./data/file_information.csv")

    stations = np.unique(file_mapping["stations"])
 
    directory = r"./../../gilbert_lab/Whitehorse_ANT/"
    # iterate over files in directory
    for station in [stations[0]]:
        print(station)
        file_names = file_mapping.loc[file_mapping["stations"] == station]["file_names"]
        
        # save each file to csv
        for file_name in file_names[:2]:
            print("\n", file_name)
            # read in data        
            stream_east = read(directory + file_name, format="mseed")
            stream_north = read(directory + file_name.replace(".E.", ".N."), format="mseed")
            stream_vert = read(directory + file_name.replace(".E.", ".Z."), format="mseed")

            if not np.all(np.array([len(stream_east), len(stream_north), len(stream_vert)]) == 1):
                raise ValueError

            trace_east = stream_east.traces[0]
            trace_north = stream_north.traces[0]
            trace_vert = stream_vert.traces[0]
            
            # make sure all directions line up for times
            dates = trace_east.times(type="matplotlib")
            times = trace_east.times()

            # starttime: 2024-06-06T18:04:52.000000Z
            # endtime: 2024-06-07T00:00:00.000000Z
            # sampling_rate: 100.0
            # delta: 0.01

            east, north, vert = trace_east.data, trace_north.data, trace_vert.data
            # *** i think sampling rate and delta are swapped in the xml..... ***
            start_date, sampling_rate, sample_spacing = trace_east.stats["starttime"], trace_east.stats["sampling_rate"], trace_east.stats["delta"]

            # moving averaged
            window_size = int(10*60 / sample_spacing) # 30 m
            convolution_kernel = np.ones(window_size)/window_size
            times_avg = np.convolve(times, convolution_kernel, mode='valid') 
            east_avg = np.convolve(east, convolution_kernel, mode='valid')
            north_avg = np.convolve(north, convolution_kernel, mode='valid')
            vert_avg = np.convolve(vert, convolution_kernel, mode='valid')

            """
            plot_station_timeseries(
                start_date, 
                station, 
                times,
                east, 
                north, 
                vert,
                times_avg, 
                east_avg,
                north_avg,
                vert_avg,
            )
            """
            freqs, hvsr = calc_hvsr(times_avg, east_avg, north_avg, vert_avg, sample_spacing)
            plot_station_hvsr(start_date, station, freqs, east_avg, north_avg, vert_avg, hvsr)

    print("done")
    
def calc_hvsr(times, east, north, vert, sample_spacing):
    n_samples = len(times)
    freqs = fft.fftfreq(n_samples, sample_spacing)
    east_fft = fft.fft(east)
    north_fft = fft.fft(north)
    vert_fft = fft.fft(vert)

    hvsr = np.sqrt(east_fft**2 + north_fft**2)/(np.sqrt(2)*np.abs(vert_fft))
    return freqs, hvsr

def process_data(station_dict):
    """
    - split data into windows (done by raydec?)
    - RAYDEC is used to try to reduce effect of body waves on data
    - average horizontal components, divide by vertical average
    - use fourier transform to move to frequency domain
    - can use wavelength to estimate layer thickness (or use mcmc with 1 layer model)
    """
    """
    # get suggested inputs for raydec
    CYCLES = 10
    DFPAR = 0.1
    NWIND such that the single time windows are about 10 minutes long
    """

    times = station_dict["time"]
    print("times: ", np.min(times), np.max(times))
    n_wind = np.round(times[-1] / (10*60)).astype(int)

    #f = numpy.linspace(0.1, 10.0, 100)
    #t = 1.0 / f[::-1]

    freq_sampling = 1/100
    freq_nyq = freq_sampling / 2


    # raydec
    filtered_data = raydec(
        vert=station_dict["vert"],
        north=station_dict["north"],
        east=station_dict["east"],
        time=station_dict["time"],
        fmin=0.002,
        fmax=0.0333,
        fsteps=100,
        cycles=10,
        dfpar=0.1,
        nwind=n_wind
    )

    hvsr = calc_hvsr(east, north, vert)

    # Fourier transform

    # use wavelength to estimate layer thickness (or use mcmc with 1 layer model)

    #filtered_data.plot()
    #plt.show()


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
    parse_data()
