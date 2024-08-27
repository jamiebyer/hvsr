from obspy import read
import xml.etree.ElementTree as ET
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


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

def is_channel(val):
    return hasattr(val, "children")

def parse_channel(channel):
    print(channel)

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


def get_tags_to_save():

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

    
    #pd.DataFrame(stations_dict).to_csv("./data/parsed_xml.csv")
    #print(stations["24025"])

    

# parse reading in the data file name

# 453025390.0029.2024.07.04.00.00.00.000.E.miniseed

"""
- split data into windows
- average horizontal components, divide by vertical average
- use fourier transform to move to frequency domain
- can use wavelength to estimate layer thickness (or use mcmc with 1 layer model)
- RAYDEC is used to try to reduce effect of body waves on data
"""



def read_data():

    # read in data
    stream_east = read("data/453025390.0029.2024.07.04.00.00.00.000.E.miniseed", format="mseed")
    stream_north = read("data/453025390.0029.2024.07.04.00.00.00.000.N.miniseed", format="mseed")
    stream_vert = read("data/453025390.0029.2024.07.04.00.00.00.000.Z.miniseed", format="mseed")
    
    trace_east = stream_east.traces[0]
    trace_north = stream_north.traces[0]
    trace_vert = stream_vert.traces[0]
    #stream = Stream(traces=[trace_east, trace_north, trace_vert])
    
    # make sure all directions line up for times
    times = trace_east.times()

    """
    trace.stats:
        network: SS
        station: 24025
        location: SW
        channel: EPE
        starttime: 2024-06-06T18:04:52.000000Z
        endtime: 2024-06-07T00:00:00.000000Z
        sampling_rate: 100.0
        delta: 0.01
        npts: 2130801
        calib: 1.0
    """
    
    """
    suggested values: CYCLES = 10
    DFPAR = 0.1
    NWIND such that the single time windows are about 10 minutes long
    """

    print("times: ", np.min(times), np.max(times))
    n_wind = np.round(times[-1] / (10*60*60)).astype(int)

    # cycles: number of periods
    
    #f = numpy.linspace(0.1, 10.0, 100)
    #t = 1.0 / f[::-1]

    # raydec
    filtered_data = raydec(
        vert=trace_vert.data,
        north=trace_north.data,
        east=trace_east.data,
        time=times,
        fmin=0.5,
        fmax=5,
        fsteps=100,
        cycles=10,
        dfpar=0.1,
        nwind=n_wind
    )

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


get_tags_to_save()