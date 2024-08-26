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

def xml_to_dict(contents):
    # recursively loop over xml to make dictionary.
    # parse station data
    
    results_dict = {}
    for c in contents:
        if hasattr(c, "name") and c.name is not None and c.name not in results_dict:
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
            result = xml_to_dict(c.contents)
            
        results_dict[c.name].append(result)
    
    return results_dict
        




def read_xml_bs():
    path = "./data/FDSN_Information.xml"

    with open(path, 'r') as f:
        file = f.read() 

    # 'xml' is the parser used. For html files, which BeautifulSoup is typically used for, it would be 'html.parser'.
    soup = BeautifulSoup(file, 'xml')

    # parse the data that is the same for all stations

    # parse station data
    #stations = soup.find_all('Station')[:2]
    stations = soup.find_all('Station')

    #results_dict = xml_to_dict(stations)
    #print(results_dict)

    results_dict = {
        "Station": [],
        "Latitude": [],
        "Longitude": [],
        "Elevation": [],
        "Site": [],
        "CreationDate": [],
        #"TotalNumberChannels": [],
        #"SelectedNumberChannels": [],
        #"Channel": []
    }

    for c in contents:
        if hasattr(c, "name") and c.name is not None and c.name not in results_dict:
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
            result = xml_to_dict(c.contents)
            
        results_dict[c.name].append(result)

    # make into pandas dataframe

    df = pd.DataFrame(results_dict)


def read_xml():
    path = "./data/FDSN_Information.xml"
    fdsn_station = "{http://www.fdsn.org/xml/station/1}"
    tree = ET.parse(path)
    root = tree.getroot()


    """
    all_descendants = list(root.iter())

    results_dict = {}
    for d in all_descendants:
        #name = d.tag.removeprefix(fdsn_station)
        name = d.tag
        if name not in results_dict:
            results_dict[name] = []
        if d.text is not None:
            results_dict[name].append(d.text)
    
    """
    
    prefix = ".//{http://www.fdsn.org/xml/station/1}"
    # overall information
    source = root.find(prefix + "Source").text # DTCC
    sender = root.find(prefix + "Sender").text #DTCC
    module = root.find(prefix + "Module").text #SOLOLITE
    module_uri = root.find(prefix + "ModuleURI").text
    created = root.find(prefix + "Created").text
    total_stations = root.find(prefix + "TotalNumberStations").text #946
    selected_stations = root.find(prefix + "SelectedNumberStations").text
    

    #for r in root:
    #    if hasattr(r, "attrib"):
    #        for a in r.attrib:
    #            print(a)


    stations = root.findall(prefix + "Station")
    #channels = root.findall(prefix + "Channel")

    for s in stations:
        print(s.attrib)
        print(s.find("Latitude"))
        

    #names = root.findall(prefix + "CreationDate")
    #names_text = [n.text for n in names]

    #all_descendants = list(root.iter())
    #for d in all_descendants:
    #    print(d.tag.removeprefix(fdsn_station))
    
    #tails = [d.tail if d.tail is not None else "a" for d in all_descendants]
    #print(np.unique(tails))

    #print(len(np.unique(names_text)))

    #for n in names:
    #    print(n.tail)


    """
    Description 14191
    Latitude 3784
    Longitude 3784
    Elevation 3784
    Name 12298
    CreationDate 946
    TotalNumberChannels 946
    SelectedNumberChannels 946
    Depth 2838
    Azimuth 2838
    Dip 2838
    Type 2838
    SampleRate 2838
    ClockDrift 2838
    Manufacturer 2838
    SerialNumber 2838
    Value 5676
    Frequency 5676
    PzTransferFunctionType 2838
    NormalizationFactor 2838
    NormalizationFrequency 2838
    Real 11352
    Imaginary 11352
    InputSampleRate 2838
    Factor 2838
    Offset 2838
    Delay 2838
    Correction 2838
    """
    
#
# READ YEG PAPERS
#


# parse reading in the data file name

# 453025390.0029.2024.07.04.00.00.00.000.E.miniseed

"""
- split data into windows
- average horizontal components, divide by vertical average
- use fourier transform to move to frequency domain
- can use wavelength to estimate layer thickness (or use mcmc with 1 layer model)
- RAYDEC is used to try to reduce effect of body waves on data
"""



def read_xml():
    tree = ET.parse("data/FDSN_Information.xml")

    # getting the parent tag of the xml document
    root = tree.getroot()

    # Print out the tag of the root and all child tags
    print("root tag", root.tag)
    #for child in root:
        #print(child.tag, child.attrib)


    tags = [elem.tag for elem in root.iter()][:5]
    print("child tags", tags)

    #for movie in root.iter('movie'):
    #    print(movie.attrib)



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
    # https://github.com/ManuelHobiger/RayDec

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

    filtered_data.plot()
    plt.show()

def window_data():
    # window data
    # try diff values for all of these parameters
    window_length = 20 * 60  # 20m in s
    step = window_length
    offset = 0
    include_partial_windows = False
    windows = stream.slide(window_length, step, offset, include_partial_windows)


read_xml_bs()