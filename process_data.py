from obspy import read
import xml.etree.ElementTree as ET
import numpy as np


def read_xml():
    path = "./data/FDSN_Information.xml"
    fdsn_station = "{http://www.fdsn.org/xml/station/1}"
    tree = ET.parse(path)
    root = tree.getroot()
    
    #print(root.attrib)
    all_descendants = list(root.iter())
    #tags = np.unique([d.tag for d in all_descendants])
    #print(tags[-10:-1])

    results_dict = {}
    for d in all_descendants:
        name = d.tag.removeprefix(fdsn_station)
        if name not in results_dict:
            results_dict[name] = []
        if d.text is not None:
            results_dict[name].append(d.text)
    
    for k, v in results_dict.items():
        print(k, len(v))
    

    """
    FDSNStationXML 0
    Source 1
    Sender 1
    Module 1
    ModuleURI 1
    Created 1
    Network 0
    Description 14191
    TotalNumberStations 1
    SelectedNumberStations 1
    Station 0
    Latitude 3784
    Longitude 3784
    Elevation 3784
    Site 0
    Name 12298
    CreationDate 946
    TotalNumberChannels 946
    SelectedNumberChannels 946
    Channel 0
    DataAvailability 0
    Extent 0
    Depth 2838
    Azimuth 2838
    Dip 2838
    Type 2838
    SampleRate 2838
    ClockDrift 2838
    Sensor 0
    Manufacturer 2838
    SerialNumber 2838
    Response 0
    InstrumentSensitivity 0
    Value 5676
    Frequency 5676
    InputUnits 0
    OutputUnits 0
    Stage 0
    PolesZeros 0
    PzTransferFunctionType 2838
    NormalizationFactor 2838
    NormalizationFrequency 2838
    Zero 0
    Real 11352
    Imaginary 11352
    Pole 0
    Decimation 0
    InputSampleRate 2838
    Factor 2838
    Offset 2838
    Delay 2838
    Correction 2838
    StageGain 0
    """

def read_raw_data():
    # loop over stations, all files,
    # process data and save results.

    # read in data
    stream = read("./data/data.miniseed", format="mseed")
    trace = stream.traces[0]

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
    _format: MSEED
    mseed: AttribDict({'dataquality': 'D', 'number_of_records': 2110, 'encoding': 'FLOAT32', 'byteorder': '>', 'record_length': 4096, 'filesize': 8642560})
    """

    # trace.plot()

def window_data():
    # prolly calculations before windowing

    # raydec
    # https://github.com/ManuelHobiger/RayDec

    # window data
    # try diff values for all of these parameters
    window_length = 20 * 60  # 20m in s
    step = window_length
    offset = 0
    include_partial_windows = False
    windows = stream.slide(window_length, step, offset, include_partial_windows)

    for win in windows:
        # hvsr
        #
        pass

def raydec_data():
    pass

#
# READ YEG PAPERS
#

# read in xml






"""
- split data into windows
- average horizontal components, divide by vertical average
- use fourier transform to move to frequency domain
- can use wavelength to estimate layer thickness (or use mcmc with 1 layer model)
- RAYDEC is used to try to reduce effect of body waves on data
"""



read_xml()