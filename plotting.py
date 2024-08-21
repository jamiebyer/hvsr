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

trace.plot()

# tutorial: https://krischer.github.io/seismo_live_build/html/ObsPy/03_waveform_data_solution_wrapper.html


# will loop over all files...
# filename = "453024025.0001.2024.06.06.18.04.52.000.E.miniseed"
# path = Path("/gilbert_lab/Whitehorse_ANT/")
# path = Path("/gilbert_lab/Whitehorse_ANT/FDSN_Information.xml")


# plot station locations

#
