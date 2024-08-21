from obspy import read

#
# READ YEG PAPERS
#

# read in xml

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

"""
- split data into windows
- average horizontal components, divide by vertical average
- use fourier transform to move to frequency domain
- can use wavelength to estimate layer thickness (or use mcmc with 1 layer model)
- RAYDEC is used to try to reduce effect of body waves on data
"""
