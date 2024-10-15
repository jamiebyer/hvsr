
# HVSR processing

## Data
- geophone sensor specs: https://smartsolo.com/cp-3.html
- seismic ambient noise collected in Whiethorse, Yukon between ---- dates
- sampling frequency:
- 


- temperature data from: https://whitehorse.weatherstats.ca/download.html



## Timeseries processing
- quality control, ends of timeseries, earthquakes, standard deviation, quiet times
- spike removal and 
- slurm script


## Raydec
- converted to Python from: https://github.com/ManuelHobiger/RayDec

- emphasizing Rayleigh waves, getting ellipticity using HVSR

### params

calculates the ellipticity of Rayleigh waves for the input data VERT, NORTH, EAST and TIME for a single station for FSTEPS frequencies (on a logarithmic scale) between FMIN and FMAX, using CYCLES periods for the stacked signal and DFPAR as the relative bandwidth for the filtering. The signal is cut into NWIND different time windows and RayDec is applied to each of them.

- VERT, NORTH, EAST and TIME have to be arrays of equal sizes
- suggested values:
    - CYCLES = 10
    - DFPAR = 0.1
    - NWIND such that the single time windows are about 10 minutes long


- **vert**: vertical component of seismic data
- **north**: northern component of seismic data
- **east**: eastern component of seismic data
- **time**: time array for seismic data
- **fmin**: minimum frequency
- **fmax**: maximum frequency
- **fsteps**: number of frequencies to calculate
- **cycles**:
- **dfpar**: relative bandwidth for the filtering
- **nwind**: number of windows


## Stacking 
- fundamental node


## App
- setting up environment
- launch with `python ./src/app.py` then visit http://0.0.0.0:8050/
- paths for where to put timeseries data


## Code structure

- **app**:
- **data_parsing**:
- **ellipticity_processing**:
- **layout**:
- **plotting**:
- **raydec**:
- **timeseries_processing**:
- **utils**:








