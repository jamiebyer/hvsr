
# HVSR processing .miniseed data
---
## Data
- geophone sensor specs: https://smartsolo.com/cp-3.html
- seismic ambient noise collected in Whitehorse, Yukon between June and August 2024
- sampling frequency: 10 Hz, 0.1 s


---

## Timeseries
### Timeseries processing
1. Parse xml. Get stations and corresponding coordinate locations.

- ends have noise from being taken out, put back in ground

1. Get timeseries miniseed files from GLADOS. Convert each station and each day to parquet. (compare sizes)
    - functions
    - use rsync to save locally
2. Get relevant stats () for each station and day and save as csv. For each station and day, select the night hours () and remove points with amplitude >= 3 standard deviation from mean for the current day. Save this "processed" timeseries as parquet.
    - This is removing earthquakes and outliers, and selecting times with less anthropogenic noise so we have ambient noise.
    - label stats first and get "cleaned" stats for the timeseries too.

- running slurm script

### Timeseries plotting
1. Map of stations and locations

## Ellipticity
### Ellipticity processing
1. run raydec on processed timeseries
    - emphasizes Rayleigh waves and filters body waves(?), then uses HVSR to calculate ellipticity of the Rayleigh waves.
    - converted to Python from: https://github.com/ManuelHobiger/RayDec
2. remove windows >= 3 standard deviation from mean
    - standard deviation for stack

- slurm script


### Ellipticity plotting
1. ellipticity compared with timeseries and station location


---
## Running the code
### environment
- setting up environment
### timeseries
### ellipticity

### app
- launch with `python ./src/app.py` then visit http://0.0.0.0:8050/
- paths for where to put timeseries data








