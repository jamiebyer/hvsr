
# HVSR processing .miniseed data
---
## Data
- geophone sensor specs: https://smartsolo.com/cp-3.html
- seismic ambient noise collected in Whitehorse, Yukon between June and August 2024
- sampling frequency: 10 Hz, 0.1 s


## xml processing
1. Parse xml. Get stations and corresponding coordinate locations.

(info from xml- sampling rate, etc.)


**Figures**:
- Map of stations and locations
---

## Timeseries  processing
1. Get timeseries miniseed files from GLADOS. Convert each station and each day to parquet. (compare sizes)
    - `convert_miniseed_to_parquet(in_path, out_path)`
    - slurm to parallelize

2. Get initial stats for full timeseries and save to csv.
    - `get_timeseries_stats(include_outliers, in_path, out_path)`
    - save locally
    - `rsync -a remote_user@remote_host_or_ip:/opt/media/ /opt/media/`


3. Remove points with amplitude >= 3 standard deviation from mean for the current day. Get stats for the new timeseries. Select the night hours () from the timeseries, get new stats. Save this "processed" timeseries as parquet, and save stats to df.
    - `get_clean_timeseries_slice(in_path, out_path)`
    - This is removing earthquakes and outliers, and selecting times with less anthropogenic noise so we have ambient noise.
    - slurm
    - use `rsync` to save locally 

4. Get stats for full timeseries with outliers removed and sliced timeseries
    - label stats first and get "cleaned" stats for the timeseries too.
    - save locally


**Figures**:
- 
**Notes **:
- ends have noise from being taken out, put back in ground

---
## Ellipticity processing
1. run raydec on processed timeseries
    - emphasizes Rayleigh waves and filters body waves(?), then uses HVSR to calculate ellipticity of the Rayleigh waves.
    - converted to Python from: https://github.com/ManuelHobiger/RayDec
2. remove windows >= 3 standard deviation from mean
    - standard deviation for stack

- slurm script

**Figures**:
- ellipticity compared with timeseries and station location


---
## Running the code
### environment
- setting up environment
### timeseries
### ellipticity

### app
- launch with `python ./src/app.py` then visit http://0.0.0.0:8050/
- paths for where to put timeseries data








