
# Timeseries -> Ellipticity processing

---
## TODO
- Determine deployment schedule
- Save current clipped timeseries to plots.
- Save current ellipticity to plots
- Fix up temperature plot
- Upload data so app can plot

- validate curve stability
- spike removal: STA LTA
- temporal ellipticity
- hvsr for each component

---
## OTHER DATA
- Yukon databases
    - Well holes

---
## Data
- geophone sensor specs: https://smartsolo.com/cp-3.html
- seismic ambient noise collected in Whitehorse, Yukon between June and August 2024
- sampling frequency: 10 Hz, 0.1 s

---
## xml processing
1. Parse FDSN_Information.xml to get geophone serial numbers and coordinates.
2. Determine full deployment schedule information.
3. Get other information from xml: sampling rate
4. Map stations and locations

---

## Timeseries processing
1. Get timeseries miniseed files from GLADOS.
    - Indicate spikes and distribution (save to csv)
    - Plot downsampled full timeseries and spikes
2. Clip night section from timeseries.
    - Remove points with amplitude >= 3 standard deviation from mean for the current day.
    - Find the quietest sections of night
        - This is removing earthquakes and outliers, and selecting times with less anthropogenic noise so we have ambient noise.
3. Clip times where geophone is moved. (starts and ends are easy, but geophones that are moved mid-deployment?)
4. Plot clipped miniseed for each timeseries and save.


- I would rather save to xarray... Try sticking with miniseed at least for initial processing.


---
## Ellipticity processing
- sensitivity test

1. Run raydec on processed timeseries
    - Emphasizes Rayleigh waves, then uses HVSR to calculate ellipticity of the Rayleigh waves
    - Note on difference between HVSR and ellipticity
2. Remove outliers
    - remove windows >= 3 standard deviation from mean
    - standard deviation for stack


- Plot ellipticity compared with timeseries and station location

### RayDec
- Converted to Python from: https://github.com/ManuelHobiger/RayDec
- Parallelize current code

---
## Transfering files between remote and local
- `rsync -a remote_user@remote_host_or_ip:/opt/media/ /opt/media/`


---
## Other plotting
- Temperature
- Permafrost
- CMIP models?

---
## Running the code
### environment
- setting up environment
### timeseries
### ellipticity

### app
- launch with `python ./src/app.py` then visit http://0.0.0.0:8050/
- paths for where to put timeseries data







