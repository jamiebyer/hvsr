
# Timeseries -> Ellipticity processing



---
## Data
- geophone sensor specs: https://smartsolo.com/cp-3.html
- Seismic ambient noise collected in Whitehorse, Yukon between June and August 2024
- 3-component geophone
- sampling frequency: 5 Hz***

- Got deployment schedule from the screenshots of: serial number, start date, end date, coords for each recording. From the SmartSolo data. Parsed coordinates.
- Station mapping in: "df_mapping.csv"
- Deployment cycles, and permanent stations.

---

## Timeseries processing
- Example sites: 


- Use Fourier transform to remove frequencies > 2.5 Hz. 
- Set frequencies > 2.5 Hz from all components to zero.




- Clean up RayDec



- Nice sensitivity test: dfpar, cycles, length of time windows.
- Plot the night hours ellipticity in 1h segments
- Remove bad windows.
- Set up hvsrpy, compare HVSR and RayDec.



- Check stability of curves over a day and over the recording time. (select most stable time period?)
- Compare to other data (wells, temperature)


1. Get timeseries miniseed files from GLADOS.
    - Indicate spikes and distribution (save to csv)
    - Plot downsampled full timeseries and spikes
2. Clip night section from timeseries.
    - Remove points with amplitude >= 3 standard deviation from mean for the current day.
    - Find the quietest sections of night
        - This is removing earthquakes and outliers, and selecting times with less anthropogenic noise so we have ambient noise.
4. Plot clipped miniseed for each timeseries and save.



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







