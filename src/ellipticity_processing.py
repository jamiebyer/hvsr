import numpy as np
import pandas as pd
import os
from raydec import raydec
from utils import make_output_folder
from dateutil import tz
import sys
import json
import time
import dask.dataframe as dd
import xarray as xr


###### RAYDEC PROCESSING ######


def get_ellipticity(
    station,
    date,
    fmin=0.8,
    fmax=40,
    fsteps=100,
    cycles=10,
    dfpar=0.1,
    len_wind=3 * 60,
    remove_spikes=False,
):
    # loop over saved time series files
    # raydec
    # number of windows based on size of slice
    dir_in = "./results/timeseries/" + str(station) + "/" + date
    df_in = pd.read_parquet(dir_in, engine="pyarrow")
    if remove_spikes:
        df_in[df_in["spikes"]] = np.nan
    df_in = df_in.dropna()

    if len(df_in["times"]) < 1:
        return None

    # df_in.compute_chunk_sizes()
    times = df_in["times"].dropna()  # .values

    times -= times[0]

    # n_wind = int((times[-1] / len_wind).round().compute())
    n_wind = int(np.round(times[len(times) - 1] / len_wind))

    freqs, ellips = raydec(
        vert=df_in["vert"],
        north=df_in["north"],
        east=df_in["east"],
        time=times,
        fmin=fmin,
        fmax=fmax,
        fsteps=fsteps,
        cycles=cycles,
        dfpar=dfpar,
        nwind=n_wind,
    )

    # df_timeseries = xr.Dataset(ellips.T, columns=freqs[:, 0])

    ds = xr.DataArray(
        ellips,
        coords={
            "freqs": freqs[:, 0],
            "wind": np.arange(n_wind),
            "fmin": fmin,
            "fmax": fmax,
            "fsteps": fsteps,
            "cycles": cycles,
            "dfpar": dfpar,
        },
        dims=["freqs", "wind"],
    )

    # df_timeseries["outliers"] = (np.abs(df_timeseries["vert"]) >= max_amplitude).astype(
    #    int
    # )
    return ds


def write_raydec_df(
    station,
    date,
    f_min,
    f_max,
    f_steps,
    cycles,
    df_par,
    len_wind,
):
    raydec_ds = get_ellipticity(
        station,
        date,
        f_min,
        f_max,
        f_steps,
        cycles,
        df_par,
        len_wind,
        remove_spikes=True,
    )
    if raydec_ds is None:
        return None

    make_output_folder("./results/raydec/")
    make_output_folder("./results/raydec/" + str(station) + "/")
    # write station df to csv

    suffix = str(time.time()).split(".")[-1]

    raydec_ds.to_netcdf(
        "./results/raydec/" + str(station) + "/" + date + "-" + suffix + ".nc"
    )

    return date


def remove_window_outliers(df_raydec, scale_factor):
    """
    remove outliers from phase dispersion. windows with values further than 3 std from mean
    """
    mean = np.mean(df_raydec, axis=1)
    std = np.std(df_raydec, axis=1)

    # diff_from_mean = np.abs(mean - df_raydec)
    diff_from_mean = df_raydec.sub(df_raydec.mean(axis=1), axis=0)

    outlier_inds = np.any(diff_from_mean.T > scale_factor * std, axis=1)
    outlier_inds = outlier_inds.astype(int).rename("outliers").to_frame().T
    df_raydec = dd.concat([df_raydec, outlier_inds])

    mean_inds = df_raydec.loc["outliers"] == 0
    mean = np.nanmean(df_raydec.loc[:, mean_inds], axis=1)
    df_raydec["mean"] = mean

    return df_raydec


def process_station_ellipticity(
    ind,
    directory=r"./results/timeseries/",
):
    # save each station to a separate folder...
    # input station list and file list to save

    f_min = 0.1
    f_max = 20
    f_steps = 100
    cycles = 10
    df_par = 0.1
    len_wind = 3 * 60

    station_list = []
    for station in os.listdir(directory):
        if station != "stats.csv":
            for date in os.listdir(directory + station):
                station_list.append([station, date])

    write_raydec_df(
        station_list[ind][0],
        station_list[ind][1],
        f_min,
        f_max,
        f_steps,
        cycles,
        df_par,
        len_wind,
    )


def sensitivity_test(ind):
    # try a range of frequencies, of df_par
    station = 24614
    date = "2024-06-15"

    params = []

    f_min, f_max = [0.8, 20]
    # for f_min, f_max in [[0.8, 20]]

    f_steps = 100

    cycles = 10
    # for cycles in np.arange(8, 13):

    df_par = 0.1
    # for df_par in np.linspace(0.05, 0.15, 10):

    len_wind = 3 * 60
    # for len_wind in np.linspace(60, 10*60, 10):

    for len_wind in np.linspace(60, 10 * 60, 10):
        params.append(
            [
                f_min,
                f_max,
                f_steps,
                cycles,
                df_par,
                len_wind,
            ]
        )

    write_raydec_df(station, date, *params[ind])


def stack_station_windows():
    in_dir = "./results/raydec/"
    for station in ["24625"]:  # os.listdir(in_dir):
        if station not in ["stacked", "raydec"]:
            stack_columns = []
            stack_data = []
            for file in os.listdir(in_dir + station):
                # add average to dataframe
                da_raydec = xr.open_dataarray(in_dir + station + "/" + file)
                stack_columns.append(file.rsplit("-", 1)[0])
                mean = da_raydec.mean(dim="wind")
                stack_data.append(mean)

                # using the last freqs, assuming theyre all the same
                freqs = da_raydec.coords["freqs"]

            stack_columns.append("mean")
            mean = np.array(stack_data).mean(axis=0)
            stack_data.append(mean)

            da_stack = xr.DataArray(
                np.array(stack_data).T, {"freqs": freqs, "date": stack_columns}
            )

            make_output_folder("./results/raydec/stacked/")
            # write station df to csv

            da_stack.to_dataframe(name="ellipticity").to_csv(
                "./results/raydec/stacked/" + str(station) + "_full.csv"
            )
            da_stack.sel(date="mean").to_dataframe(name="ellipticity").to_csv(
                "./results/raydec/stacked/" + str(station) + "_mean.csv"
            )

            # da_stack.to_netcdf("./results/raydec/stacked/" + str(station) + ".nc")


if __name__ == "__main__":
    """
    run from terminal
    """
    # ind = int(sys.argv[1])

    process_station_ellipticity(ind=1)
