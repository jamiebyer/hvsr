import numpy as np
import pandas as pd
import os
from processing.timeseries_processing import slice_timeseries
from processing.raydec import raydec
from utils.utils import make_output_folder
import time
import dask.dataframe as dd
import dask.array as da
import xarray as xr

from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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
    remove_spikes=True,
):
    # loop over saved time series files
    # raydec
    # number of windows based on size of slice

    dir_in = "./results/timeseries/clipped/" + str(station) + "/" + date
    df_in = pd.read_parquet(dir_in, engine="pyarrow")
    if remove_spikes:
        df_in[df_in["spikes"]] = np.nan
    df_in = df_in.dropna()

    if len(df_in["time"]) < 1:
        return None

    # df_in.compute_chunk_sizes()
    times = df_in["time"].values  # .dropna()  # .values
    # print(times)

    # times -= times[0]

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

    # make_output_folder("./results/raydec/")
    make_output_folder("./results/raydec/0-2-dfpar/" + str(station) + "/")
    # write station df to csv

    # suffix = str(time.time()).split(".")[-1]

    raydec_ds.to_netcdf(
        # "./results/raydec/" + str(station) + "/" + date + ".nc"
        "./results/raydec/0-2-dfpar/"
        + str(station)
        + "/"
        + date
        + ".nc"
    )

    return date


def example_ellipticity(in_path, out_path):

    times, ts, night_inds, delta = slice_timeseries(in_path)
    # need to save timeseries
    times = np.arange(0, len(night_inds) * delta, delta)

    #f_min = 0.8
    f_min = 0.001
    f_max = 40
    f_steps = 500
    cycles = 10
    df_par = 0.1

    len_wind = 3 * 60
    n_wind = int(np.round(delta * len(times[night_inds]) / len_wind))
    freqs, ellips = raydec(
        vert=ts[0][night_inds],
        north=ts[1][night_inds],
        east=ts[2][night_inds],
        time=times[night_inds],
        fmin=f_min,
        fmax=f_max,
        fsteps=f_steps,
        cycles=cycles,
        dfpar=df_par,
        nwind=n_wind,
    )

    ds = xr.DataArray(
        ellips,
        coords={
            "freqs": freqs[:, 0],
            "wind": np.arange(n_wind),
            "fmin": f_min,
            "fmax": f_max,
            "fsteps": f_steps,
            "cycles": cycles,
            "dfpar": df_par,
        },
        dims=["freqs", "wind"],
    )

    ds.to_netcdf(out_path)


def all_station_ellipticities(ind):
    in_path = "./results/timeseries/sorted/"
    out_path = "./results/ellipticity/"
    # out_path = "./results/ellipticity/freq_range/"
    
    # sites
    # sites = ["06", "07A", "17", "23", "24", "25", "32B", "34A", "38B", "41A", "41B", "42B", "47", "50"]
    # sites = ["06"]
    sites = os.listdir(in_path)

    in_paths = []
    out_paths = []
    make_folders = True
    for s in sites:
        for f in os.listdir(in_path + s + "/"):
            if ".E." not in f:
                continue
            if make_folders:
                make_output_folder(out_path + s + "/")
            in_paths.append(in_path + s + "/" + f)
            out_paths.append(out_path + s + "/" + f.replace(".E.miniseed", ".nc"))
    
    example_ellipticity(in_paths[ind], out_paths[ind])



def temporal_ellipticity():
    # loop over day of station
    times, ts, ts_filt = filter_timeseries()
    # for every 1 hour segment, compute ellipticity
    hours = [t.hour for t in times]
    ellips = []
    for h in np.unique(hours):
        ts_slice = ts_filt[hours == h]
        ellips.append(
            raydec(
                ts_slice[0],
                ts_slice[1],
                ts_slice[2],
                times,
                fmin=0.8,
                fmax=40,
                fsteps=100,
                cycles=10,
                dfpar=0.1,
                nwind=60,
            )
        )

    # look at how much it varies over the day
    # (stability of fundamental mode; autocorrelation)
    pass


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


def calc_stacked_std(std_n, da_ellipticity, outlier_inds):
    n_wind = len(da_ellipticity.coords["wind"])
    n_freqs = len(da_ellipticity.coords["freqs"])
    n_valid = np.sum(~outlier_inds)

    if n_valid == 0:
        median = np.full(n_freqs, np.nan)
        mean = np.full(n_freqs, np.nan)
        s = np.full(n_freqs, np.nan)
        return s, mean, median, outlier_inds.data

    # calculate current mean, std

    # get mean with current valid windows
    mean = da_ellipticity[:, ~outlier_inds].mean(dim="wind")
    diff_from_mean = np.abs(da_ellipticity - mean)

    # sample std

    s = np.sqrt(
        (1 / (n_valid - 1)) * np.sum(diff_from_mean[:, ~outlier_inds] * 2, axis=1)
    )

    new_outlier_inds = np.any(diff_from_mean > std_n * s, axis=0)

    # calculate updated mean, std without outliers
    n_valid = np.sum(~new_outlier_inds)
    if n_valid == 0:
        median = np.full(n_freqs, np.nan)
        mean = np.full(n_freqs, np.nan)
        s = np.full(n_freqs, np.nan)
    else:

        median = da_ellipticity[:, ~new_outlier_inds].median(dim="wind").data

        # get mean with current valid windows
        mean = da_ellipticity[:, ~new_outlier_inds].mean(dim="wind")
        diff_from_mean = np.abs(da_ellipticity - mean)
        mean = mean.data

        # sample std
        # s = np.full(n_freqs, np.nan)
        s = np.sqrt(
            (1 / (n_valid - 1)) * np.sum(diff_from_mean[:, ~outlier_inds] * 2, axis=1)
        ).data

    return s, mean, median, new_outlier_inds.data


def label_window_outliers(df_raydec, order, std_n):
    """
    remove outlier windows with values further than 3 std from mean
    """
    N = len(df_raydec.coords["wind"])
    outlier_inds = np.zeros(N).astype(bool)

    for i in range(order):
        std, mean, median, new_outlier_inds = calc_stacked_std(
            std_n, df_raydec, outlier_inds
        )
        outlier_inds = outlier_inds | new_outlier_inds

        df_raydec = df_raydec.assign_coords(
            {
                "QC_" + str(i): ("wind", outlier_inds),
                "std_" + str(i): ("freqs", std),
                "mean_" + str(i): ("freqs", mean),
                "median_" + str(i): ("freqs", median),
            }
        )

    return df_raydec


def find_fundamental_node(in_path="./results/raydec/csv/"):
    for file in os.listdir(in_path):
        df = pd.read_csv(in_path + file)

        peaks, _ = find_peaks(df["median"], height=0.7 * df["median"].max())

        plt.clf()

        plt.plot(df["median"])
        plt.axvline(peaks[0], color="grey")
        plt.savefig("./results/figures/f_0/" + file.replace(".csv", ".png"))
