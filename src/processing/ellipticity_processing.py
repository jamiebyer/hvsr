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


def compute_ellipticity(
    ts_path, out_path, f_min, f_max, f_steps, cycles, df_par, len_wind
):

    times, ts, night_inds, delta = slice_timeseries(ts_path)
    # need to save timeseries
    times = np.arange(0, len(night_inds) * delta, delta)

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


def determine_ellipticity_outliers(in_path_timeseries, in_path_raydec, out_path):
    # determine ellipticity outliers
    # save the time windows associated with the outliers

    # find raydec windows with any points > 3sd from median, and remove them
    # find positions of big spikes in timeseries (using fourier transform)
    # determine which windows from raydec correspond to big spikes, and if
    # they were removed with the quality control

    # read in timeseries
    # plot overtop the time sections removed by raydec quality control

    # for a single day of a station

    times, ts, ts_filt = filter_timeseries(in_path_timeseries)

    # read in raydec file
    da = xr.open_dataarray(in_path_raydec)

    # compute median
    median = da.median(dim="wind")
    # create da variable for outliers, with dimension window
    print(da)

    ellip = da.values
    n_wind = len(da["wind"])
    # determine std
    sigma = np.sqrt((1 / n_wind) * np.sum((ellip - median) ** 2))

    # add a column indicating 3 std from median
    outliers = ellip >= 3 * sigma

    # label windows as outliers
    # convert windows back to timeseries...


def stack_station_ellipticity():
    # remove outlier ellipticities
    # stack all ellipticities for a station deployment
    # determine std
    # find the fundamental mode
    pass


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
    # look for a peak in frequency range
    for file in os.listdir(in_path):
        df = pd.read_csv(in_path + file)

        peaks, _ = find_peaks(df["median"], height=0.7 * df["median"].max())

        plt.clf()

        plt.plot(df["median"])
        plt.axvline(peaks[0], color="grey")
        plt.savefig("./results/figures/f_0/" + file.replace(".csv", ".png"))
