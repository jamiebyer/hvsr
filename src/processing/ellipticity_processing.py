import numpy as np
import pandas as pd
import os
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
    times = df_in["time"].values#.dropna()  # .values
    #print(times)

    #times -= times[0]

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

    #make_output_folder("./results/raydec/")
    make_output_folder("./results/raydec/0-2-dfpar/" + str(station) + "/")
    # write station df to csv

    #suffix = str(time.time()).split(".")[-1]

    raydec_ds.to_netcdf(
        #"./results/raydec/" + str(station) + "/" + date + ".nc"
        "./results/raydec/0-2-dfpar/" + str(station) + "/" + date + ".nc"
    )

    return date


def label_window_outliers_og(df_raydec):
    """
    remove outlier windows with values further than 3 std from mean
    """
    std = np.std(da_raydec, axis=1)

    # diff_from_mean = np.abs(mean - df_raydec)
    # diff_from_mean = da.abs(da.subtract(da_raydec, da_raydec.mean(axis=1), axis=0))
    diff_from_mean = da.abs(da.subtract(da_raydec, da_raydec.mean(axis=1)))

    outlier_inds = np.any(diff_from_mean.T >= 3 * std, axis=1)
    da_raydec["outliers"] = outlier_inds.astype(int)

    # outlier_inds = outlier_inds.astype(int).rename("outliers").compute().T
    # da_raydec = dd.concat([da_raydec, outlier_inds])

    return da_raydec


def remove_all_window_outliers(ind):
    in_dir = "./results/raydec/0-2-dfpar/"
    # clean up
    station_list = []
    for station in os.listdir(in_dir):
        for date in os.listdir(in_dir + station):
            station_list.append([station, date])

    # later add this to first ellipticity run

    station, date = station_list[ind][0], station_list[ind][1]
    da_raydec = xr.open_dataarray(in_dir + "/" + station + "/" + date)

    # label outliers
    da_raydec = remove_window_outliers(da_raydec)

    # save labeled outliers back to nc
    da_raydec.to_netcdf(in_dir + str(station) + "/" + date)

    # and save to csv without outliers
    df_raydec = da_raydec.to_dataframe(name="ellipticity")
    make_output_folder("./results/raydec/0-2-dfpar/csv/")
    make_output_folder("./results/raydec/0-2-dfpar/csv/" + station)
    df_raydec[df_raydec["outliers"] == 0].to_csv(
        in_dir + "csv/" + str(station) + "/" + date.replace(".nc", ".csv")
    )


def process_station_ellipticity(
    ind,
    directory=r"./results/timeseries/clipped/",
):
    # save each station to a separate folder...
    # input station list and file list to save

    f_min = 0.1
    f_max = 20
    f_steps = 100
    cycles = 10
    df_par = 0.2
    len_wind = 3 * 60

    station_list = []
    for station in os.listdir(directory):
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
        std, mean, median, new_outlier_inds = calc_stacked_std(std_n, df_raydec, outlier_inds)
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


def label_all_window_outliers(ind, order, std_n, save_csv=False):
    in_dir = "./results/raydec/"
    # clean up
    station_list = []
    for station in os.listdir(in_dir):
        if station not in ["csv", "QC_std"]:
            for date in os.listdir(in_dir + station):
                station_list.append([station, date])

    # later add this to first ellipticity run

    station, date = station_list[ind][0], station_list[ind][1]
    da_raydec = xr.open_dataarray(in_dir + "/" + station + "/" + date)
    # label outliers
    # da_raydec = da_raydec.drop("std")
    da_raydec = label_window_outliers(da_raydec, order, std_n)

    # save labeled outliers back to nc
    path = (
        in_dir
        + "QC_std"
        + str(std_n)
        + "/"
        + str(station)
        + "/"
        + date.replace(".nc", "-QC_" + str(std_n) + ".nc")
    )
    print(path)
    # in_dir + "QC_std"+str(std_n)+"/" str(station) + "/" + date
    make_output_folder(in_dir + "QC_std" + str(std_n) + "/" + str(station) + "/")
    da_raydec.to_dataset(name="ellipticity").to_netcdf(path)

    # and save to csv without outliers
    if save_csv:
        # filter out outliers
        # da_raydec = da_raydec.loc["QC"] == 0

        da_raydec.to_dataframe(name="ellipticity").to_csv(
            in_dir + "csv/" + str(station) + "/" + date.replace(".nc", ".csv")
        )


def find_fundamental_node(in_path="./results/raydec/csv/"):
    for file in os.listdir(in_path):
        df = pd.read_csv(in_path + file)

        peaks, _ = find_peaks(df["median"], height=0.7*df["median"].max())

        plt.clf()

        plt.plot(df["median"])
        plt.axvline(peaks[0], color="grey")
        plt.savefig("./results/figures/f_0/"+ file.replace(".csv", ".png"))

        





def save_to_csv():
    stations = [
        "453024025",
        "453024237",
        "453024321",
        "453024323",
        "453024387",
        "453024446",
        "453024510",
        "453024527",
        "453024614",
        "453024625",
        "453024645",
        "453024702",
        "453024704",
        "453024708",
        "453024718",
        "453024741",
        "453024928",
        "453024952",
        "453024968",
        "453025009",
        "453025057",
        "453025088",
        "453025089",
        "453025097",
        "453025215",
        "453025226",
        "453025229",
        "453025242",
        "453025257",
        "453025354",
        "453025361",
        "453025390",
    ]
    dates = [
        ["0011-2024-06-16", "0070-2024-08-22"],
        ["0012-2024-06-16", "0028-2024-07-01", "0066-2024-08-15"],
        ["0026-2024-06-30", "0071-2024-08-21"],
        ["0013-2024-06-17", "0054-2024-08-04"],
        ["0027-2024-07-03", "0062-2024-08-15"],
        ["0009-2024-06-14", "0055-2024-08-07"],
        ["0009-2024-06-13", "0037-2024-07-18", "0054-2024-08-03"],
        ["0026-2024-07-01", "0034-2024-07-17", "0063-2024-08-14"],
        ["0004-2024-06-09", "0038-2024-07-20", "0064-2024-08-14"],
        ["0005-2024-06-09", "0038-2024-07-20"],
        ["0006-2024-06-11", "0047-2024-07-30"],
        ["0005-2024-06-11", "0037-2024-07-21", "0067-2024-08-19"],
        ["0023-2024-06-27", "0068-2024-08-18"],
        ["0022-2024-06-28", "0039-2024-07-24", "0059-2024-08-12"],
        ["0007-2024-06-11", "0023-2024-06-27", "0065-2024-08-16"],
        ["0004-2024-06-09", "0039-2024-07-21"],
        ["0012-2024-06-17", "0020-2024-06-24"],
        ["0005-2024-06-09", "0053-2024-08-04"],
        ["0029-2024-07-04", "0065-2024-08-16"],
        ["0010-2024-06-16", "0044-2024-07-27", "0052-2024-08-03"],
        ["0027-2024-07-01", "0040-2024-07-21", "0064-2024-08-13"],
        ["0009-2024-06-14", "0026-2024-07-01", "0039-2024-07-21"],
        ["0002-2024-06-06", "0039-2024-07-21", "0068-2024-08-19"],
        ["0025-2024-06-29", "0068-2024-08-18"],
        ["0007-2024-06-12", "0022-2024-06-26", "0067-2024-08-17"],
        ["0017-2024-06-22", "0071-2024-08-22"],
        ["0006-2024-06-10", "0026-2024-06-29", "0039-2024-07-20"],
        ["0017-2024-06-21", "0027-2024-06-30", "0040-2024-07-20", "0071-2024-08-19"],
        ["0004-2024-06-09", "0018-2024-06-23", "0039-2024-07-21"],
        ["0029-2024-07-04", "0046-2024-07-28"],
        ["0007-2024-06-11", "0052-2024-08-04"],
        ["0026-2024-07-01"]
    ]

    order = [
        [3, 3],
        [1, 3, 3],
        [3, 3],
        [3, 3],
        [3, 3],
        [3, 1],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3],
        [3, 3],
        [3, 3, 3],
        [3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3],
        [3, 3],
        [3, 3],
        [3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3],
        [3, 3, 3],
        [3, 3],
        [3, 3, 3],
        [3, 3, 3, 3],
        [3, 3, 3],
        [3, 3],
        [3, 3],
        [3]
    ]

    in_dir = "./results/raydec/0-2-dfpar-QC/"
    for i in range(len(stations)):
        print(stations[i])
        for file in os.listdir(in_dir + stations[i]):
            for date_ind in range(len(dates[i])):
                if dates[i][date_ind] in file:
                    da_raydec = xr.open_dataarray(in_dir + "/" + stations[i] + "/" + file)
                    # filter out outliers
                    da_raydec = da_raydec[:, da_raydec["QC_" + str(order[i][date_ind] - 1)] == 0]

                    da_raydec["std"] = da_raydec["std_" + str(order[i][date_ind] - 1)]
                    da_raydec["mean"] = da_raydec["mean_" + str(order[i][date_ind] - 1)]
                    da_raydec["median"] = da_raydec["median_" + str(order[i][date_ind] - 1)]

                    for c in da_raydec.coords:
                        if c not in [
                            "freqs",
                            "wind",
                            "std",
                            "mean",
                            "median",
                        ]:
                            da_raydec = da_raydec.drop(c)

                    da_raydec.to_dataframe(name="ellipticity").to_csv(
                        "./results/raydec/0-2-dfpar-csv/"
                        + str(stations[i])
                        + "_"
                        + file.replace(".parquet.nc", ".csv")
                    )


def stack_station_windows():
    in_dir = "./results/raydec/"
    for station in os.listdir(in_dir):

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

    return da_raydec


def stack_station_windows(ind):
    in_dir = "./results/raydec/"
    station = get_station_list()[ind]
    if station not in ["stacked", "csv"]:
        data_arrays = []
        files = os.listdir(in_dir + station)
        for file in files:
            # to dataframe
            da_raydec = xr.open_dataarray(in_dir + station + "/" + file).to_dataset(
                name="ellipticity"
            )
            # print(da_raydec.coords)
            # print(da_raydec.dims)
            # da_raydec.coords["file"] = file
            # da_raydec.expand_dims("file")
            data_arrays.append(da_raydec)

        # da_stack = xr.combine_by_coords(data_arrays, data_vars="ellipticity")
        # da_stack = xr.combine_nested(data_arrays, concat_dim=["ellipticity"])
        da_stack = xr.combine_nested(data_arrays, concat_dim="files")
        # da_stack = xr.combine_by_coords(data_arrays, data_vars="ellipticity", coords=["wind", "freqs"])
        # da_stack = xr.combine_by_coords(data_arrays, data_vars="ellipticity", coords=["wind", "freqs"])
        make_output_folder("./results/raydec/stacked/")
        # write station df to csv

        da_stack.to_dataframe().to_csv(
            "./results/raydec/stacked/csv/" + str(station) + "_full.csv"
        )

        da_stack.to_netcdf("./results/raydec/stacked/nc/" + str(station) + "_full.nc")


if __name__ == "__main__":
    """
    run from terminal
    """
    # ind = int(sys.argv[1])
    # #SBATCH --array=1-32 #838
    # python src/process_data.py $SLURM_ARRAY_TASK_ID
    # sbatch slice_timeseries_job.slurm

    #ind = int(sys.argv[1])
    # ind = 2
    save_to_csv()
