import numpy as np
import pandas as pd
import os
from processing.raydec import raydec
from utils.utils import make_output_folder
import time
import dask.dataframe as dd
import dask.array as da
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


def label_window_outliers(df_raydec):
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
    in_dir = "./results/raydec/"
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
    make_output_folder("./results/raydec/csv/")
    make_output_folder("./results/raydec/csv/" + station)
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
    df_par = 0.1
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
        mean = np.full(n_freqs, np.nan)
        s = np.full(n_freqs, np.nan)
        return s, mean, outlier_inds.data

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
        mean = np.full(n_freqs, np.nan)
        s = np.full(n_freqs, np.nan)
    else:

        # get mean with current valid windows
        mean = da_ellipticity[:, ~new_outlier_inds].mean(dim="wind")
        diff_from_mean = np.abs(da_ellipticity - mean)
        mean = mean.data

        # sample std
        # s = np.full(n_freqs, np.nan)
        s = np.sqrt(
            (1 / (n_valid - 1)) * np.sum(diff_from_mean[:, ~outlier_inds] * 2, axis=1)
        ).data

    return s, mean, new_outlier_inds.data


def label_window_outliers(df_raydec, order, std_n):
    """
    remove outlier windows with values further than 3 std from mean
    """
    N = len(df_raydec.coords["wind"])
    outlier_inds = np.zeros(N).astype(bool)

    for i in range(order):
        std, mean, new_outlier_inds = calc_stacked_std(std_n, df_raydec, outlier_inds)
        outlier_inds = outlier_inds | new_outlier_inds

        df_raydec = df_raydec.assign_coords(
            {
                "QC_" + str(i): ("wind", outlier_inds),
                "std_" + str(i): ("freqs", std),
                "mean_" + str(i): ("freqs", mean),
            }
        )

    return df_raydec


def label_all_window_outliers(ind, order, std_n, save_csv=False):
    in_dir = "./results/raydec/"
    # clean up
    station_list = []
    for station in os.listdir(in_dir):
        if station not in ["stacked", "csv", "QC_std1", "QC_std2", "QC_std3"]:
            for date in os.listdir(in_dir + station):
                if "-QC" not in date:
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


def save_to_csv():
    stations = [
        "24025",
        "24237",
        "24321",
        "24323",
        "24387",
        "24446",
        "24510",
        "24527",
        "24614",
        "24625",
        "24645",
        "24702",
        # "24704",
        "24708",
        "24718",
        "24741",
        "24928",
        "24952",
        "24968",
        "25009",
        "25088",
        "25089",
        "25097",
        "25215",
        "25226",
        "25229",
        "25242",
        "25257",
        "25354",
        "25361",
        "25390",
    ]
    dates = [
        "2024-07-03",
        "2024-06-14",
        "2024-06-13",
        "2024-07-04",
        "2024-07-03",
        "2024-06-14",
        "2024-07-04",
        "2024-07-01",
        "2024-06-09",
        "2024-07-04",
        "2024-07-03",
        "2024-06-30",
        # "",
        "2024-06-09",
        "2024-06-24",
        "2024-06-30",
        "2024-06-30",
        "2024-06-30",
        "2024-06-26",
        "2024-06-15",
        "2024-06-12",
        "2024-07-04",
        "2024-06-26",
        "2024-07-04",
        "2024-07-04",
        "2024-07-01",
        "2024-07-01",
        "2024-06-14",
        "2024-06-14",
        "2024-06-21",
        "2024-07-04",
    ]

    order = [
        3,
        1,
        3,
        3,
        3,
        1,
        1,
        3,
        3,
        1,
        3,
        1,
        # None,
        3,
        3,
        3,
        3,
        1,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        1,
        1,
        1,
        3,
        3,
    ]

    in_dir = "./results/raydec/QC_std2/"
    for i in range(len(stations)):
        print(stations[i])
        for file in os.listdir(in_dir + stations[i]):
            if dates[i] in file:
                da_raydec = xr.open_dataarray(in_dir + "/" + stations[i] + "/" + file)
                # filter out outliers
                print(da_raydec)
                da_raydec = da_raydec[:, da_raydec["QC_" + str(order[i] - 1)] == 0]

                print(da_raydec.coords)
                for c in da_raydec.coords:
                    if c not in [
                        "freqs",
                        "wind",
                        "std_" + str(order[i] - 1),
                        "mean_" + str(order[i] - 1),
                    ]:
                        da_raydec = da_raydec.drop(c)

                da_raydec.to_dataframe(name="ellipticity").to_csv(
                    "./results/best/csv/"
                    + str(stations[i])
                    + "_"
                    + file.replace(".nc", ".csv")
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

    ind = int(sys.argv[1])
    # ind = 2
    stack_station_windows(ind)
