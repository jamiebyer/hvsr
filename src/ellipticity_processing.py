import numpy as np
import pandas as pd
import os
from raydec import raydec
from utils import make_output_folder
from dateutil import tz
import sys
import json
import time


###### RAYDEC PROCESSING ######


def get_ellipticity(
    station, date, fmin=0.8, fmax=40, fsteps=300, cycles=10, dfpar=0.1, len_wind=60
):
    # loop over saved time series files
    # raydec
    # number of windows based on size of slice
    dir_in = "./results/timeseries/" + str(station) + "/" + date + ".csv"
    df_in = pd.read_csv(dir_in).dropna()
    if len(df_in["times"]) < 1:
        return None

    times = df_in["times"].dropna().values
    times -= times[0]
    n_wind = int(np.round(times[-1] / len_wind))

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

    df_timeseries = pd.DataFrame(ellips.T, columns=freqs[:, 0])

    #df_timeseries["outliers"] = (np.abs(df_timeseries["vert"]) >= max_amplitude).astype(
    #    int
    #)
    return df_timeseries


def write_json(raydec_info, filename="./results/raydec/raydec_info.json"):
    with open(filename, "r+") as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        if (len(file_data["raydec_info"])) == 0:
            file_data["raydec_info"].append(raydec_info)
        else:
            for i in range(len(file_data["raydec_info"])):
                if raydec_info["name"] == file_data["raydec_info"][i]["name"]:
                    file_data["raydec_info"][i] = raydec_info
                elif i == len(file_data["raydec_info"]) - 1:
                    file_data["raydec_info"].append(raydec_info)
        # Sets file's current position at offset.
        file.seek(0)
        json.dump(file_data, file, indent=4)


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
    raydec_df = get_ellipticity(
        station,
        date,
        f_min,
        f_max,
        f_steps,
        cycles,
        df_par,
        len_wind,
    )
    if raydec_df is None:
        return None

    make_output_folder("./results/raydec/")
    make_output_folder("./results/raydec/" + str(station) + "/")
    # write station df to csv

    suffix = str(time.time()).split(".")[-1]
    raydec_df.to_csv(
        "./results/raydec/" + str(station) + "/" + date + "-" + suffix + ".csv"
    )

    # python object to be appended
    raydec_info = {
        "name": str(station) + "/" + date + "-" + suffix,
        "station": station,
        "date": date,
        "f_min": f_min,
        "f_max": f_max,
        "f_steps": f_steps,
        "cycles": cycles,
        "df_par": df_par,
        "n_wind": raydec_df.shape,
        "len_wind": len_wind,
    }

    write_json(raydec_info)

    return date


def remove_window_bounds():
    pass


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
    df_raydec = pd.concat([df_raydec, outlier_inds])

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

    f_min = 0.8
    f_max = 20
    f_steps = 300
    cycles = 10
    df_par = 0.1
    len_wind = 60

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

    print("done")


def sensitivity_test(ind):
    # try a range of frequencies, of df_par
    station = 24614
    date = "2024-06-15"

    params = []

    f_min, f_max = [0.8, 20]
    #for f_min, f_max in [[0.8, 20]]
    
    f_steps = 150
    
    cycles = 10
    #for cycles in np.arange(8, 13):
    
    df_par = 0.1
    #for df_par in np.linspace(0.05, 0.15, 10):

    len_wind = 3 * 60
    #for len_wind in np.linspace(60, 10*60, 10):

    for len_wind in np.linspace(60, 10*60, 10):
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


def stack_station_windows(station, date_range, raydec_properties):
    # search through json for files with the correct station, dates, properties

    # average the dispersion curves from each file

    # calculate range in values, error

    filename = "./results/raydec/raydec_info.json"
    with open(filename, "r+") as file:
        # First we load existing data into a dict.
        file_data = json.load(file)

    to_stack = []
    for saved_run in range(len(file_data["raydec_info"])):
        if raydec_info["name"] == file_data["raydec_info"][i]["name"]:
            for k, v in raydec_properties:
                if saved_run[k] != v:
                    continue
            to_stack.append(saved_run)

    for s in to_stack:
        # get csv

        # append average to new df
        pass



if __name__ == "__main__":
    """
    run from terminal
    """

    # #SBATCH --array=1-32 #838
    # python src/process_data.py $SLURM_ARRAY_TASK_ID
    # sbatch slice_timeseries_job.slurm

    ind = int(sys.argv[1])

    # process_station_ellipticity(ind)
    sensitivity_test(ind)
