import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import plotly.express as px
import plotly.graph_objects as go
import json
from utils.utils import make_output_folder
import xarray as xr
from scipy.signal import find_peaks
from plotting.map_plotting import plot_map

# RAYDEC PLOT


def plot_raydec(da_raydec, station, date, fig_dict, scale_factor):
    # ideally the outliers are dropped and a new df is saved to read in,
    # but after a good threshold is set

    fig_dict = {
        "station": station,
        "date": date.rsplit("-", 1)[0],
        "fmin": da_raydec["fmin"].values,
        "fmax": da_raydec["fmax"].values,
        "fsteps": da_raydec["fsteps"].values,
        "nwind": len(da_raydec.coords["wind"].values),
        "cycles": da_raydec["cycles"].values,
        "dfpar": da_raydec["dfpar"].values,
    }

    # skip nans
    # plot raydec
    da_raydec = da_raydec.dropna(dim="freqs")
    # df_raydec.index = pd.to_numeric(df_raydec.index)

    # remove outlier windows
    # df_raydec = remove_window_outliers(df_raydec, scale_factor)

    # outliers = df_raydec.loc["outliers"]
    # df_raydec = df_raydec.drop("outliers")

    mean = da_raydec.mean(dim="wind")

    # df_raydec = df_raydec.drop("mean", axis=1)
    """outliers = outliers.drop("mean")

    df_outliers = df_raydec[outliers.index[outliers == 1]]
    df_keep = df_raydec[outliers.index[outliers == 0]]

    fig_dict["outliers"] = str(
        df_outliers.shape[1] / (df_outliers.shape[1] + df_keep.shape[1])
    )"""

    df_keep = da_raydec
    # stats = df_keep.describe()

    raydec_fig = px.line(
        df_keep,
        color_discrete_sequence=["rgba(100, 100, 100, 0.2)"],
        log_x=True,
    )

    raydec_fig.add_traces(
        # list(
        #    px.line(
        #        df_outliers,
        #        color_discrete_sequence=["rgba(255, 0, 0, 0.2)"],
        #        log_x=True,
        #    ).select_traces()
        # )
        # +
        list(
            px.line(
                pd.DataFrame(mean, df_keep.coords["freqs"].values),
                color_discrete_sequence=["rgba(0, 0, 0, 1)"],
                log_x=True,
            ).select_traces()
        )
    )

    groups = (
        ["mean"]
        # + df_outliers.shape[1] * ["outliers"]
        # ["mean", "max", "min"]
        + df_keep.shape[1] * ["keep"]
    )
    for ind, trace in enumerate(raydec_fig["data"]):
        trace["legendgroup"] = groups[ind]

    raydec_fig.update_layout(
        title=str(station) + ": " + str(date),
        xaxis_title="frequency (Hz)",
        yaxis_title="ellipticity",
        yaxis_range=[0, 10],
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
    )

    text = ""
    for k, v in fig_dict.items():
        if k == "name":
            continue
        text += k + ": " + str(v) + "<br>"

    raydec_fig.add_annotation(
        x=1,
        y=9,
        text=text,
        showarrow=False,
    )

    # plot confidence interval above/below

    return raydec_fig


def plot_sensitivity_test():
    station = 24614
    # json_path = "./results/raydec/raydec_info.json"
    for date in os.listdir("./results/raydec/24614/"):
        date = date.removesuffix(".nc")
        file_name = str(station) + "/" + str(date)
        path_raydec = "./results/raydec/" + file_name + ".nc"

        da_raydec = xr.open_dataarray(path_raydec)

        raydec_fig = plot_raydec(
            da_raydec, 24614, date.rsplit("-", 1)[0], scale_factor=1
        )

        raydec_fig.write_image(
            "./results/figures/sensitivity_analysis/" + date + ".png"
        )


def plot_ellipticity(
    station,
    date,
    in_path="./results/raydec/",
    out_path="./results/figures/ellipticity/",
):
    da_raydec = xr.open_dataarray(in_path + station + "/" + date + ".nc")

    da_raydec = da_raydec.dropna(dim="freqs")

    da_raydec["median"] = da_raydec.median(dim="wind")

    raydec_fig = px.line(
        da_raydec,
        color_discrete_sequence=["rgba(100, 100, 100, 0.2)"],
        log_x=True,
    )
    raydec_fig.update_layout(title=str(station) + ", " + str(date))
    make_output_folder(out_path + str(station) + "/")
    raydec_fig.write_image(out_path + station + "/" + date + ".png")


def plot_ellipticity_examples(out_path="./results/figures/ellipticity/"):
    in_path = "./results/raydec/examples/"

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(6, 8))
    axes = [ax1, ax2, ax3]

    examples = ["0069-2024-08-18", "0012-2024-06-16", "0006-2024-06-10"]

    print(os.listdir(in_path))
    for ind, filename in enumerate(examples):
        print(filename)
        da_raydec = xr.open_dataarray(in_path + filename + ".parquet.nc")
        da_raydec = da_raydec.dropna(dim="freqs")
        
        inds = np.squeeze(np.array([da_raydec["QC_2"] == False]).T)
        
        axes[ind].plot(
            da_raydec["freqs"].values,
            da_raydec.values[:, inds],
            c=(0.6, 0.6, 0.6, 0.05),
            # log_x=True,
        )
        axes[ind].plot(
            da_raydec["freqs"].values,
            da_raydec.values[:, inds].mean(axis=1),
            c=(0, 0, 0),
            # log_x=True,
        )
        # axes[ind].set_title(filename)
        axes[ind].set_xscale("log")
        axes[ind].set_ylabel("ellipticity")

        axes[ind].set_xlim([0.1, 20])
        axes[ind].set_ylim([0, 10])

    axes[2].set_xlabel("frequency (Hz)")

    plt.tight_layout()
    plt.savefig(out_path + "categories.png")


def plot_ellipticity_outliers():
    in_path = "./results/raydec/QC/"
    out_path = "./results/figures/ellipticity/"
    make_output_folder(out_path)
    for station in os.listdir(in_path):
        print(station)
        make_output_folder(out_path + str(station) + "/")
        for date in os.listdir(in_path + station):
            try:
                da_ellipticity = xr.open_dataarray(in_path + station + "/" + date)

                plt.close()

                fig, axs = plt.subplots(figsize=(20, 14), nrows=2, ncols=2)

                da_ellipticity.plot.line(
                    x="freqs",
                    ax=axs[0, 0],
                    color=(0.1, 0.1, 0.1, 0.1),
                    xscale="log",
                    add_legend=False,
                )

                if np.any(da_ellipticity["QC_0"] == False):
                    da_ellipticity[:, da_ellipticity["QC_0"] == False].plot.line(
                        x="freqs",
                        ax=axs[1, 0],
                        color=(0.1, 0.1, 0.1, 0.1),
                        xscale="log",
                        add_legend=False,
                    )
                axs[1, 0].set_title(np.sum(da_ellipticity["QC_0"].values))

                if np.any(da_ellipticity["QC_1"] == False):
                    da_ellipticity[:, da_ellipticity["QC_1"] == False].plot.line(
                        x="freqs",
                        ax=axs[0, 1],
                        color=(0.1, 0.1, 0.1, 0.1),
                        xscale="log",
                        add_legend=False,
                    )
                axs[0, 1].set_title(np.sum(da_ellipticity["QC_1"].values))

                if np.any(da_ellipticity["QC_2"] == False):
                    da_ellipticity[:, da_ellipticity["QC_2"] == False].plot.line(
                        x="freqs",
                        ax=axs[1, 1],
                        color=(0.1, 0.1, 0.1, 0.1),
                        xscale="log",
                        add_legend=False,
                    )

                axs[1, 1].set_title(np.sum(da_ellipticity["QC_2"].values))

                plt.tight_layout()

                plt.savefig(out_path + station + "/" + date.replace(".nc", ".png"))
            except AttributeError:
                continue


def plot_best_ellipticity():
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

    in_path = "./results/raydec/QC_std2/"
    out_path = "./results/figures/ellipticity/best/"
    make_output_folder(out_path)
    for i in range(len(stations)):
        station = stations[i]
        print(station)
        make_output_folder(out_path + str(station) + "/")
        for date in os.listdir(in_path + station):
            if dates[i] in date:
                da_ellipticity = xr.open_dataarray(in_path + station + "/" + date)

                plt.close()

                # fig, axs = plt.subplots(figsize=(20, 14), nrows=2, ncols=2)

                if np.any(da_ellipticity["QC_" + str(order[i] - 1)] == False):
                    da_ellipticity[
                        :, da_ellipticity["QC_" + str(order[i] - 1)] == False
                    ].plot.line(
                        x="freqs",
                        color=(0.1, 0.1, 0.1, 0.1),
                        xscale="log",
                        add_legend=False,
                    )
                plt.title(np.sum(da_ellipticity["QC_" + str(order[i] - 1)].values))

                plt.tight_layout()

                plt.savefig(out_path + station + "_" + date.replace(".nc", ".png"))


def plot_best_csv(
    in_path="./results/raydec/0-2-dfpar-csv/", out_path="./results/figures/0-2-dfpar/"
):
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
        # ["0004-2024-06-09", "0039-2024-07-21"],
        ["0039-2024-07-21"],
        ["0012-2024-06-17", "0020-2024-06-24"],
        ["0005-2024-06-09", "0053-2024-08-04"],
        ["0029-2024-07-04", "0065-2024-08-16"],
        ["0010-2024-06-16", "0044-2024-07-27", "0052-2024-08-03"],
        # ["0027-2024-07-01", "0040-2024-07-21", "0064-2024-08-13"],
        ["0027-2024-07-01", "0064-2024-08-13"],
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
        ["0026-2024-07-01"],
    ]
    """
    peaks_dict = {
        "453024025" = {

        },
        "453024321" = {
            
        },
        "453024323" = {
            
        },
        "453024510" = {
            
        },
        "453024614" = {
            
        },
        "453024645" = {
            
        },
        "453024704" = {
            
        },
        "453024708" = {
            
        },
        "453024741" = {
            
        },
        "453025088" = {
            
        },
        "453025097",
    }
    """
    peaks = [
        [3, 6],
        [0, 0, 0],
        [6, 3],
        [10, 11],
        [0, 0],
        [0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0],
        [0, 2],
        [0, 0, 0],
        [8, 0],  # didn't identify peak
        [4, 2, 0],
        [0, 0, 0],
        # [0, 0],
        [1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0, 0],
        # [0, 0, 0],
        [0, 0],
        [2, 8, 11],
        [0, 0, 0],
        [3, 0],
        [0, 0, 0],
        [0, 0],
        [0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0],
        [0, 0],
        [0, 0],
        [0],
    ]

    for s_ind in range(len(stations)):
        for d_ind in range(len(dates[s_ind])):
            station = stations[s_ind]
            date = dates[s_ind][d_ind]
            peak = peaks[s_ind][d_ind]
            file_name = station + "_" + date + ".csv"
            df = pd.read_csv(in_path + file_name)

            plt.clf()

            for w in np.unique(df["wind"]):
                plt.plot(
                    df["freqs"][df["wind"] == w],
                    df["ellipticity"][df["wind"] == w],
                    color=(0.8, 0.8, 0.8, 0.2),
                )

            all_peaks, _ = find_peaks(
                df["median"][df["wind"] == w],
                height=0.5 * df["median"][df["wind"] == w].max(),
            )

            plt.plot(df["freqs"][df["wind"] == w], df["median"][df["wind"] == w])

            # plt.scatter(df["freqs"][df["wind"]==w].values[all_peaks], df["median"][df["wind"]==w].values[all_peaks], color="black")

            [
                plt.axvline(
                    df["freqs"][df["wind"] == w].values[p],
                    color="black",
                    linestyle="dashed",
                )
                for p in all_peaks
            ]
            plt.axvline(
                df["freqs"][df["wind"] == w].values[all_peaks[peak]],
                color="red",
                linestyle="dashed",
            )

            plt.title(file_name.replace(".csv", ""))
            plt.xscale("log")

            plt.savefig(out_path + file_name.replace(".csv", ".png"))

