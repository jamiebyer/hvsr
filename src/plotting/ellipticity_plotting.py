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


def plot_ellipticity():
    in_path = "./results/raydec/"
    out_path = "./results/figures/ellipticity/"
    make_output_folder(out_path)
    for station in os.listdir(in_path):
        make_output_folder(out_path + str(station) + "/")
        for date in os.listdir(in_path + "/" + station):

            # date = date.removesuffix(".nc").rsplit("-")[1]
            date = date.removesuffix(".nc")
            file_name = str(station) + "/" + str(date)
            path_raydec = "./results/raydec/" + file_name + ".nc"

            da_raydec = xr.open_dataarray(path_raydec)
            da_raydec = da_raydec.dropna(dim="freqs")
            # mean = da_raydec.mean(dim="wind")

            raydec_fig = px.line(
                da_raydec,
                color_discrete_sequence=["rgba(100, 100, 100, 0.2)"],
                log_x=True,
            )
            raydec_fig.write_image(
                "./results/figures/ellipticity/" + station + "/" + date + ".png"
            )


def plot_ellipticity_outliers():
    in_path = "./results/raydec/QC_std3/"
    out_path = "./results/figures/ellipticity/QC_3/"
    make_output_folder(out_path)
    for station in os.listdir(in_path)[30:35]:
        print(station)
        make_output_folder(out_path + str(station) + "/")
        for date in os.listdir(in_path + station):
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


# STACKING PLOT


def plot_stacking(ind):
    stations = os.listdir("./results/raydec/stacked/nc/")
    station = stations[ind]
    in_dir = "./results/raydec/stacked/nc/" + station  # + "_full.nc"
    da_raydec = xr.open_dataset(in_dir)

    freqs = np.array(da_raydec.coords["freqs"].values)
    mean = np.array(da_raydec.mean(["wind", "files"]).data_vars["ellipticity"].values)
    std = np.array(da_raydec.mean(["wind", "files"]).data_vars["ellipticity"].values)

    # create figure
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])

    # ax1 = fig.add_subplot(1, 2, 1)

    ax1.errorbar(
        freqs,
        mean,
        yerr=std,
        c="black",
        label="mean",
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("frequency (log)")
    ax1.set_ylabel("ellipticity")

    fig = plot_map(fig, gs, station.split("_full")[0])

    fig.suptitle(station + " (all days)")

    plt.savefig("./results/figures/stacked/" + station.replace(".nc", ".png"))
