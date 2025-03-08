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
from processing.hvsr_processing import microtremor_hvsr_diffuse_field

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
    raydec_fig.update_layout(title="453024237_0012-2024-06-16")
    make_output_folder(out_path + str(station) + "/")
    raydec_fig.write_image(out_path + station + "/" + date + ".png")


def plot_ellipticity_examples(out_path="./results/figures/ellipticity/"):
    in_path = "./results/raydec/examples/"

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    axes = [ax1, ax2, ax3]

    for ind, filename in enumerate(os.listdir(in_path)):
        da_raydec = xr.open_dataarray(in_path + filename)
        da_raydec = da_raydec.dropna(dim="freqs")
        da_raydec["median"] = da_raydec.median(dim="wind")
        # print(da_raydec)
        # break
        axes[ind].plot(
            da_raydec["freqs"].values,
            da_raydec.values,
            c=(0.6, 0.6, 0.6, 0.2),
            # log_x=True,
        )
        axes[ind].plot(
            da_raydec["freqs"].values,
            da_raydec["median"].values,
            c=(0, 0, 0),
            # log_x=True,
        )
        # axes[ind].set_title(filename)
        axes[ind].set_xscale("log")
        axes[ind].set_ylabel("ellipticity")

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


def compare_hvsr_raydec():
    # get hvsr
    # f_name = "./data/example_site/453024237.0005.2024.06.09.00.00.00.000.E.miniseed"
    f_name = "./results/timeseries/example_timeseries_slice_E.miniseed"
    fnames = [[f_name, f_name.replace("_E.", "_N."), f_name.replace("_E.", "_Z.")]]

    srecords, hvsr = microtremor_hvsr_diffuse_field(fnames)

    # get ellipticity
    da_ellipticity = xr.open_dataarray("./results/raydec/example_site.nc")

    # plt.subplot(1, 2, 1)
    plt.plot(hvsr.frequency, hvsr.amplitude)
    # plt.xscale("log")
    # plt.xlim([0.8, 50])

    # plt.subplot(1, 2, 2)

    plt.plot(da_ellipticity.freqs, da_ellipticity.median(dim="wind"))
    plt.xscale("log")
    plt.xlim([0.8, 50])

    plt.legend(["hvsr", "ellipticity"])

    plt.show()
