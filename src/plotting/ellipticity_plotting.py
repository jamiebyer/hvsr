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
from scipy import stats
from plotting.map_plotting import plot_map
from processing.hvsr_processing import microtremor_hvsr_diffuse_field
import datetime
from matplotlib.colors import LogNorm


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
    in_path="./results/ellipticity/",
    out_path="./figures/ellipticity/",
):
    da_raydec = xr.open_dataarray(in_path + station + "/" + date + ".nc")
    da_raydec = da_raydec.dropna(dim="freqs")
    da_raydec["median"] = da_raydec.median(dim="wind")

    plt.plot(da_raydec["freqs"], da_raydec.values, c="grey", alpha=0.2)
    plt.plot(da_raydec["freqs"], da_raydec["median"], c="black")
    plt.xscale("log")

    plt.title("station " + str(station) + ", " + str(date).split("_")[1])
    plt.xlabel("frequency (Hz)")
    plt.xlim([da_raydec["freqs"][0], da_raydec["freqs"][-1]])
    plt.ylabel("ellipticity")
    make_output_folder(out_path + str(station) + "/")
    plt.savefig(out_path + station + "/" + date + ".png")




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


def plot_spatial_ellipticity():
    # plot median, stacked curve, with errors
    from matplotlib.gridspec import GridSpec

    # Create a figure
    fig = plt.figure(figsize=(12, 8))

    # Define GridSpec layout
    gs = GridSpec(4, 6, figure=fig)

    # phase 1 stations
    p1_stations = [
        [23, 24, 25, 26, 27, 28],
        [17, 18, 19, 22, 20, 21],
        [9, 12, 10, 13, 11, 14],
        [2, 3, 5, 4, 6, 8],
    ]

    for r in range(len(p1_stations)):
        for c in range(len(p1_stations[r])):
            station = p1_stations[r, c]
            ax = fig.add_subplot(gs[r, c])

            # get path for this phase and station
            ax.plot([1, 2, 3], [4, 5, 6])
            ax.set_title(station)

    plt.savefig()


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


def compare_hvsr_raydec(in_path_hvsr, in_path_raydec, out_path):
    # get hvsr
    # f_name = "./data/example_site/453024237.0005.2024.06.09.00.00.00.000.E.miniseed"
    # f_name = "./results/timeseries/example_timeseries_slice_E.miniseed"
    f_name = in_path_hvsr
    fnames = [[f_name, f_name.replace(".E.", ".N."), f_name.replace(".E.", ".Z.")]]

    srecords, hvsr = microtremor_hvsr_diffuse_field(fnames)

    # get ellipticity
    da_ellipticity = xr.open_dataarray(in_path_raydec)

    # plt.subplot(1, 2, 1)
    plt.plot(hvsr.frequency, hvsr.amplitude)
    # plt.xscale("log")
    # plt.xlim([0.8, 50])

    # plt.subplot(1, 2, 2)

    plt.plot(da_ellipticity.freqs, da_ellipticity.median(dim="wind"))
    plt.xscale("log")
    plt.xlim([0.8, 50])

    plt.legend(["hvsr", "ellipticity"])

    print(out_path)
    plt.savefig(out_path)


def plot_hvsr_station():
    # get hvsr
    # f_name = "./data/example_site/453024237.0005.2024.06.09.00.00.00.000.E.miniseed"
    f_name = "./results/timeseries/example_timeseries_slice_E.miniseed"
    fnames = [[f_name, f_name.replace("_E.", "_N."), f_name.replace("_E.", "_Z.")]]

    srecords, hvsr = microtremor_hvsr_diffuse_field(fnames)

    in_path = "./results/raydec/"
    site = "06"

    meds = []
    for f in os.listdir(in_path + site + "/"):
        da = xr.open_dataarray(in_path + site + "/" + f)
        meds.append(da.median(dim="wind"))

    plt.plot(hvsr.frequency, hvsr.amplitude)

    # plt.plot(meds, da.freqs)
    plt.imshow(np.array(meds).T)
    # plt.xscale("log")
    # plt.xlim([0.8, 50])

    plt.savefig("./results/figures/ellipticity/ellipticity_timeseries.png")


    meds = []
    for f in os.listdir(in_path + site + "/"):
        da = xr.open_dataarray(in_path + site + "/" + f)
        meds.append(da.median(dim="wind"))

    # plt.plot(meds, da.freqs)
    plt.imshow(np.array(meds).T)
    # plt.xscale("log")
    # plt.xlim([0.8, 50])

    plt.savefig("./results/figures/ellipticity/ellipticity_timeseries.png")


def all_station_ellipticities():
    in_path_hvsr = "./results/timeseries/sorted/"
    in_path_raydec = "./results/raydec/"
    # in_path = "./results/timeseries/sorted/"
    out_path = "./results/figures/ellipticity/examples/"
    # sites
    # sites = ["06", "07A", "17", "23", "24", "25", "32B", "34A", "38B", "41A", "41B", "42B", "47", "50"]
    sites = ["06"]
    # sites = os.listdir(in_paths)

    in_paths_hvsr = []
    in_paths_raydec = []
    # in_paths = []
    out_paths = []
    make_folders = False
    for s in sites:
        # for f in os.listdir(in_path + s + "/"):
        for f in os.listdir(in_path_hvsr + s + "/"):
            if ".E." not in f:
                continue
            if make_folders:
                make_output_folder(out_path + s + "/")
            in_paths_hvsr.append(in_path_hvsr + s + "/" + f)
            in_paths_raydec.append(in_path_raydec + s + "/" + f.replace(".E.miniseed", ".nc"))
            #in_paths.append(in_path + s + "/" + f)
            out_paths.append(out_path + s + "/" + f.replace(".E.miniseed", ".png"))
            # out_paths.append(out_path + s + "/" + f.replace(".E.miniseed", ".nc"))

    compare_hvsr_raydec(in_paths_hvsr[ind], in_paths_raydec[ind], out_paths[ind])
    # example_ellipticity(in_paths[ind], out_paths[ind])



def plot_raydec_station_timeseries_full(ind):
    in_path = "./results/ellipticity/"
    out_path = "./figures/ellipticity_timeseries/"
    # sites
    # sites = ["06", "07A", "17", "23", "24", "25", "32B", "34A", "38B", "41A", "41B", "42B", "47", "50"]
    sites = os.listdir(in_path)

    in_paths = []
    out_paths = []
    for s in sites:
        in_paths.append(in_path + s + "/")
        out_paths.append(out_path + s + ".png")
    
    plot_raydec_station_timeseries(in_paths[ind], out_paths[ind])


def plot_raydec_station_timeseries(in_path, out_path):
    dates = []
    meds = []
    for f in os.listdir(in_path):
        date = f.split("_")[1].removesuffix(".nc").split("-")
        dates.append(datetime.datetime(year=int(date[0]), month=int(date[1]), day=int(date[2])))
        da = xr.open_dataarray(in_path + f)
        meds.append(da.median(dim="wind"))
    
    inds = np.argsort(dates)
    dates = np.array(dates)[inds]
    meds = np.array(meds)[inds]


    extent = [np.log10(np.min(da["freqs"])), np.log10(np.max(da["freqs"])), dates[0], dates[-1]]
    plt.imshow(meds, cmap="coolwarm", extent=extent, aspect="auto", origin='lower', interpolation="none")
    
    #plt.xscale("log")
    plt.xlabel("frequency (Hz)")
    plt.xlim([np.log10(0.06), np.log10(np.max(da["freqs"]))])
    
    plt.xticks([-1, 0, 1], ["10^-1", "10^0", "10^1"])

    plt.colorbar()

    plt.rcParams["figure.figsize"] = (8,6)
    plt.tight_layout()
    plt.savefig(out_path)


def plot_raydec_station_timeseries_og(in_path, out_path):
    dates = []
    meds = []
    for f in os.listdir(in_path):
        date = f.split("_")[1].removesuffix(".nc").split("-")
        dates.append(datetime.datetime(year=int(date[0]), month=int(date[1]), day=int(date[2])))
        da = xr.open_dataarray(in_path + f)
        meds.append(da.median(dim="wind"))

    extent = [np.min(da["freqs"]), np.max(da["freqs"]), dates[0], dates[-1]]
    plt.imshow(meds, cmap="coolwarm", extent=extent, aspect="auto")
    #plt.yticks()
    plt.xscale("log")
    plt.xlabel("freq")

    plt.colorbar()
    plt.rcParams["figure.figsize"] = (8,6)
    plt.tight_layout()
    plt.savefig(out_path)



def plot_hvsr_station_timeseries_full(ind):
    in_path = "./results/timeseries/sorted/"
    out_path = "./figures/hvsr_timeseries/examples/"
    # sites
    sites = ["06", "07A", "17", "23", "24", "25", "32B", "34A", "38B", "41A", "41B", "42B", "47", "50"]
    # sites = os.listdir(in_paths)

    in_paths = []
    out_paths = []
    for s in sites:
        in_paths.append(in_path + s + "/")
        out_paths.append(out_path + s + ".png")
    
    plot_hvsr_station_timeseries(in_paths[ind], out_paths[ind])


def plot_hvsr_station_timeseries(in_path, out_path):
    freqs = []
    hvsrs = []
    dates = []
    for f in os.listdir(in_path):
        if ".E." not in f:
            continue        
        f_name = in_path + f
        fnames = [[f_name, f_name.replace(".E.", ".N."), f_name.replace(".E.", ".Z.")]]
        srecords, hvsr = microtremor_hvsr_diffuse_field(fnames)

        date = f.split("_")[1].removesuffix(".E.miniseed").split("-")
        dates.append(datetime.datetime(year=int(date[0]), month=int(date[1]), day=int(date[2])))

        freqs.append(hvsr.frequency)
        hvsrs.append(hvsr.amplitude)

    #extent = [np.min(da["freqs"]), np.max(da["freqs"]), dates[0], dates[-1]]
    extent = [np.min(freqs[0]), np.max(freqs[0]), dates[0], dates[-1]]
    plt.imshow(hvsrs, cmap="coolwarm", extent=extent, aspect="auto")
    #plt.imshow(hvsrs, cmap="coolwarm", aspect="auto")

    #plt.yticks()
    plt.xscale("log")
    plt.xlabel("freq")

    plt.colorbar()
    # plt.rcParams["figure.figsize"] = (8,6)
    plt.tight_layout()
    plt.savefig(out_path)






def plot_compare_station_timeseries_full(ind):
    in_path_hvsr = "./results/timeseries/sorted/"
    in_path_raydec = "./results/ellipticity/examples/"
    out_path = "./figures/compare_timeseries/examples/"
    # sites
    sites = ["06", "07A", "17", "23", "24", "25", "32B", "34A", "38B", "41A", "41B", "42B", "47", "50"]
    # sites = ["06"]
    # sites = os.listdir(in_paths)

    in_paths_hvsr = []
    in_paths_raydec = []
    out_paths = []
    for s in sites:
        in_paths_hvsr.append(in_path_hvsr + s + "/")
        in_paths_raydec.append(in_path_raydec + s + "/")
        out_paths.append(out_path + s + ".png")
    
    plot_compare_station_timeseries(in_paths_hvsr[ind], in_paths_raydec[ind], out_paths[ind])


def plot_compare_station_timeseries(in_path_hvsr, in_path_raydec, out_path):
    freqs = []
    hvsrs = []
    dates = []
    for f in os.listdir(in_path_hvsr):
        if ".E." not in f:
            continue        
        f_name = in_path_hvsr + f
        fnames = [[f_name, f_name.replace(".E.", ".N."), f_name.replace(".E.", ".Z.")]]
        srecords, hvsr = microtremor_hvsr_diffuse_field(fnames)

        date = f.split("_")[1].removesuffix(".E.miniseed").split("-")
        dates.append(datetime.datetime(year=int(date[0]), month=int(date[1]), day=int(date[2])))

        freqs.append(hvsr.frequency)
        hvsrs.append(hvsr.amplitude)

    inds = np.argsort(dates)
    dates = np.array(dates)[inds]
    hvsrs = np.array(hvsrs)[inds]

    plt.subplot(2, 2, 1)
    
    extent = [np.log10(np.min(freqs[0])), np.log10(np.max(freqs[0])), dates[0], dates[-1]]
    # plt.imshow(hvsrs, cmap="coolwarm", extent=extent, aspect="auto", norm=LogNorm(), origin='lower', interpolation="none")
    #plt.imshow(hvsrs, cmap="coolwarm", extent=extent, aspect="auto", norm="linear", origin='lower', interpolation="none")
    plt.imshow(hvsrs, cmap="coolwarm", extent=extent, aspect="auto", origin='lower', interpolation="none")
    # plt.imshow(hvsrs, cmap="coolwarm", aspect="auto", norm=LogNorm())

    plt.xlim([np.log10(0.8), np.log10(20)])
    plt.xticks([0, 1], ["10^0", "10^1"])

    #plt.xscale("log")
    plt.xlabel("freq")

    plt.colorbar()
    plt.title("hvsr")

    plt.subplot(2, 2, 2)

    for h in hvsrs:
        plt.plot(freqs[0], h, c="grey", alpha=0.3)
    
    #plt.plot(freqs, hvsrs) # , c="grey", alpha=0.3)
    
    plt.xscale("log")
    plt.xlabel("freq")
    plt.title("hvsr")
    plt.xlim([0.8, 20])


    dates = []
    meds = []
    for f in os.listdir(in_path_raydec):
        date = f.split("_")[1].removesuffix(".nc").split("-")
        dates.append(datetime.datetime(year=int(date[0]), month=int(date[1]), day=int(date[2])))
        da = xr.open_dataarray(in_path_raydec + f)
        meds.append(da.median(dim="wind"))
    
    inds = np.argsort(dates)
    dates = np.array(dates)[inds]
    meds = np.array(meds)[inds]

    meds_norm = meds/np.linalg.norm(meds)

    plt.subplot(2, 2, 3)

    extent = [np.log10(np.min(da["freqs"])), np.log10(np.max(da["freqs"])), dates[0], dates[-1]]
    plt.imshow(meds_norm, cmap="coolwarm", extent=extent, aspect="auto", origin='lower', interpolation="none")
    
    #plt.xscale("log")
    plt.xlabel("freq")
    plt.title("raydec")
    plt.xlim([np.log10(0.8), np.log10(20)])
    
    plt.xticks([0, 1], ["10^0", "10^1"])

    plt.colorbar()

    
    plt.subplot(2, 2, 4)

    plt.plot(da["freqs"], np.array(meds_norm).T, c="grey", alpha=0.3)
    
    plt.xscale("log")
    plt.xlabel("freq")
    plt.title("raydec")
    plt.xlim([0.8, 20])


    plt.rcParams["figure.figsize"] = (20,25)
    plt.tight_layout()
    plt.savefig(out_path)





def plot_raydec_station_peaks_full(ind):
    in_path = "./results/ellipticity/examples/"
    out_path = "./figures/ellipticity_peaks/examples/"
    # sites
    sites = ["06", "07A", "17", "23", "24", "25", "32B", "34A", "38B", "41A", "41B", "42B", "47", "50"]
    # sites = os.listdir(in_paths)

    in_paths = []
    out_paths = []
    for s in sites:
        in_paths.append(in_path + s + "/")
        out_paths.append(out_path + s + ".png")
    
    plot_raydec_station_peaks(in_paths[ind], out_paths[ind])


def plot_raydec_station_peaks(in_path, out_path):
    dates = []
    meds = []
    peaks = []
    for f in os.listdir(in_path):
        #date = f.split("_")[1].removesuffix(".nc").split("-")
        #dates.append(datetime.datetime(year=int(date[0]), month=int(date[1]), day=int(date[2])))
        date = f.split("_")[1].removesuffix(".nc")
        dates.append(date)

        da = xr.open_dataarray(in_path + f)
        freqs = da["freqs"]
        med = da.median(dim="wind")
        
        i_peaks, _ = find_peaks(med)

        i_max_peak = i_peaks[np.argsort(med[i_peaks].values)[-3:]]
        x_max = list(freqs[i_max_peak])
        # peaks.append(x_max)
        peaks += x_max

        plt.plot(freqs, med)
    
    #plt.scatter(freqs[i_max_peak], med[i_max_peak])
    plt.axvline(stats.mode(peaks)[0])

    plt.legend(dates)
    #extent = [np.min(da["freqs"]), np.max(da["freqs"]), dates[0], dates[-1]]
    
    # plt.imshow(meds, cmap="coolwarm", extent=extent, aspect="auto")
    #plt.yticks()
    plt.xscale("log")
    plt.xlabel("freq")

    #plt.colorbar()
    plt.rcParams["figure.figsize"] = (16,8)
    plt.tight_layout()
    plt.savefig(out_path)




def plot_averaged_station_ellipticity():
    pass

def plot_raydec_quality_control():
    in_path_timeseries = "./results/timeseries/examples/"
    in_path_raydec = "./results/ellipticity/examples/"
    out_path = "./figures/quality_control/examples/"

    raydec_quality_control(in_path_timeseries, in_path_raydec, out_path)


def plot_raydec_processing():
    # plot raydec windowing
    # plot raydec detrending
    # filtering for each frequency
    pass
