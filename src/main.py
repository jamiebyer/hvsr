from processing.timeseries_processing import *
from processing.ellipticity_processing import *
from processing.data_parsing import *

from plotting.timeseries_plotting import *
from plotting.ellipticity_plotting import *
from plotting.map_plotting import *


# Looping over all stations

def create_file_list(ind):
    path = "./results/timeseries/clipped/"
    file_path = []
    for station in os.listdir(path):
        for date in os.listdir(path + "/" + station):
            if not os.path.isfile(path + "/" + station + "/" + date):
                continue

            file_path.append([station, date])

    station = file_path[ind][0]
    date = file_path[ind][1]

    df = pd.read_parquet(path + station + "/" + date)

    return df, station, date

def process_timeseries():
    # parsing xml....
    # save data collection info, metadata whatever

    # convert miniseed on glados to netCDF
    # get stats from full timeseries,
    # slice night,
    # remove outliers,
    # save as netCDF

    # for one station and for all stations

    pass


def process_ellipticity():
    pass


if __name__ == "__main__":
    """
    run from terminal
    """

    plot_map()
    #parse_xml()
