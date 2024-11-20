from processing.timeseries_processing import *
from processing.ellipticity_processing import *

from plotting.timeseries_plotting import *
from plotting.ellipticity_plotting import *


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

def process_data():
    # parsing xml....
    pass

def process_timeseries():
    # save data collection info, metadata whatever

    # convert miniseed on glados to parquet
    #convert_miniseed_to_parquet()

    # get stats from full timeseries,
    # get_timeseries_stats(include_outliers=True, in_path=r"./results/timeseries/raw/", out_path=r"./results/timeseries/stats/", out_file_name="full_timeseries")
    
    # label spikes and get stats for cleaned timeseries
    # get_timeseries_stats(include_outliers=False, in_path=r"./results/timeseries/raw/", out_path=r"./results/timeseries/stats/", out_file_name="full_timeseries_cleaned")

    # slice night,
    # save as netCDF

    # for one station and for all stations
    get_clean_timeseries_slice(False)



def process_ellipticity():
    pass


if __name__ == "__main__":
    """
    run from terminal
    """
    
    process_timeseries()
