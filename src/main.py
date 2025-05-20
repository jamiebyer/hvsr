import sys
from processing.timeseries_processing import *
from processing.ellipticity_processing import *
from processing.hvsr_processing import *
from processing.data_parsing import *
from processing.map_parsing import *

from plotting.timeseries_plotting import *
from plotting.ellipticity_plotting import *
from plotting.map_plotting import *

import os


def create_file_list(ind, in_path, suffix):
    # Looping over all stations
    file_path = []
    for station in os.listdir(in_path):
        for date in os.listdir(in_path + "/" + station):
            if not os.path.isfile(in_path + "/" + station + "/" + date):
                continue

            file_path.append([station, date])

    station = file_path[ind][0]
    date = file_path[ind][1]

    return station, date.replace(suffix, "")


if __name__ == "__main__":
    """
    run from terminal
    """

    # ind = int(sys.argv[1])

    # Launch app
    """
    from app.app import app
    app.run_server(debug=True, host="0.0.0.0", port=8050)
    """

    # plot_station_noise()
    # plot_drillholes()
    # plot_stations_wells_map()

    determine_ellipticity_outliers(in_path_timeseries, in_path_raydec, out_path)
