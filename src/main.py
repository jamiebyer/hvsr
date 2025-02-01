import sys
from processing.timeseries_processing import label_spikes, get_time_slice
from processing.ellipticity_processing import *
from processing.data_parsing import *

from plotting.timeseries_plotting import *
from plotting.ellipticity_plotting import *
from plotting.map_plotting import *

import os


# Looping over all stations


def create_file_list(ind, in_path, suffix):

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

    get_station_file_mapping()
    plot_station_schedule()

    # Launch app
    """
    from app.app import app

    app.run_server(debug=True, host="0.0.0.0", port=8050)
    """
    # parse_xml()

    # Process timeseries
    """
    df, station, date = create_file_list(ind, "./results/timeseries/raw/", ".parquet")
    df = pd.read_parquet(in_path + station + "/" + date)
    df = label_spikes2(df, station, date)
    df = get_time_slice2(df)
    #make_output_folder("./results/timeseries/clipped/" + station)
    df.to_parquet("./results/timeseries/raw/" + station + "/" + date)
    """
    # Plot timeseries
    """
    in_path="./results/timeseries/clipped/"
    station, date = create_file_list(ind, in_path, ".parquet")
    plot_timeseries(station, date)
    """
    # Process ellipticity
    # process_station_ellipticity(ind)

    # Plot ellipticity
    """
    station, date = create_file_list(ind, "./results/raydec/", ".nc")
    make_output_folder("./results/figures/ellipticity/")
    plot_ellipticity(station, date, in_path="./results/raydec/", out_path="./results/figures/ellipticity/")
    """

    # label window outliers
    """
    in_path = "./results/raydec/0-2-dfpar/"
    out_path = "./results/raydec/0-2-dfpar-QC/"
    station, date = create_file_list(ind, in_path, ".nc")
    
    da_raydec = xr.open_dataarray(in_path + "/" + station + "/" + date + ".nc")
    
    da_raydec = label_window_outliers(da_raydec, order=3, std_n=3)

    # save labeled outliers back to nc
    make_output_folder(out_path)
    make_output_folder(out_path + station)
    da_raydec.to_dataset(name="ellipticity").to_netcdf(out_path + station + "/" + date + ".nc")
    """

    # plot ellipticity QC
    # plot_ellipticity_outliers()
    # save_to_csv()

    # plot_best_csv()

    # plot_f_0_map()
    """
    get_station_positions(0)
    get_station_positions(1)
    get_station_positions(2)
    """

    pass
