from processing.ellipticity_processing import label_all_window_outliers, save_to_csv
from plotting.ellipticity_plotting import (
    plot_ellipticity_outliers,
    plot_best_ellipticity,
)

if __name__ == "__main__":
    """
    run from terminal
    """

    save_to_csv()
    """
    std_n = 2
    for ind in range(900):
        print(ind)
        label_all_window_outliers(ind, 3, std_n)
    """
