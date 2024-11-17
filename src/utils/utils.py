import os
import datetime


def make_output_folder(dir_path):
    if not os.path.isdir(dir_path) and not os.path.isfile(dir_path):
        os.mkdir(dir_path)


def create_file_list(ind ,in_path):
    files = []
    for station in os.listdir(in_path):
        for file in os.listdir(in_path + station):
            files.append((station, file.split(".")[0]))

    return files[ind][0], files[ind][1]


# functions for parsing xml


def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False


def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def is_date(val):
    try:
        datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
        return True
    except ValueError:
        return False
