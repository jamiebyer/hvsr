import os
import datetime


def make_output_folder(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


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
