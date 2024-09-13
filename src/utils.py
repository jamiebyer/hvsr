import os

def make_output_folder(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)