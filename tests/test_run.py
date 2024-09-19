import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np
from raydec import raydec
import matplotlib.pyplot as plt
from obspy import read
from scipy.io import loadmat

import pandas as pd


def test_raydec():
    df = pd.read_csv("./timeseries/24025/2024-06-18.csv")

    total_seconds = df.iloc[-1, df.columns.get_loc("times")]
    n_wind=int(np.round(total_seconds/30)) # 30 second windows
    
    # raydec
    V, W = raydec(
        vert=df["vert"],
        north=df["north"],
        east=df["east"],
        time=df["times"],
        fmin=0.01,
        fmax=20,
        fsteps=100,
        cycles=10,
        dfpar=0.1,
        nwind=n_wind
    )
    
    # save matrix. 
    freqs = V[:, 0]
    ellips = W

    df = pd.DataFrame(ellips, index=freqs)
    df.to_csv("./tests/python_test_file.csv")


def compare_raydec():
    python_results = pd.read_csv("./tests/python_test_file.csv", index_col=0)
    matlab_results = loadmat("./tests/matlab_test_file.mat")

    # checking that the frequencies match between matlab and python raydec
    python_freqs = python_results.index.values
    python_ellips = python_results.values

    matlab_freqs = matlab_results["V"][:, 0]
    matlab_ellips = matlab_results["W"]

    assert np.all(np.isclose(python_freqs, matlab_freqs))

    # check that ellipticities match
    py_e = python_ellips
    py_inds = not list(np.isnan(py_e))
    py_e = py_e[py_inds]

    mat_e = matlab_ellips
    mat_inds = not list(np.isnan(mat_e))
    mat_e = mat_e[mat_inds]

    assert np.all(np.isclose(py_e, mat_e))


if __name__ == "__main__":
    """
    run from terminal
    """
    compare_raydec()