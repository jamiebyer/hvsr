import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np
from raydec import raydec
import matplotlib.pyplot as plt
from obspy import read

import pandas as pd


def test_raydec():
    df = pd.read_csv("./timeseries/24025/2024-06-18.csv")
    #print(len(times), len(trace_east.data))
    #print(np.unique(times[1:]-times[:-1]))
    #print(np.sum(np.isnan(trace_east.data)))
    #print(stream_east)
    #print(stream_east.traces)
    #print(trace_east.__dict__)

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
        fsteps=1000,
        cycles=10,
        dfpar=0.1,
        nwind=n_wind
    )
    
    # save matrix. 

    #print(V.shape, np.min(V), np.max(V))
    print(W.shape, np.nanmin(W), np.nanmax(W))

    plt.plot(V[:, 0], W[:, 0])
    #plt.show()


test_raydec()
