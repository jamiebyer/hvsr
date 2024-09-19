import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np
from raydec import raydec
import matplotlib.pyplot as plt
from obspy import read


def test_raydec():
    stream_east = read("./data/453025390.0029.2024.07.04.00.00.00.000.E.miniseed", format="mseed")
    stream_north = read("./data/453025390.0029.2024.07.04.00.00.00.000.N.miniseed", format="mseed")
    stream_vert = read("./data/453025390.0029.2024.07.04.00.00.00.000.Z.miniseed", format="mseed")
    
    trace_east = stream_east.traces[0]
    trace_north = stream_north.traces[0]
    trace_vert = stream_vert.traces[0]

    times = trace_east.times()
    n_wind=int(np.round(times[-1])/30) # 30 second windows

    # raydec
    V, W = raydec(
        vert=trace_vert.data,
        north=trace_north.data,
        east=trace_east.data,
        time=times,
        fmin=0.0001,
        fmax=50,
        fsteps=1000,
        cycles=10,
        dfpar=0.1,
        nwind=n_wind
    )

    # save matrix. 

    #print(V.shape, np.min(V), np.max(V))
    print(W.shape, np.min(W), np.max(W))

    plt.plot(V[:, 0], W[:, 0])
    plt.show()


test_raydec()
