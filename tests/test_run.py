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

    # raydec
    V, W = raydec(
        vert=np.round(trace_vert.data, 7),
        north=np.round(trace_north.data, 7),
        east=np.round(trace_east.data, 7),
        time=trace_east.times(),
        fmin=1,
        fmax=20,
        fsteps=100,
        cycles=10,
        dfpar=0.1,
        nwind=24*60
    )

    # save matrix. 

    #print(V.shape, np.min(V), np.max(V))
    print(W.shape, np.min(W), np.max(W))

    plt.plot(V[:, 0], W[:, 0])
    plt.show()

def test_raydec_og():
    #f_min = 1 # Hz
    #f_max = 20 # Hz
    f_min = 0.5 # Hz
    f_max = 5 # Hz

    freq_sampling = 2 * f_max # samples/second
    
    n_wind = 3 #10
    #t_max = n_wind*10*60
    t_max = n_wind*10*20
    n_steps = freq_sampling * t_max

    freq = np.linspace(f_min, f_max, n_steps) #Hz
    time = np.linspace(0, t_max, n_steps) #s

    period = 1/freq # s
    omega = 2*np.pi/period # 
    vel_rayleigh = 3000 # m/s
    wavelength = vel_rayleigh / freq # m
    k_mag = 2*np.pi/wavelength

    # 3D phase velocity
    # phase velocity is angular frequency / wave vector

    k = np.random.rand(3)
    k_hat = k / np.linalg.norm(k)
    k = np.outer(k_mag, k_hat)

    vel_phase = np.outer(omega, 1/k)

    # 3D group velocity
    # group velocity is the gradient of angular frequency as a function of k

    # add noise # check scale
    noise = np.random.normal(len(freq))
    vel_phase_noise = vel_phase + noise

    # raydec
    filtered_data = raydec(
        vert=vel_phase[0],
        north=vel_phase[1],
        east=vel_phase[2],
        time=time,
        fmin=f_min,
        fmax=f_max,
        fsteps=n_steps,
        cycles=10,
        dfpar=0.1,
        nwind=n_wind
    )

test_raydec()
