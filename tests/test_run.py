import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np
from raydec import raydec




def test_raydec():
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
    vel_phase += noise

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
