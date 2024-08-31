import numpy as np



def test_raydec():
    n_wind = 10
    time = np.linspace(0, n_wind*10*60) #s

    f_min = 1 # Hz
    f_max = 20 # Hz
    f_steps = 100

    freq = np.linspace(f_min, f_max, f_steps) #Hz
    period = 1/freq # s
    omega = 2*np.pi/period #
    vel_rayleigh = 3000 # m/s
    wavelength = vel_rayleigh / freq # m
    k_mag = 2*np.pi/wavelength

    # 3D phase velocity
    # phase velocity is angular frequency / wave vector

    k = np.random.rand(3)
    k_hat = k / np.linalg.norm(k)
    k = k_mag * k_hat

    vel_phase = omega/k

    # 3D group velocity
    # group velocity is the gradient of angular frequency as a function of k

    # add noise
    noise = np.random.normal(len(freq))
    vel_phase += noise

    #freq_sampling = 1/100
    #freq_nyq = freq_sampling / 2


    # raydec
    filtered_data = raydec(
        vert=vel_phase[0],
        north=vel_phase[1],
        east=vel_phase[2],
        time=time,
        fmin=f_min,
        fmax=f_max,
        fsteps=f_steps,
        cycles=10,
        dfpar=0.1,
        nwind=n_wind
    )