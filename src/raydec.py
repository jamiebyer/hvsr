import numpy as np
from scipy.signal import cheb1ord, cheby1, lfilter, detrend


def raydec(vert, north, east, time, fmin, fmax, fsteps, cycles, dfpar, nwind):
    # from: # https://github.com/ManuelHobiger/RayDec
    """
    RAYDEC1STATION(VERT, NORTH, EAST, TIME, FMIN, FMAX, FSTEPS, CYCLES, DFPAR, NWIND)
    calculates the ellipticity of Rayleigh waves for the
    input data VERT, NORTH, EAST and TIME for a single station
    for FSTEPS frequencies (on a logarithmic scale) between
    FMIN and FMAX, using CYCLES periods for the stacked signal
    and DFPAR as the relative bandwidth for the filtering.
    The signal is cut into NWIND different time windows and
    RayDec is applied to each of them.

    VERT, NORTH, EAST and TIME have to be arrays of equal sizes

    suggested values: CYCLES = 10
    DFPAR = 0.1
    NWIND such that the single time windows are about 10 minutes long
    """

    v1, n1, e1, t1 = vert, north, east, time

    # setting up
    K0 = v1.shape[0]
    K = np.floor(K0 / nwind).astype(int)
    tau = t1[1] - t1[0]
    DTmax = 30
    fnyq = 1 / (2 * tau)
    fstart = np.max([fmin, 1 / DTmax])
    fend = np.min([fmax, fnyq])
    flist = np.zeros(fsteps)
    constlog = (fend / fstart) ** (1 / (fsteps - 1))
    fl = fstart * constlog ** (np.cumsum(np.ones((fsteps, nwind)), axis=0) - 1)
    el = np.zeros((fsteps, nwind))

    # loop over the time windows
    for ind1 in range(nwind):
        # *** matlab detrend -> scipy
        vert = detrend(v1[ind1 * K + 1 : (ind1 + 1) * K])
        north = detrend(n1[ind1 * K + 1 : (ind1 + 1) * K])
        east = detrend(e1[ind1 * K + 1 : (ind1 + 1) * K])
        time = t1[ind1 * K + 1 : (ind1 + 1) * K]

        horizontalamp = np.zeros(fsteps)
        verticalamp = np.zeros(fsteps)
        Tmax = np.max(time)
        thetas = np.zeros(fsteps, np.ceil(Tmax * fend))
        corr = np.zeros(fsteps, np.ceil(Tmax * fend))
        ampl = np.zeros(fsteps, np.ceil(Tmax * fend))
        dvmax = np.zeros(fsteps)

        # loop over the frequencies
        for findex in range(fsteps):
            f = fl[findex, ind1]

            # setting up the filter limits
            df = dfpar * f
            fmin = np.max([fstart, f - df / 2])
            fmax = np.min([fnyq, f + df / 2])
            flist[findex] = f
            DT = cycles / f
            wl = np.round(DT / tau)

            # setting up the Chebyshev filter
            Wp = [fmin + (fmax - fmin) / 10, fmax - (fmax - fmin) / 10] / fnyq
            Ws = [fmin - (fmax - fmin) / 10, fmax + (fmax - fmin) / 10] / fnyq
            Rp = 1
            Rs = 5

            N, Wn = cheb1ord(Wp, Ws, Rp, Rs)
            b, a = cheby1(N, 0.5, Wn, btype="bandpass")
            # w, h = signal.freqz(b, a)

            taper1 = np.arange(0, 1, 1 / np.round(time.shape[0] / 100))
            taper2 = np.ones(time.shape[0]- 2*taper1.shape[0])
            #taper2 = np.ones(time.shape[0]-2)
            taper3 = np.flip(taper1)

            print(time.shape)
            print(taper1.shape)
            print(taper2.shape)
            print(taper3.shape)

            taper = np.transpose([taper1, taper2, taper3])

            # filtering the signals
            norths = lfilter(b, a, taper * north)
            easts = lfilter(b, a, taper * east)
            verts = lfilter(b, a, taper * vert)

            derive = (
                np.sign(verts[1:K]) - np.sign(verts[: (K - 1)])
            ) / 2  # finding the negative-positive zero crossings on the vertical component

            vertsum = np.zeros(wl)
            horsum = np.zeros(wl)
            dvindex = 0

            # loop over all zero crossings
            # *** check bounds
            for index in range(np.ceil(1 / (4 * f * tau)) + 1, len(derive) - wl):
                if derive(index) == 1:
                    dvindex = dvindex + 1
                    vsig = verts[index : (index + wl - 1)]
                    esig = easts[
                        index
                        - np.floor(1 / (4 * f * tau)) : (
                            index - np.floor(1 / (4 * f * tau)) + wl - 1
                        )
                    ]
                    nsig = norths[
                        index
                        - np.floor(1 / (4 * f * tau)) : (
                            index - np.floor(1 / (4 * f * tau)) + wl - 1
                        )
                    ]
                    integral1 = np.sum(vsig * esig)
                    integral2 = np.sum(vsig * nsig)
                    theta = np.atan(integral1 / integral2)

                    if integral2 < 0:
                        theta = theta + np.pi

                    theta = (
                        theta + np.pi % 2 * np.pi
                    )  # The azimuth is well estimated in this way (assuming retrograde)
                    hsig = (
                        np.sin(theta) * esig + np.cos(theta) * nsig
                    )  # The horizontal signal is projected in the azimuth direction.
                    correlation = np.sum(vsig * hsig) / np.sqrt(
                        np.sum(vsig * vsig) * np.sum(hsig * hsig)
                    )  # The correlation is always negative (between -1 and 0).

                    if correlation >= -1:
                        vertsum = vertsum + correlation**2 * vsig
                        horsum = horsum + correlation**2 * hsig

                    thetas[findex, dvindex] = theta
                    corr[findex, dvindex] = correlation
                    dvmax[findex] = dvindex
                    ampl[findex, dvindex] = np.sum(vsig**2 + hsig**2)

            klimit = np.round(DT / tau)
            verticalamp[findex] = np.sqrt(np.sum(vertsum[:klimit] ** 2))
            horizontalamp[findex] = np.sqrt(np.sum(horsum[:klimit] ** 2))

        ellist = horizontalamp / verticalamp

        fl[:, ind1] = flist
        el[:, ind1] = ellist

    V = fl
    W = el

    return V, W
