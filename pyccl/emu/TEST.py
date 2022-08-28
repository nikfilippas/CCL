from emu22 import get_emu_data
import numpy as np
from scipy.linalg import cholesky, svd
from scipy.interpolate import Akima1DInterpolator

data = get_emu_data()

m = 111
neta = 2808
peta = 45
rs = 8
p = 8
nmode = 351


# Kriging basis computed by emu_init.
# Sizes of each basis will be `peta` and `m`.
KrigBasis = np.zeros((peta, m))



def emu_init():
    global KrigBasis
    b = np.zeros(m)

    # Loop over the basis.
    for i in range(peta):

        diag = 1 / data["lamz"][i] + 1 / data["lamws"][i]
        SigmaSim = np.diag(np.ones(m) * m)

        # Loop over the number of simulations.
        for j in range(m):

            # Off-diagonals.
            for k in range(j):

                # compute the covariance
                cov = 0.
                for l in range(p):
                    cov -= data["beta"][i, l] * (data["x"][j, l] - data["x"][k, l])**2
                cov = np.exp(cov) / data["lamz"][i]

                # Put the covariance where it belongs.
                SigmaSim[j, k] = cov
                SigmaSim[k, j] = cov

            # Vector for the PC weights.
            b[j] = data["w"][i, j]

        # Cholesky and solve.
        SigmaSim = cholesky(SigmaSim)

        # Put b where it belongs in the Kriging basis.
        KrigBasis[i] = svd(SigmaSim)[1]


def emu(xstar):
    if (KrigBasis == 0).all():
        emu_init()

    xstar = np.asarray(xstar)
    Sigmastar = np.empty((peta, m))
    wstar = np.empty(peta)
    ystaremu = np.empty(neta)
    ybyz = np.empty(rs)
    ystar = np.zeros(nmode)

    # Transform w_a --> (-w_0 - w_a)^(1/4).
    xstar[6] = pow(-xstar[5]-xstar[6], 0.25)

    # Check inputs.
    for i in range(p):
        if not data["xmin"][i] <= xstar[i] <= data["xmax"][i]:
            raise ValueError(f"Emulator out of bounds, param no. {i}.")

    if not data["z"][-1] <= xstar[-1] <= data["z"][0]:
        raise ValueError("Emulator redshift out of bounds.")

    # Standardize the inputs.
    xstarstd = (xstar[:-1] - data["xmin"]) / data["xrange"]

    # Compute the covariances between the new input and sims for all the PCs.
    for i in range(peta):
        for j in range(m):
            logc = 0.
            for k in range(p):
                logc -= data["beta"][i, k] * (data["x"][j, k] - xstarstd[k])**2

            Sigmastar[i, j] = np.exp(logc) / data["lamz"][i]

    # Compute wstar
    for i in range(peta):
        wstar[i] = 0
        for j in range(m):
            wstar[i] += Sigmastar[i, j] * KrigBasis[i, j]

    # Compute ystar, the new output
    for i in range(neta):
        ystaremu[i] = 0
        for j in range(peta):
            ystaremu[i] += data["K"][i, j] * wstar[j]

        ystaremu[i] = ystaremu[i] * data["sd"] + data["mean"][i]

    # Interpolate to the desired redshift.
    for i in range(nmode):
        for j in range(rs):
            ybyz[rs-j-1] = ystaremu[j*nmode+i]

        zinterp = Akima1DInterpolator(data["z_asc"], ybyz)
        ystar[i] = zinterp(xstar[p])

    # Convert to P(k).
    for i in range(nmode):
        ystar[i] = ystar[i] - 1.5*np.log10(data["mode"][i]) + np.log10(2) + 2*np.log10(np.pi)
        ystar[i] = 10**ystar[i]

    return ystar
