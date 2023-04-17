from .base.parameters import physical_constants as const
import numpy as np


__all__ = ("mu_MG", "Sigma_MG")


def _musigma(cosmo, k, a, *, c_mg=None):
    r"""Helper for the :math:`\mu(k, a)` and :math:`\Sigma(k, a)`
    parametrizations of modified gravity."""
    # This function can be extended to include other
    # redshift and scale z-dependencies for μ and Σ in the future.
    k_use, a_use = map(np.atleast_1d, [k, a])
    k_use, a_use = k_use[None, :], a_use[:, None]
    shp = np.broadcast_shapes(a.shape, k.shape)
    out = np.full(shp, c_mg)

    idx = k_use != 0
    H0 = 100*cosmo["h"]
    hoh0 = cosmo.h_over_h0(a)
    s2_k = 1000 * cosmo["lambda_mg"] * hoh0 * H0 / (k_use[idx] * const.CLIGHT)
    s1_k = (1 + c_mg * s2_k**2) / (1 + s2_k**2)
    vals = (s1_k * cosmo["mu_0"]
            * cosmo.omega_x(a, "dark_energy")
            / cosmo["Omega_l"])
    np.place(out, idx, vals)

    if np.isscalar(k):
        return out[0]
    return out


def mu_MG(cosmo, k, a):
    r"""Compute :math:`\mu(k, a)`, where :math:`\mu` is one of the
    parametrizing functions of modifications to GR in the quasistatic
    approximation.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    k : float or (nk,) array_like
        Wavenumber(s) in units of :math:`\rm Mpc^{-1}`.
    a : float
        Scale factor, normalized to 1 today.

    Returns
    -------
    μ_MG : float or (nk,) ``numpy.ndarray``
        Modification to Poisson equation under modified gravity.
    """
    return _musigma(cosmo, k, a, c_mg=cosmo["c1_mg"])


def Sigma_MG(cosmo, k, a):
    r"""Compute :math:`\Sigma(a, k)`, where :math:`\Sigma` is one of the
    parametrizing functions of modifications to GR in the quasistatic
    approximation.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    k : float or (nk,) array_like
        Wavenumber(s) in units of :math:`\rm Mpc^{-1}`.
    a : float
        Scale factor, normalized to 1 today.

    Returns
    -------
    Σ_MG : float or (nk,) ``numpy.ndarray``
        Modification to Poisson equation under modified gravity.
    """
    return _musigma(cosmo, k, a, c_mg=cosmo["c2_mg"])
