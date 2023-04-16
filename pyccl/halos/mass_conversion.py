from ..base.deprecations import warn_api, CCLDeprecationWarning
from ..base.parameters import gsl_params
import numpy as np
from scipy.optimize import newton
from scipy.interpolate import RectBivariateSpline as Interp
import warnings


__all__ = ("convert_concentration",)


@warn_api
def convert_concentration(cosmo=None, *, c_old, Delta_old, Delta_new):
    r"""Compute the concentration parameter for a different mass definition.
    This is done assuming an NFW profile. The output concentration ``c_new``
    is found by solving the equation:

    .. math::

        f(c_{\rm old}) \Delta_{\rm old} = f(c_{\rm new}) \Delta_{\rm new}

    where

    .. math::

        f(x) = \frac{x^3}{\log(1+x) - x/(1+x)}.

    .. note::

        Vectorization for this function is specialized. It assumes that
        ``c_old`` has shape (na, nM) and ``Delta_old``, ``Delta_new`` both have
        shape (na,), so that the Delta-conversion only depends on scale factor.
        Every mass is only translated for its scale factor. The output has the
        same shape as the input, and all concentrations are translated with
        :math:`\frac{\Delta_{\rm old}}{\Delta_{\rm new}}`.

    Arguments
    ---------
    c_old : float or (na, nM,) array_like
        Concentration values to translate.
    Delta_old, Delta_new : float or (na,) array_like
        Overdensity (:math:`\Delta`) parameters associated with the
        halo mass definition of the old and new concentrations, respectively.
        See :class:`~pyccl.halos.massdef.MassDef` for details.

    Returns
    -------
    c_new : float or ``np.shape(c_old)`` ndarray
        Translated concentration to the new mass definition.
    """
    if cosmo is not None:
        warnings.warn("cosmo is deprecated as an argument in "
                      "convert_concentration", CCLDeprecationWarning)
    c_old = np.asarray(c_old, dtype=float)
    c_use = c_old.ravel()  # `ravel` returns a view - 10x faster than `flatten`

    # Delta_fac repeats itself every nM elements in the flattened c_use array.
    Delta_fac = np.asarray(Delta_old / Delta_new)
    Delta_use = np.repeat(Delta_fac, c_use.size/Delta_fac.size)

    # If Delta_new == Delta_old, then c_new == c_old.
    idx = Delta_use != 1
    c_new = c_use.copy()

    _compute_concentration_conversion()
    # Evaluate the spline at zip(c_use, Delta_use).
    c_new[idx] = np.exp(_cM_spline(c_use[idx], Delta_use[idx], grid=False))
    return c_new.reshape(c_old.shape)


# This is where the c(M) Delta-conversion spline is stored.
_cM_spline = None


def _compute_concentration_conversion():
    """Compute the c(M) Delta-conversion spline."""
    global _cM_spline
    if _cM_spline is not None:
        return

    c_old = np.geomspace(0.01, 30, 128)
    Delta_fac = np.geomspace(0.01, 20, 64)

    # Initial guess: Delta_old / Delta_new. || Second guess: 10x larger.
    x0 = np.ones_like(c_old)
    x1 = 10 * x0
    nfw = lambda x: x**3 / (np.log(x+1) - x / (x+1))                     # noqa
    func = lambda c_new, Delta_fac: nfw(c_new) - nfw(c_old) * Delta_fac  # noqa

    res = np.array([
        newton(func, x0=x0, x1=x1, args=(Delta,),
               maxiter=gsl_params.ROOT_N_ITERATION,
               rtol=gsl_params.ROOT_EPSREL)
        for Delta in Delta_fac])

    _cM_spline = Interp(c_old, Delta_fac, np.log(res).T)
