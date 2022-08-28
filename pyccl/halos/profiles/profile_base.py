from ...parameters import _FFTLogParams
from ...pyutils import get_broadcastable, resample_array, _fftlog_transform
import numpy as np
from abc import ABC, abstractproperty
from functools import partial


class HaloProfile(ABC):
    """Functionality associated to halo profiles.

    This abstract class contains methods to compute a halo profile in 3-D
    real and Fourier space, as well as projected (2-D) and the cumulative
    mean surface density.

    A minimal profile implementation should contain a method ``_real``
    or a ``_fourier`` method with signatures as in ``real`` and ``fourier``
    of this class, respectively. Fast Hankel transforms from real to fourier
    space and vice versa are performed internally with ``FFTLog``. Subclasses
    may contain analytic implementations of any of those methods to bypass
    the ``FFTLog`` calculation.
    """
    is_number_counts = False

    def __init__(self):
        self.precision_fftlog = _FFTLogParams()

    @abstractproperty
    def normprof(self) -> bool:
        """Normalize the profile in auto- and cross-correlations by
        :math:`I^0_1(k\\rightarrow 0, a|u)`
        (see :meth:`~pyccl.halos.halo_model.HMCalculator.I_0_1`).
        """

    def update_precision_fftlog(self, **kwargs):
        self.precision_fftlog.update_parameters(**kwargs)

    update_precision_fftlog.__doc__ = _FFTLogParams.update_parameters.__doc__

    _get_plaw_fourier = partial(_FFTLogParams._get_plaw, name="plaw_fourier")
    _get_plaw_projected = partial(_FFTLogParams._get_plaw, name="plaw_projected")  # noqa
    _get_plaw_fourier.__doc__ = _get_plaw_projected.__doc__ = _FFTLogParams._get_plaw.__doc__  # noqa

    def real(self, cosmo, r, M, a, mass_def):
        """3-D real-space profile.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a : float or array_like
            Scale factor.
        mass_def : :class:`~pyccl.halos.massdef.MassDef`
            The mass definition of ``M``.

        Returns
        -------
        P_r : float or array_like
            Real halo profile.
        """
        prof = self.__class__
        if "_real" in vars(prof):
            return self._real(cosmo, r, M, a, mass_def)
        if "_fourier" in vars(prof):
            return self._fftlog_wrap(cosmo, r, M, a, mass_def,
                                     fourier_out=False)
        raise NotImplementedError

    def fourier(self, cosmo, k, M, a, mass_def):
        """3-D Fourier-space profile.

        .. math::

           \\rho(k) = \\frac{1}{2 \\pi^2} \\int \\mathrm{d}r \\, r^2 \\,
           \\rho(r) \\, j_0(k \\, r)

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        k : float or array_like
            Comoving wavenumber in :math:`\\mathrm{Mpc}^{-1}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a : float or array_like
            Scale factor.
        mass_def : :class:`~pyccl.halos.massdef.MassDef`
            The mass definition of ``M``.

        Returns
        -------
        P_r : float or array_like
            Fourier halo profile.
        """
        # axes are [a, k, M]
        a, k, M = map(np.asarray, [a, k, M])
        a, k, M = get_broadcastable(a, k, M)
        prof = self.__class__
        if "_fourier" in vars(prof):
            return self._fourier(cosmo, k, M, a, mass_def)
        if "_real" in vars(prof):
            return self._fftlog_wrap(cosmo, k, M, a, mass_def,
                                     fourier_out=True)
        raise NotImplementedError

    def projected(self, cosmo, r_t, M, a, mass_def):
        """2-D projected profile.

        .. math::

           \\Sigma(R) = \\int \\mathrm{d}r_\\parallel \\,
           \\rho( \\sqrt{ r_\\parallel^2 + R^2} )

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a : float or array_like
            Scale factor.
        mass_def : :class:`~pyccl.halos.massdef.MassDef`
            The mass definition of ``M``.

        Returns
        -------
        P_proj : float or array_like
            Projected halo profile.
        """
        prof = self.__class__
        if "_projected" in vars(prof):
            return self._projected(cosmo, r_t, M, a, mass_def)
        return self._projected_fftlog_wrap(cosmo, r_t, M, a, mass_def,
                                           is_cumul2d=False)

    def cumul2d(self, cosmo, r_t, M, a, mass_def):
        """2-D cumulative surface density.

        .. math::

           \\Sigma(<R) = \\frac{2}{R^2} \\int \\mathrm{d}R' \\, R' \\,
           \\Sigma(R')

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a : float or array_like
            Scale factor.
        mass_def : :class:`~pyccl.halos.massdef.MassDef`
            The mass definition of ``M``.

        Returns
        -------
        P_cumul : float or array_like
            Cumulative halo profile.
        """
        prof = self.__class__
        if "_cumul2d" in vars(prof):
            return self._cumul2d(cosmo, r_t, M, a, mass_def)
        return self._projected_fftlog_wrap(cosmo, r_t, M, a, mass_def,
                                           is_cumul2d=True)

    def convergence(self, cosmo, r, M, a_lens, a_source, mass_def):
        """Profile onvergence.

        .. math::

           \\kappa(R) = \\frac{\\Sigma(R)}{\\Sigma_{\\mathrm{crit}}},

        where :math:`\\Sigma(R)` is the 2D projected surface mass density.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a_lens, a_source : float or array_like
            Scale factors of the lens and the source, respectively.
            If ``a_source`` is array_like, ``r.shape == a_source.shape``.
        mass_def : :class:`~pyccl.halos.massdef.MassDef`
            The mass definition of ``M``.

        Returns
        -------
        P_conv : float or array_like
            Convergence :math:`\\kappa` of the profile.
        """
        Sigma = self.projected(cosmo, r, M, a_lens, mass_def) / a_lens**2
        Sigma_crit = cosmo.sigma_critical(a_lens, a_source)
        return Sigma / Sigma_crit

    def shear(self, cosmo, r, M, a_lens, a_source, mass_def):
        """Tangential shear of a profile.

        .. math::

           \\gamma(R) = \\frac{\\Delta \\Sigma(R)}{\\Sigma_{\\mathrm{crit}}} =
           \\frac{\\overline{\\Sigma}(< R) -
           \\Sigma(R)}{\\Sigma_{\\mathrm{crit}}},

        where :math:`\\overline{\\Sigma}(< R)` is the average surface density
        within R.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a_lens, a_source : float or array_like
            Scale factors of the lens and the source, respectively.
            If ``a_source`` is array_like, ``r.shape == a_source.shape``.
        mass_def : :class:`~pyccl.halos.massdef.MassDef`
            The mass definition of ``M``.

        Returns
        -------
        P_shear : float or array_like
            Tangential shear :math:`\\gamma` of the profile.
        """
        Sigma = self.projected(cosmo, r, M, a_lens, mass_def)
        Sigma_bar = self.cumul2d(cosmo, r, M, a_lens, mass_def)
        Sigma_crit = cosmo.sigma_critical(a_lens, a_source)
        return (Sigma_bar - Sigma) / (Sigma_crit * a_lens**2)

    def reduced_shear(self, cosmo, r, M, a_lens, a_source, mass_def):
        """Reduced shear of a profile.

        .. math::

           g_t (R) = \\frac{\\gamma(R)}{(1 - \\kappa(R))},

        where :math:`\\gamma(R)` is the shear and :math:`\\kappa(R)` is the
        convergence.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a_lens, a_source : float or array_like
            Scale factors of the lens and the source, respectively.
            If ``a_source`` is array_like, ``r.shape == a_source.shape``.
        mass_def : :class:`~pyccl.halos.massdef.MassDef`
            The mass definition of ``M``.

        Returns
        -------
        P_red_shear : float or array_like
            Reduced shear :math:`g_t` of the profile.
        """
        convergence = self.convergence(cosmo, r, M, a_lens, a_source, mass_def)
        shear = self.shear(cosmo, r, M, a_lens, a_source, mass_def)
        return shear / (1.0 - convergence)

    def magnification(self, cosmo, r, M, a_lens, a_source, mass_def):
        """Magnification of a profile.

        .. math::

           \\mu (R) = \\frac{1}{\\left[(1 - \\kappa(R))^2 -
           \\vert \\gamma(R) \\vert^2 \\right]]},

        where :math:`\\gamma(R)` is the shear and :math:`\\kappa(R)` is the
        convergence.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        r : float or array_like
            Comoving radius in :math:`\\mathrm{Mpc}`.
        M : float or array_like
            Halo mass in :math:`\\mathrm{M}_{\\odot}`.
        a_lens, a_source : float or array_like
            Scale factors of the lens and the source, respectively.
            If ``a_source`` is array_like, ``r.shape == a_source.shape``.
        mass_def : :class:`~pyccl.halos.massdef.MassDef`
            The mass definition of ``M``.

        Returns
        -------
        P_magn : float or array_like
            Magnification :math:`\\mu` of the profile.
        """
        convergence = self.convergence(cosmo, r, M, a_lens, a_source, mass_def)
        shear = self.shear(cosmo, r, M, a_lens, a_source, mass_def)
        return 1.0 / ((1.0 - convergence)**2 - np.abs(shear)**2)

    def _fftlog_wrap(self, cosmo, k, M, a, mass_def,
                     fourier_out=False,
                     large_padding=True):
        # This computes the 3D Hankel transform
        #  \rho(k) = 4\pi \int dr r^2 \rho(r) j_0(k r)
        # if fourier_out == False, and
        #  \rho(r) = \frac{1}{2\pi^2} \int dk k^2 \rho(k) j_0(k r)
        # otherwise.

        # Select which profile should be the input
        if fourier_out:
            p_func = self._real
        else:
            p_func = self._fourier
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)
        lk_use = np.log(k_use)
        nM = len(M_use)

        # k/r ranges to be used with FFTLog and its sampling.
        if large_padding:
            k_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(k_use)
        else:
            k_min = self.precision_fftlog['padding_lo_extra'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_extra'] * np.amax(k_use)
        n_k = (int(np.log10(k_max / k_min)) *
               self.precision_fftlog['n_per_decade'])
        r_arr = np.geomspace(k_min, k_max, n_k)

        p_k_out = np.zeros([nM, k_use.size])
        # Compute real profile values
        p_real_M = p_func(cosmo, r_arr, M_use, a, mass_def)
        # Power-law index to pass to FFTLog.
        plaw_index = self._get_plaw_fourier(cosmo, a)

        # Compute Fourier profile through fftlog
        k_arr, p_fourier_M = _fftlog_transform(r_arr, p_real_M,
                                               3, 0, plaw_index)
        lk_arr = np.log(k_arr)

        for im, p_k_arr in enumerate(p_fourier_M):
            # Resample into input k values
            p_fourier = resample_array(lk_arr, p_k_arr, lk_use,
                                       self.precision_fftlog['extrapol'],
                                       self.precision_fftlog['extrapol'],
                                       0, 0)
            p_k_out[im, :] = p_fourier
        if fourier_out:
            p_k_out *= (2 * np.pi)**3

        if np.ndim(k) == 0:
            p_k_out = np.squeeze(p_k_out, axis=-1)
        if np.ndim(M) == 0:
            p_k_out = np.squeeze(p_k_out, axis=0)
        return p_k_out

    def _projected_fftlog_wrap(self, cosmo, r_t, M, a, mass_def,
                               is_cumul2d=False):
        # This computes Sigma(R) from the Fourier-space profile as:
        # Sigma(R) = \frac{1}{2\pi} \int dk k J_0(k R) \rho(k)
        r_t_use = np.atleast_1d(r_t)
        M_use = np.atleast_1d(M)
        lr_t_use = np.log(r_t_use)
        nM = len(M_use)

        # k/r range to be used with FFTLog and its sampling.
        r_t_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(r_t_use)
        r_t_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(r_t_use)
        n_r_t = (int(np.log10(r_t_max / r_t_min)) *
                 self.precision_fftlog['n_per_decade'])
        k_arr = np.geomspace(r_t_min, r_t_max, n_r_t)

        sig_r_t_out = np.zeros([nM, r_t_use.size])
        # Compute Fourier-space profile
        if getattr(self, '_fourier', None):
            # Compute from `_fourier` if available.
            p_fourier = self._fourier(cosmo, k_arr, M_use,
                                      a, mass_def)
        else:
            # Compute with FFTLog otherwise.
            lpad = self.precision_fftlog['large_padding_2D']
            p_fourier = self._fftlog_wrap(cosmo,
                                          k_arr,
                                          M_use, a,
                                          mass_def,
                                          fourier_out=True,
                                          large_padding=lpad)
        if is_cumul2d:
            # The cumulative profile involves a factor 1/(k R) in
            # the integrand.
            p_fourier *= 2 / k_arr[None, :]

        # Power-law index to pass to FFTLog.
        if is_cumul2d:
            i_bessel = 1
            plaw_index = self._get_plaw_projected(cosmo, a) - 1
        else:
            i_bessel = 0
            plaw_index = self._get_plaw_projected(cosmo, a)

        # Compute projected profile through fftlog
        r_t_arr, sig_r_t_M = _fftlog_transform(k_arr, p_fourier,
                                               2, i_bessel,
                                               plaw_index)
        lr_t_arr = np.log(r_t_arr)

        if is_cumul2d:
            sig_r_t_M /= r_t_arr[None, :]
        for im, sig_r_t_arr in enumerate(sig_r_t_M):
            # Resample into input r_t values
            sig_r_t = resample_array(lr_t_arr, sig_r_t_arr,
                                     lr_t_use,
                                     self.precision_fftlog['extrapol'],
                                     self.precision_fftlog['extrapol'],
                                     0, 0)
            sig_r_t_out[im, :] = sig_r_t

        if np.ndim(r_t) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=-1)
        if np.ndim(M) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=0)
        return sig_r_t_out


class HaloProfileNumberCounts(HaloProfile):
    """Abstract profile implementing a number counts quantity."""
    is_number_counts = True
    normprof = True
