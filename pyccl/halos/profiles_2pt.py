from .profiles import HaloProfileHOD, HaloProfileCIBShang12


_doc = """

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    k : float or (..., nk, ...) array_like
        Comoving wavenumber in :math:`\\mathrm{Mpc}^{-1}`.
    M : float or (..., nM, ...) array_like
        Halo mass in :math:`\\mathrm{M}_{\\odot}`.
    a : float or (..., na, ...) array_like
        Scale factor.
    prof, prof2 : :class:`~pyccl.halos.profiles.HaloProfile`
        Halo profiles. If ``prof2 is None``, ``prof`` will be used
        (i.e. profile will be auto-correlated).
    mass_def : :obj:`~pyccl.halos.massdef.MassDef`
        The mass definition of ``M``.

    Returns
    -------
    F_var : float or (..., na, nk, nM, ...) ndarray
        Second-order Fourier-space moment between the input profiles.
    """


class Profile2pt:
    """Fourier-space 1-halo 2-point correlator between two halo profiles:

    .. math::

       (1 + \\rho_{u_1, u_2})
       \\langle u_1(k) \\rangle
       \\langle u_2(k) \\rangle

    In the simplest scenario, the second-order cumulant is the product
    of the individual Fourier-space profiles. More complicated cases may
    be implemented via the parameters of this class, or by subclassing.

    Parameters
    ----------
    r_corr : float
        Tuning knob for the 1-halo 2-point correlation.
        Scale the correlation by :math:`(1 + \\rho_{u_1, u_2})`.
        Useful when the individual 1-halo terms are not fully correlated.
        Examples can be found in :arXiv:1909.09102 and :arXiv:2102.07701.
        The default is 0, returning the product of the profiles.
    """
    def __init__(self, r_corr=0):
        self.r_corr = r_corr

    def update_parameters(self, r_corr=None):
        """Update the parameters of 1-halo 2-point correlator."""
        if r_corr is not None:
            self.r_corr = r_corr

    def fourier_2pt(self, cosmo, k, M, a, prof,
                    prof2=None, mass_def=None):
        """Compute the Fourier-space 2-point moment between two profiles."""
        if prof2 is None:
            prof2 = prof

        uk1 = prof.fourier(cosmo, k, M, a, mass_def=mass_def)
        uk2 = uk1 if prof2 == prof else prof2.fourier(cosmo, k, M, a,
                                                      mass_def=mass_def)
        return uk1 * uk2 * (1 + self.r_corr)

    fourier_2pt.__doc__ += _doc


class Profile2ptHOD(Profile2pt):
    """Fourier-space 1-halo 2-point correlator for the HOD profile:

    .. math::

       \\langle n_g^2(k)|M,a\\rangle = \\bar{N}_c(M,a)
       \\left[2f_c(a)\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a)+
       (\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a))^2\\right],

    where all quantities are described in the documentation of
    :class:`~pyccl.halos.profiles.HaloProfileHOD`.
    """
    def fourier_2pt(self, cosmo, k, M, a, prof,
                    prof2=None, mass_def=None):
        """Compute the Fourier-space two-point moment between two HOD profiles.
        """
        if not isinstance(prof, HaloProfileHOD):
            raise TypeError("prof must be `HaloProfileHOD`.")
        if prof2 is not None and prof2 != prof:
            raise ValueError("prof2 must be the same as prof.")
        return prof._fourier_variance(cosmo, k, M, a, mass_def)

    fourier_2pt.__doc__ += _doc


class Profile2ptCIB(Profile2pt):
    """Fourier-space 1-halo 2-point correlator between two CIB halo profiles
    (see Eq. 15 of McCarthy & Madhavacheril, ``2021PhRvD.103j3515M``).
    """
    def fourier_2pt(self, cosmo, k, M, a, prof,
                    prof2=None, mass_def=None):
        """Compute the Fourier-space 2-point moment between two CIB profiles.
        """
        if not isinstance(prof, HaloProfileCIBShang12):
            raise TypeError("prof must be of type `HaloProfileCIB`.")
        if prof2 is not None or not isinstance(prof2, HaloProfileCIBShang12):
                raise TypeError("prof must be of type `HaloProfileCIB`")

        nu2 = None if prof2 is None else prof2.nu
        return prof._fourier_variance(cosmo, k, M, a, mass_def, nu_other=nu2)

    fourier_2pt.__doc__ += _doc
