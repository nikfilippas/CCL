from .. import ccllib as lib
from ..background import species_types
from ..base import CCLAutoRepr, CCLNamedClass, warn_api, deprecate_attr
from ..base.parameters import physical_constants as const
from .mass_conversion import convert_concentration
import numpy as np


__all__ = ("dc_NakamuraSuto", "Dv_BryanNorman", "mass2radius_lagrangian",
           "MassDef", "MassDef200m", "MassDef200c", "MassDef500c",
           "MassDefVir", "MassDefFof",)


def dc_NakamuraSuto(cosmo, a):
    r"""Compute the peak threshold :math:`\delta_c(z)` assuming ΛCDM.

    Cosmology dependence of the critical linear density according to the
    spherical-collapse model. Fitting function from Nakamura & Suto (1997).

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s).

    Returns
    -------
    delta_c : float or (na,) ndarray
        Peak theshold at ``a``.
    """
    Om_mz = cosmo.omega_x(a, "matter")
    return const.DELTA_C * (1 + 0.012299 * np.log10(Om_mz))


def Dv_BryanNorman(cosmo, a):
    """Compute the virial collapse density contrast w.r.t. matter density,
    assuming ΛCDM.

    Cosmology dependence of the critical linear density according to the
    spherical-collapse model. Fitting function from Bryan & Norman (1997).
    :arXiv:`astro-ph/9710107`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.c
    a : float or (na,) array_like
        Scale factor(s).

    Returns
    -------
    Delta_vir : float or (na,) ndarray
        Virial collapse density contrast w.r.t. matter density.
    """
    Om_mz = cosmo.omega_x(a, "matter")
    x = Om_mz - 1
    Dv0 = 18 * np.pi * np.pi
    return (Dv0 + 82*x - 39*x**2) / Om_mz


def mass2radius_lagrangian(cosmo, M):
    """ Returns Lagrangian radius for a halo of mass M.
    The lagrangian radius is defined as that enclosing
    the mass of the halo assuming a homogeneous Universe.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        M (float or array_like): halo mass in units of M_sun.

    Returns:
        float or array_like: lagrangian radius in comoving Mpc.
    """
    M = np.asarray(M)
    return (M / (4.18879020479 * cosmo.rho_x(1, 'matter')))**(1./3.)



class MassDef(CCLAutoRepr, CCLNamedClass):
    """Halo mass definition. Halo masses are defined in terms of an overdensity
    parameter :math:`\\Delta` and an associated density :math:`X` (either the
    matter density or the critical density):

    .. math::
        M = \\frac{4 \\pi}{3} \\Delta\\,\\rho_X\\, R^3

    where :math:`R` is the halo radius. This object also holds methods to
    translate between :math:`R` and :math:`M`, and to translate masses between
    different definitions if a concentration-mass relation is provided.

    Args:
        Delta (float): overdensity parameter. Pass 'vir' if using virial
            overdensity.
        rho_type (string): either 'critical' or 'matter'.
        concentration (function, optional): concentration-mass relation.
            Provided as a `Concentration` object, or a string corresponding
            to one of the supported concentration-mass relations.
            If `None`, no c(M) relation will be attached to this mass
            definition (and hence one can't translate into other definitions).
    """
    __repr_attrs__ = __eq_attrs__ = ("name",)
    __getattr__ = deprecate_attr(pairs=[('c_m_relation', 'concentration')]
                                 )(super.__getattribute__)

    @warn_api(pairs=[("c_m_relation", "concentration")])
    def __init__(self, Delta, rho_type=None, *, concentration=None):
        # Check it makes sense
        if isinstance(Delta, str) and Delta not in ["fof", "vir"]:
            raise ValueError(f"Unknown Delta type {Delta}.")
        if isinstance(Delta, (int, float)) and Delta < 0:
            raise ValueError("Delta must be a positive number.")
        if rho_type not in ['matter', 'critical']:
            raise ValueError("rho_type must be either ['matter'|'critical].'")

        self.Delta = Delta
        self.rho_type = rho_type
        self.species = species_types[rho_type]
        # c(M) relation
        if concentration is None:
            self.concentration = None
        else:
            from .concentration import Concentration
            self.concentration = Concentration.create_instance(
                concentration, mass_def=self)

    @property
    def name(self):
        """Give a name to this mass definition."""
        if isinstance(self.Delta, (int, float)):
            return f"{self.Delta}{self.rho_type[0]}"
        return f"{self.Delta}"

    def get_Delta(self, cosmo, a):
        """ Gets overdensity parameter associated to this mass
        definition.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            a (float): scale factor

        Returns:
            float : value of the overdensity parameter.
        """
        if self.Delta == 'fof':
            raise ValueError("FoF masses don't have an associated overdensity."
                             "Nor can they be translated into other masses")
        if self.Delta == 'vir':
            status = 0
            D, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
            return D
        return self.Delta

    def _get_Delta_m(self, cosmo, a):
        """ For SO-based mass definitions, this returns the corresponding
        value of Delta for a rho_matter-based definition.
        """
        delta = self.get_Delta(cosmo, a)
        if self.rho_type == 'matter':
            return delta
        om_this = cosmo.omega_x(a, self.rho_type)
        om_matt = cosmo.omega_x(a, 'matter')
        return delta * om_this / om_matt

    def get_mass(self, cosmo, R, a):
        """ Translates a halo radius into a mass

        .. math::
            M = \\frac{4 \\pi}{3} \\Delta\\,\\rho_X\\, R^3

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            R (float or array_like): halo radius in units of Mpc (physical, not
                comoving).
            a (float): scale factor.

        Returns:
            float or array_like: halo mass in units of M_sun.
        """
        R_use = np.atleast_1d(R)
        Delta = self.get_Delta(cosmo, a)
        M = 4.18879020479 * cosmo.rho_x(a, self.rho_type) * Delta * R_use**3
        if np.ndim(R) == 0:
            return M[0]
        return M

    def get_radius(self, cosmo, M, a):
        """ Translates a halo mass into a radius

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.

        Returns:
            float or array_like: halo radius in units of Mpc (physical, not
                comoving).
        """
        M_use = np.atleast_1d(M)
        Delta = self.get_Delta(cosmo, a)
        R = (M_use / (4.18879020479 * Delta *
                      cosmo.rho_x(a, self.rho_type)))**(1./3.)
        if np.ndim(M) == 0:
            return R[0]
        return R

    @warn_api(pairs=[("mdef_other", "mass_def_other")])
    def translate_mass(self, cosmo, M, a, *, mass_def_other):
        """ Translate halo mass in this definition into another definition

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def_other (:obj:`MassDef`): another mass definition.

        Returns:
            float or array_like: halo masses in new definition.
        """
        if self == mass_def_other:
            return M
        if self.concentration is None:
            raise AttributeError("mass_def has no associated concentration.")
        om_this = cosmo.omega_x(a, self.rho_type)
        D_this = self.get_Delta(cosmo, a) * om_this
        c_this = self.concentration(cosmo, M, a)
        R_this = self.get_radius(cosmo, M, a)
        om_new = cosmo.omega_x(a, mass_def_other.rho_type)
        D_new = mass_def_other.get_Delta(cosmo, a) * om_new
        c_new = convert_concentration(c_old=c_this,
                                      Delta_old=D_this,
                                      Delta_new=D_new)
        R_new = c_new * R_this / c_this
        return mass_def_other.get_mass(cosmo, R_new, a)

    @classmethod
    def from_name(cls, name):
        """ Return mass definition subclass from name string.

        Args:
            name (string):
                a mass definition name (e.g. '200m' for Delta=200 matter)

        Returns:
            MassDef subclass corresponding to the input name.
        """
        MassDefName = f"MassDef{name.capitalize()}"
        if MassDefName in globals():
            # MassDef is defined in one of the implementations below.
            return globals()[MassDefName]
        parser = {"c": "critical", "m": "matter"}
        if len(name) < 2 or name[-1] not in parser:
            # Bogus input - can't parse it.
            raise ValueError("Could not parse mass definition string.")
        Delta, rho_type = name[:-1], parser[name[-1]]
        return lambda cm=None: cls(Delta, rho_type, concentration=cm)  # noqa


@warn_api(pairs=[('c_m', 'concentration')])
def MassDef200m(concentration='Duffy08'):
    r""":math:`\Delta = 200m` mass definition.

    Args:
        concentration (string): concentration-mass relation.
    """
    return MassDef(200, 'matter', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDef200c(concentration='Duffy08'):
    r""":math:`\Delta = 200c` mass definition.

    Args:
        concentration (string): concentration-mass relation.
    """
    return MassDef(200, 'critical', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDef500c(concentration='Ishiyama21'):
    r""":math:`\Delta = 500m` mass definition.

    Args:
        c_m (string): concentration-mass relation.
    """
    return MassDef(500, 'critical', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDefVir(concentration='Klypin11'):
    r""":math:`\Delta = \rm vir` mass definition.

    Args:
        concentration (string): concentration-mass relation.
    """
    return MassDef('vir', 'critical', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDefFof(concentration=None):
    r""":math:`\Delta = \rm FoF` mass definition.

    Args:
        concentration (string): concentration-mass relation.
    """
    return MassDef('fof', 'matter', concentration=concentration)
