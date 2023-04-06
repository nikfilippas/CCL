import warnings
from . import ccllib as lib
from .errors import CCLDeprecationWarning


__all__ = ("CCLParameters", "SplineParams", "spline_params", "GSLParams",
           "gsl_params", "PhysicalConstants", "physical_constants",
           "CosmologyParams",)


class Parameters:
    """Base for classes holding global CCL parameters and their values.

    Subclasses are coupled to the C-struct with the collection of parameters
    they represent (and their values), via SWIG.
    """
    _allowed_keys = ("_object_lock", "_instance", "_index",)

    def __init_subclass__(cls, *, instance=None, factory=None, freeze=False):
        """Routine for subclass initialization.

        Parameters
        ----------
        instance : :obj:`pyccl.ccllib`
            Reference to the instance where the parameters are implemented.
        factory : :class:`pyccl.ccllib`
            The SWIG factory class where the parameters are stored.
        freeze : bool
            Disable parameter mutation.
        """
        super().__init_subclass__()
        if not (bool(instance) ^ bool(factory)):  # XNOR
            raise ValueError(
                "Provide either the instance, or an instance factory.")
        cls._instance = instance
        cls._factory = factory
        cls._freeze = freeze

        def _new_setattr(self, key, value):
            # Make instances of the SWIG-level class immutable
            # so that everything is handled through this interface.
            # SWIG only assigns `this` via the low level `_ccllib`;
            # we therefore disable all other direct assignments.
            if key == "this":
                return object.__setattr__(self, key, value)
            name = self.__class__.__name__
            # TODO: Deprecation cycle for fully immutable Cosmology objects.
            # raise AttributeError(f"Direct assignment in {name} not supported.")  # noqa
            warnings.warn(
                f"Direct assignment of {name} is deprecated "
                "and an error will be raised in the next CCL release. "
                f"Set via `pyccl.{name}.{key}` before instantiation.",
                CCLDeprecationWarning)
            object.__setattr__(self, key, value)

        # Replace C-level `__setattr__`.
        class_ = cls._factory if cls._factory else cls._instance.__class__
        class_.__setattr__ = _new_setattr

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._object_lock = ObjectLock()
        if cls._factory:
            obj._instance = cls._factory()
        if cls._freeze:
            obj._object_lock.lock()
        return obj

    def __init__(self):
        self.reload()

    @cached_property
    def _parameters(self):
        return [par for par in dir(self) if self._is_parameter(par)]

    def _is_parameter(self, par) -> bool:
        # Check if `par` is a parameter: exclude private and callables.
        return not (par.startswith("_") or callable(getattr(self, par)))

    def __setattr__(self, key, value):
        if key not in self._allowed_keys:
            if self._object_lock.locked:
                name = self.__class__.__name__
                raise AttributeError(f"Instances of {name} are frozen.")
            if key not in self:
                raise AttributeError(f"Parameter {key} does not exist.")

        object.__setattr__(self, key, value)  # update Python
        object.__setattr__(self._instance, key, value)  # update C

    def __getitem__(self, key):
        return getattr(self, key)

    __setitem__ = __setattr__

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        for par in self._parameters:
            if getattr(self, par) != getattr(other, par):
                return False
        return True

    def __repr__(self):
        name = self.__class__.__name__
        pars = {par: self[par] for par in self}
        return f"<{name}>\n\t{repr(pars)}"

    def __contains__(self, key):
        return key in self._parameters

    def __len__(self):
        return len(self._parameters)

    @unlock_instance
    def __iter__(self):
        self._index = 0
        return self

    @unlock_instance
    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        self._index += 1
        return self._parameters[self._index-1]

    def freeze(self):
        self._object_lock.lock()

    def unfreeze(self):
        self._object_lock.unlock()

    def copy(self):
        out = type(self)()
        out.reload(source=self)
        return out

    @unlock_instance
    def reload(self, source=None):
        source = self.__class__ if source is None else source
        for par in self:
            setattr(self, par, getattr(source, par))


class SplineParams(CCLParameters, instance=lib.cvar.user_spline_params):
    """Instances of this class hold the spline parameters."""

    def __setattr__(self, key, value):
        if (key, value) == ("A_SPLINE_MAX", 1.0):
            return  # Setting `A_SPLINE_MAX` to its default value; do nothing.
        super().__setattr__(key, value)


class GSLParams(CCLParameters, instance=lib.cvar.user_gsl_params):
    """Instances of this class hold the gsl parameters."""


class PhysicalConstants:
    """Instances of this class hold the physical constants."""
    PI = 3.1415926535897932

    # ~~ ASTRONOMICAL CONSTANTS ~~ #
    # Astronomical Unit, unit conversion (m/au). [exact]
    AU = 149_597_870_800
    # Mean solar day (s/day). [exact]
    DAY = 86400.
    # Sidereal year (days/yr). [IERS2014 in J2000.0]
    YEAR = 365.256_363_004 * DAY

    # ~~ FUNDAMENTAL PHYSICAL CONSTANTS ~~ #
    # Speed of light (m/s). [exact]
    CLIGHT = 299_792_458.
    # Unit conversion (J/eV). [exact]
    EV_IN_J = 1.602_176_634e-19
    ELECTRON_CHARGE = EV_IN_J
    # Electron mass (kg). [CODATA2018]
    ELECTRON_MASS = 9.109_383_7015e-31
    # Planck's constant (J s). [exact]
    HPLANCK = 6.626_070_15e-34
    # Boltzmann's constant (J/K). [exact]
    KBOLTZ = 1.380_649e-23
    # Universal gravitational constant (m^3/kg/s^2). [CODATA2018]
    GNEWT = 6.674_30e-11
    # Vacuum permeability (N / A^2). [CODATA2018]
    MU_0 = 1.256_637_062_12e-6

    # ~~ DERIVED CONSTANTS ~~ #
    # Reduced Planck's constant (J s).
    HBAR = HPLANCK / 2 / PI
    # Vacuum permittivity (F / m).
    EPSILON_0 = 1 / (MU_0 * CLIGHT**2)
    # Fine structure constant (~ 1/137).
    ALPHA = ELECTRON_CHARGE**2 / (4*PI*EPSILON_0*HBAR*CLIGHT)
    # Thomson scattering cross-section for the electron (m^2).
    SIG_THOMSON = (8*PI/3) * (ALPHA*HBAR*CLIGHT / (ELECTRON_MASS*CLIGHT**2))**2
    # Speed of light (Mpc/h).
    CLIGHT_HMPC = CLIGHT / 1e5
    # Unit conversion (m/pc).
    PC_TO_METER = 180*60*60/PI * AU
    # Unit conversion (m/Mpc).
    MPC_TO_METER = 1e6 * PC_TO_METER
    # Stefan-Boltzmann's constant (kg m^2 / s).
    STBOLTZ = (PI**2/60) * KBOLTZ**4 / HBAR**3 / CLIGHT**2
    # Solar mass in (kg).
    SOLAR_MASS = 4 * PI*PI * AU**3 / GNEWT / YEAR**2
    # Critical density (100 M_sun/h / (Mpc/h)^3).
    RHO_CRITICAL = 3*1e4/(8*PI*GNEWT) * 1e6 * MPC_TO_METER / SOLAR_MASS
    # Neutrino constant required in Omeganuh2.
    NU_CONST = (8*PI**5 * (KBOLTZ/HPLANCK)**3 * (KBOLTZ/15/CLIGHT**3)
                * (8*PI*GNEWT/3) * (MPC_TO_METER**2/CLIGHT**2/1e10))
    # Linear density contrast of spherical collapse.
    DELTA_C = (3/20) * (12*PI)**(2/3)

    # ~~ OTHER CONSTANTS ~~ #
    # Neutrino mass splitting differences.
    # Lesgourgues & Pastor (2012)
    # Adv. High Energy Phys. 2012 (2012) 608515
    # arXiv:1212.6154, p.13
    DELTAM12_sq = 7.62e-5
    DELTAM13_sq_pos = 2.55e-3
    DELTAM13_sq_neg = -2.43e-3

    _instance = lib.cvar.constants

    def __init__(self):
        self._factory = self._instance.__class__


    @property
    def parameters(self):
        return


    def _is_parameter(self, par):
        return not par.startswith(("_", "this"))



class CosmologyParams(CCLParameters, factory=lib.parameters):
    """Instances of this class hold cosmological parameters."""

    def __getattribute__(self, key):
        if key == "m_nu":
            N_nu_mass = self._instance.N_nu_mass
            nu_masses = lib.parameters_get_nu_masses(self._instance, N_nu_mass)
            return nu_masses.tolist()
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key == "m_nu":
            return lib.parameters_m_nu_set_custom(self._instance, value)
        if key == "mgrowth":
            return lib.parameters_mgrowth_set_custom(self._instance, *value)
        super().__setattr__(key, value)


spline_params = SplineParams()
gsl_params = GSLParams()
physical_constants = PhysicalConstants()
