from . import ccllib as lib
from .errors import warnings, CCLDeprecationWarning


class CCLParameters:
    """Base for classes holding global CCL parameters and their values.

    Subclasses contain a reference to the C-struct with the collection
    of parameters and their values (via SWIG). All subclasses act as proxies
    to the CCL parameters at the C-level.

    Subclasses automatically store a backup of the initial parameter state
    to enable ad hoc reloading.
    """

    def __init_subclass__(cls, instance=None, freeze=False):
        """Routine for subclass initialization.

        Parameters:
            instance (``pyccl.ccllib.cvar``):
                Reference to ``cvar`` where the parameters are implemented.
            freeze (``bool``):
                Disable parameter mutation.
        """
        super().__init_subclass__()
        cls._instance = instance
        cls._frozen = freeze

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

        cls._instance.__class__.__setattr__ = _new_setattr

    def __init__(self):
        # Keep a copy of the default parameters.
        object.__setattr__(self, "_bak", CCLParameters.get_params_dict(self))

    def __getattribute__(self, name):
        get = object.__getattribute__
        try:
            return get(get(self, "_instance"), name)
        except AttributeError:
            return get(self, name)

    def __setattr__(self, key, value):
        if self._frozen and key != "T_CMB":
            # `T_CMB` mutates in Cosmology.
            name = self.__class__.__name__
            raise AttributeError(f"Instances of {name} are frozen.")
        if not hasattr(self._instance, key):
            raise KeyError(f"Parameter {key} does not exist.")
        if (key, value) == ("A_SPLINE_MAX", 1.0):
            # Setting `A_SPLINE_MAX` to its default value; do nothing.
            return
        object.__setattr__(self._instance, key, value)

    __getitem__ = __getattribute__

    __setitem__ = __setattr__

    def __repr__(self):
        out = self._bak.copy()
        for par in out:
            out[par] = getattr(self, par)
        return repr(out)

    def reload(self):
        """Reload the C-level default CCL parameters."""
        frozen = self._frozen
        if frozen:
            object.__setattr__(self, "_frozen", False)
        for param, value in self._bak.items():
            setattr(self, param, value)
        object.__setattr__(self, "_frozen", frozen)

    @classmethod
    def get_params_dict(cls, name):
        """Get a dictionary of the current parameters.

        Arguments:
            name (str or :obj:`CCLParameters`):
                Name or instance of the parameters to look up.
        """
        pars = eval(name) if isinstance(name, str) else name
        out = {}
        for par in dir(pars):
            if not par.startswith("_") and par not in ["this", "thisown"]:
                out[par] = getattr(pars, par)
        return out


class SplineParams(CCLParameters, instance=lib.cvar.user_spline_params):
    """Instances of this class hold the spline parameters."""


class GSLParams(CCLParameters, instance=lib.cvar.user_gsl_params):
    """Instances of this class hold the gsl parameters."""


class PhysicalConstants(CCLParameters, instance=lib.cvar.constants,
                        freeze=True):
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
    # Effective number of massless neutrinos. [2008.01074, 2012.02726]
    NEFF = 3.044

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

spline_params = SplineParams()
gsl_params = GSLParams()
physical_constants = PhysicalConstants()
