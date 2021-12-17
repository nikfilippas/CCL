import warnings
import numpy as np

from ..core import CosmologyVanillaLCDM
from .halo_model import HMCalculator, halomod_power_spectrum
from .profiles import HaloProfileNFW


from ..pyutils import CCLWarning


class HalomodCorrection(object):
    """
    """
    name = "base"
    interpolators = {}

    def __init__(self):
        pass

    @classmethod
    def from_name(cls, name):
        """
        """
        hmcorrs = {hm.name for hm in cls.__subclasses__()}
        if name in hmcorrs:
            return hmcorrs[name]
        else:
            raise ValueError("Halo model correction method not recognised")

    def _correct_1h2h(self, pk1, pk2):
        raise NotImplementedError("Nothing in the base class")

    def correct_1h2h(self, pk1, pk2):
        """
        """
        pk_new = self._correct_1h2h(pk1, pk2)
        return pk_new


class HalomodCorrectionHMCode(HalomodCorrection):
    """
    """
    name = "HMCode"

    def __init__(self):
        super().__init__()

    def _correct_1h2h(self, pk1, pk2):
        pass


class HalomodCorrectionRatio(HalomodCorrection):
    """
    """
    name = "ratio"

    def __init__(self, cosmo=None, hmc=None):
        self.cosmo = cosmo
        self.hmc = hmc
        super().__init__()

        # setup
        if self.cosmo is None:
            warnings.warn("No input Cosmology. Defaulting to Vanilla LCDM.",
                          CCLWarning)
            self.cosmo = CosmologyVanillaLCDM()
        if self.hmc is None:
            hmc_config = {"mass_function": "Tinker08",
                          "halo_bias": "Tinker10",
                          "mass_def": "200c"}
            warnings.warn("No input HM Calculator. Defaulting to\n"
                          f"{hmc_config}", CCLWarning)
            self.hmc = HMCalculator(**hmc_config)

        # keyword lookup
        self.hash_ = hash(frozenset({**cosmo._params_init_kwargs,
                                **cosmo._config_init_kwargs,
                                **hmc_config}.items()))

        if self.hash_ not in self.interpolators:
            self._setup()

    def _setup(self, lk_min=-1., lk_max=1., nlk=64,
               a_min=0.4, a_max=1., na=32):
        from scipy.interpolate import RectBivariateSpline

        # setup
        lk_arr = np.linspace(lk_min, lk_max, nlk)
        k_arr = 10**lk_arr
        a_arr = np.linspace(a_min, a_max, na)
        prof = HaloProfileNFW(c_m_relation=self.hmc.mass_def.concentration)

        # calculate
        pk_hm = np.array([
            halomod_power_spectrum(self.cosmo, self.hmc,
                                   k_arr, a, prof, normprof=True)
            for a in a_arr])
        pk_hf = np.array([
            self.cosmo.nonlin_matter_power(k_arr, a) for a in a_arr])
        ratio = pk_hf/pk_hm - 1

        # interpolate
        func = RectBivariateSpline(a_arr, lk_arr, ratio)

        # store
        self.interpolators[self.hash_] = func

    def _eval(self, k_arr, a_arr):
        func = self.interpolators[self.hash_]
        return func(a_arr, np.log10(k_arr))

    def _correct_1h2h(self, pk_1h, pk2h):
        pass


class HalomodCorrectionGauss(HalomodCorrection):
    """
    """
    name = "gauss"

    def __init__(self):
        pass

    def _correct_1h2h(self, pk1, pk2):
        pass
