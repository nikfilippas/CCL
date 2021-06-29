import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
COSMO2 = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.73,
                       sigma8=0.8, n_s=0.95)
HBFS = [ccl.halos.HaloBiasSheth99,
        ccl.halos.HaloBiasSheth01,
        ccl.halos.HaloBiasTinker10,
        ccl.halos.HaloBiasBhattacharya11]
MS = [1E13, [1E12, 1E15], np.array([1E12, 1E15])]
MFOF = ccl.halos.MassDef('fof', 'matter')
MVIR = ccl.halos.MassDef('vir', 'critical')
MDFS = [MVIR, MVIR, MFOF, MVIR]


@pytest.mark.parametrize('bM_class', HBFS)
def test_bM_subclasses_smoke(bM_class):
    bM = bM_class(COSMO)
    for m in MS:
        b = bM.get_halo_bias(COSMO, m, 0.9)
        assert np.all(np.isfinite(b))
        assert np.shape(b) == np.shape(m)


def test_bM_cosmo():
    bMclass = ccl.halos.HaloBiasTinker10

    # check that the last-used Cosmology is stored
    bM = bMclass()
    bM.get_halo_bias(COSMO2, 1e14, 1)
    assert bM.cosmo == COSMO2

    # now, force the halo bias be cosmology-dependent
    def func(bMclass):
        return True
    bMclass._check_cosmo_dependent = func

    # check that it needs Cosmology as an argument
    with pytest.raises(ValueError):
        bMclass(cosmo=None)

    # define an almost-equivalent cosmology
    eps = 1e-5
    cosmo_new = ccl.Cosmology(Omega_c=COSMO2["Omega_c"]*(1+eps),
                              Omega_b=COSMO2["Omega_b"],
                              h=COSMO2["h"],
                              sigma8=COSMO2["sigma8"],
                              n_s=COSMO2["n_s"])
    assert not COSMO2.__eq__(cosmo_new)

    # check that the halo bias functions yield almost-equivalent results
    bM = bMclass(COSMO2)
    hb0 = bM.get_halo_bias(COSMO2, 1e14, 1)
    hb1 = bM.get_halo_bias(cosmo_new, 1e14, 1)
    assert np.allclose(hb0, hb1, rtol=2*eps)

    # revert the change and check
    bMclass._check_cosmo_dependent = ccl.halos.HaloBias._check_cosmo_dependent
    assert not bMclass._check_cosmo_dependent(bMclass)


@pytest.mark.parametrize('bM_pair', zip(HBFS, MDFS))
def test_bM_mdef_raises(bM_pair):
    bM_class, mdef = bM_pair
    with pytest.raises(ValueError):
        bM_class(COSMO, mdef)


def test_bM_SO_allgood():
    bM = ccl.halos.HaloBiasTinker10(COSMO, MVIR)
    for m in MS:
        b = bM.get_halo_bias(COSMO, m, 0.9)
        assert np.all(np.isfinite(b))
        assert np.shape(b) == np.shape(m)


@pytest.mark.parametrize('name', ['Tinker10', 'Sheth99'])
def test_bM_from_string(name):
    bM_class = ccl.halos.halo_bias_from_name(name)
    bM = bM_class(COSMO)
    for m in MS:
        b = bM.get_halo_bias(COSMO, m, 0.9)
        assert np.all(np.isfinite(b))
        assert np.shape(b) == np.shape(m)


def test_bM_from_string_raises():
    with pytest.raises(ValueError):
        ccl.halos.halo_bias_from_name('Tinker11')


def test_bM_default():
    bM = ccl.halos.HaloBias(COSMO)
    with pytest.raises(NotImplementedError):
        bM._get_bsigma(COSMO, 1., 1.)

    M_in = 1E12
    lM_out = bM._get_consistent_mass(COSMO,
                                     M_in, 1., bM.mdef)
    assert np.fabs(np.log10(M_in) - lM_out) < 1E-10
