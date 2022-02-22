import pickle
import tempfile
import pytest
import numpy as np
import pyccl as ccl


def test_cosmo_methods():
    """ Check that all pyccl functions that take cosmo
    as their first argument are methods of the Cosmology object.
    """
    from inspect import getmembers, isfunction, signature
    from pyccl import background, baryons, boltzmann, \
        cells, correlations, covariances, neutrinos, \
        pk2d, power, tk3d, tracers, halos, nl_pt
    from pyccl.core import CosmologyVanillaLCDM
    cosmo = CosmologyVanillaLCDM()
    subs = [background, baryons, boltzmann, cells, correlations, covariances,
            neutrinos, pk2d, power, tk3d, tracers, halos, nl_pt]
    funcs = [getmembers(sub, isfunction) for sub in subs]
    funcs = [func for sub in funcs for func in sub]
    for name, func in funcs:
        pars = signature(func).parameters
        if list(pars)[0] == "cosmo":
            _ = getattr(cosmo, name)

    # quantitative
    assert ccl.sigma8(cosmo) == cosmo.sigma8()
    assert ccl.rho_x(cosmo, 1., "matter", is_comoving=False) == \
        cosmo.rho_x(1., "matter", is_comoving=False)
    assert ccl.get_camb_pk_lin(cosmo).eval(1., 1., cosmo) == \
        cosmo.get_camb_pk_lin().eval(1., 1., cosmo)
    prof = ccl.halos.HaloProfilePressureGNFW()
    hmd = ccl.halos.MassDef200m()
    hmf = ccl.halos.MassFuncTinker08()
    hbf = ccl.halos.HaloBiasTinker10()
    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def=hmd)
    P1 = ccl.halos.halomod_power_spectrum(cosmo, hmc, 1., 1., prof,
                                          normprof=False)
    P2 = cosmo.halomod_power_spectrum(hmc, 1., 1., prof, normprof=False)
    assert P1 == P2


def test_cosmology_critical_init():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=0,
        m_nu=0.0,
        w0=-1.0,
        wa=0.0,
        m_nu_type='normal',
        Omega_g=0,
        Omega_k=0)
    assert np.allclose(cosmo.cosmo.data.growth0, 1)


def test_cosmology_init():
    """
    Check that Cosmology objects can only be constructed in a valid way.
    """
    # Make sure error raised if invalid transfer/power spectrum etc. passed
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, matter_power_spectrum='x')
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, transfer_function='x')
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, baryons_power_spectrum='x')
    with pytest.warns(ccl.CCLDeprecationWarning):
        with pytest.raises(ValueError):
            ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                          n_s=0.96, mass_function='x')
    with pytest.warns(ccl.CCLDeprecationWarning):
        with pytest.raises(ValueError):
            ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                          n_s=0.96, halo_concentration='x')
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, emulator_neutrinos='x')
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, m_nu=np.array([0.1, 0.1, 0.1, 0.1]))
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, m_nu=ccl)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      m_nu=np.array([0.1, 0.1, 0.1]), m_nu_type='normal')


def test_cosmology_setitem():
    cosmo = ccl.CosmologyVanillaLCDM()
    with pytest.raises(NotImplementedError):
        cosmo['a'] = 3


def test_cosmology_output():
    """
    Check that status messages and other output from Cosmology() object works
    correctly.
    """
    # Create test cosmology object
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                          n_s=0.96)

    # Return and print status messages
    with pytest.warns(None) as w_rec:
        cosmo.status()
        print(cosmo)
    assert len(w_rec) == 0

    # Test status methods for different precomputable quantities
    assert cosmo.has_distances is False
    assert cosmo.has_growth is False
    assert cosmo.has_linear_power is False
    assert cosmo.has_nonlin_power is False
    assert cosmo.has_sigma is False

    # Check that quantities can be precomputed
    with pytest.warns(None) as w_rec:
        cosmo.compute_distances()
        cosmo.compute_growth()
        cosmo.compute_linear_power()
        cosmo.compute_nonlin_power()
        cosmo.compute_sigma()
    assert len(w_rec) == 0

    assert cosmo.has_distances is True
    assert cosmo.has_growth is True
    assert cosmo.has_linear_power is True
    assert cosmo.has_nonlin_power is True
    assert cosmo.has_sigma is True


def test_cosmology_equal_hash():
    """Check the Cosmology equivalence method."""
    # equivalent cosmologies
    cosmo1 = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                           sigma8=0.81, n_s=0.96)
    cosmo2 = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                           sigma8=0.81, n_s=0.96)
    assert cosmo1 == cosmo2
    assert hash(cosmo1) == hash(cosmo2)

    # different cosmological parameters
    cosmo2 = ccl.Cosmology(Omega_c=0.24, Omega_b=0.04, h=0.75,
                           sigma8=0.8, n_s=0.95)
    assert cosmo1 != cosmo2
    assert hash(cosmo1) != hash(cosmo2)

    # different power spectra
    a = np.linspace(0.5, 1., 16)
    k = np.logspace(-2, 1, 128)
    pk = np.ones((a.size, k.size))
    pk_dict_1 = {"a": a, "k": k, "delta_matter:delta_matter": pk}
    pk_dict_2 = {"a": a, "k": k, "delta_matter:delta_matter": 2*pk}

    # linear
    cosmo1 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
        pk_linear=pk_dict_1)
    cosmo2 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
        pk_linear=pk_dict_2)
    assert cosmo1 != cosmo2
    assert hash(cosmo1) != hash(cosmo2)

    # non-linear
    cosmo1 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
        pk_nonlin=pk_dict_1)
    cosmo2 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
        pk_nonlin=pk_dict_2)
    assert cosmo1 != cosmo2
    assert hash(cosmo1) != hash(cosmo2)

    # TODO: uncomment once this is implemented
    # different CCL global parameters
    # cosmo1 = ccl.CosmologyVanillaLCDM()
    # ccl.gsl_params.HM_MMIN = 1e6
    # cosmo2 = ccl.CosmologyVanillaLCDM()
    # assert cosmo1 != cosmo2
    # assert hash(cosmo1) != hash(cosmo2)


def test_cosmology_equal_hash():
    """Check the Cosmology equivalence method and hashing."""
    # equivalent cosmologies
    cosmo1 = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                           sigma8=0.81, n_s=0.96)
    cosmo2 = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                           sigma8=0.81, n_s=0.96)
    assert cosmo1 == cosmo2
    assert hash(cosmo1) == hash(cosmo2)

    # different cosmological parameters
    cosmo2 = ccl.Cosmology(Omega_c=0.24, Omega_b=0.04, h=0.75,
                           sigma8=0.8, n_s=0.95)
    assert cosmo1 != cosmo2
    assert hash(cosmo1) != hash(cosmo2)

    # different power spectra
    a = np.linspace(0.5, 1., 16)
    k = np.logspace(-2, 1, 128)
    pk = np.ones((a.size, k.size))
    pk_dict_1 = {"a": a, "k": k, "delta_matter:delta_matter": pk}
    pk_dict_2 = {"a": a, "k": k, "delta_matter:delta_matter": 2*pk}
    # linear
    cosmo1 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
        pk_linear=pk_dict_1)
    cosmo2 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
        pk_linear=pk_dict_2)
    assert cosmo1 != cosmo2
    assert hash(cosmo1) != hash(cosmo2)

    # non-linear
    cosmo1 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
        pk_nonlin=pk_dict_1)
    cosmo2 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
        pk_nonlin=pk_dict_2)
    assert cosmo1 != cosmo2
    assert hash(cosmo1) != hash(cosmo2)


def test_cosmology_pickles():
    """Check that a Cosmology object pickles."""
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo = ccl.Cosmology(
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            m_nu=[0.02, 0.1, 0.05], m_nu_type='list',
            z_mg=[0.0, 1.0], df_mg=[0.01, 0.0])

        with tempfile.TemporaryFile() as fp:
            pickle.dump(cosmo, fp)

            fp.seek(0)
            cosmo2 = pickle.load(fp)

    assert (ccl.comoving_radial_distance(cosmo, 0.5) ==
            ccl.comoving_radial_distance(cosmo2, 0.5))


def test_cosmology_lcdm():
    """Check that the default vanilla cosmology behaves
    as expected"""
    c1 = ccl.Cosmology(Omega_c=0.25,
                       Omega_b=0.05,
                       h=0.67, n_s=0.96,
                       sigma8=0.81)
    c2 = ccl.CosmologyVanillaLCDM()
    assert (ccl.comoving_radial_distance(c1, 0.5) ==
            ccl.comoving_radial_distance(c2, 0.5))


def test_cosmology_p18lcdm_raises():
    with pytest.raises(ValueError):
        kw = {'Omega_c': 0.1}
        ccl.CosmologyVanillaLCDM(**kw)


def test_cosmology_repr():
    """Check that we can make a Cosmology object from its repr."""
    import pyccl  # noqa: F401

    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo = ccl.Cosmology(
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            m_nu=[0.02, 0.1, 0.05], m_nu_type='list',
            z_mg=[0.0, 1.0], df_mg=[0.01, 0.0])

        cosmo2 = eval(str(cosmo))
        assert (ccl.comoving_radial_distance(cosmo, 0.5) ==
                ccl.comoving_radial_distance(cosmo2, 0.5))

        cosmo3 = eval(repr(cosmo))
        assert (ccl.comoving_radial_distance(cosmo, 0.5) ==
                ccl.comoving_radial_distance(cosmo3, 0.5))

    # same test with arrays to be sure
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo = ccl.Cosmology(
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            m_nu=np.array([0.02, 0.1, 0.05]), m_nu_type='list',
            z_mg=np.array([0.0, 1.0]), df_mg=np.array([0.01, 0.0]))

        cosmo2 = eval(str(cosmo))
        assert (ccl.comoving_radial_distance(cosmo, 0.5) ==
                ccl.comoving_radial_distance(cosmo2, 0.5))

        cosmo3 = eval(repr(cosmo))
        assert (ccl.comoving_radial_distance(cosmo, 0.5) ==
                ccl.comoving_radial_distance(cosmo3, 0.5))

    # adding extra parameters
    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        extra_parameters={"camb": {"halofit_version": "mead2020",
                                   "HMCode_logT_AGN": 7.8}})

    cosmo2 = eval(str(cosmo))
    assert (ccl.comoving_radial_distance(cosmo, 0.5) ==
            ccl.comoving_radial_distance(cosmo2, 0.5))

    cosmo3 = eval(repr(cosmo))
    assert (ccl.comoving_radial_distance(cosmo, 0.5) ==
            ccl.comoving_radial_distance(cosmo3, 0.5))

    # testing with vanilla cosmology
    cosmo = ccl.CosmologyVanillaLCDM()

    cosmo2 = eval(str(cosmo))
    assert (ccl.comoving_radial_distance(cosmo, 0.5) ==
            ccl.comoving_radial_distance(cosmo2, 0.5))

    cosmo3 = eval(repr(cosmo))
    assert (ccl.comoving_radial_distance(cosmo, 0.5) ==
            ccl.comoving_radial_distance(cosmo3, 0.5))


def test_cosmology_context():
    """Check that using a Cosmology object in a context manager
    frees C resources properly."""
    with pytest.warns(ccl.CCLDeprecationWarning):
        with ccl.Cosmology(
                Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                m_nu=np.array([0.02, 0.1, 0.05]),
                m_nu_type='list',
                z_mg=np.array([0.0, 1.0]),
                df_mg=np.array([0.01, 0.0])) as cosmo:
            # make sure it works
            assert not cosmo.has_distances
            ccl.comoving_radial_distance(cosmo, 0.5)
            assert cosmo.has_distances

    # make sure it does not!
    assert not hasattr(cosmo, "cosmo")
    assert not hasattr(cosmo, "_params")

    with pytest.raises(AttributeError):
        cosmo.has_growth


@pytest.mark.parametrize('c', ['bhattacharya2011', 'duffy2008',
                               'constant_concentration'])
def test_cosmology_concentrations(c):
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo = ccl.CosmologyVanillaLCDM(matter_power_spectrum="halo_model",
                                         halo_concentration=c)
    with pytest.warns(ccl.CCLWarning):
        cosmo.compute_nonlin_power()


@pytest.mark.parametrize('hmf', ['angulo', 'tinker', 'tinker10',
                                 'watson', 'shethtormen'])
def test_cosmology_mass_functions(hmf):
    """Check halo concentration and halo mass function passed into choice
    `halo_model` of matter power spectrum in Cosmology."""
    valid_hmf = ["tinker10", "shethtormen"]
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo = ccl.CosmologyVanillaLCDM(matter_power_spectrum="halo_model",
                                         mass_function=hmf)
    if hmf in valid_hmf:
        with pytest.warns(ccl.CCLWarning):
            cosmo.compute_nonlin_power()


def test_cosmology_halomodel_power():
    """This test checks default behavior for the halo model parameters in
    Cosmology.
    """
    k_arr = np.logspace(-1, 2, 64)
    a = 0.8

    def Cosmo(**pars):
        return ccl.CosmologyVanillaLCDM(
            transfer_function="bbks",
            matter_power_spectrum="halo_model",
            **pars)

    # halo model in extra parameters
    cosmo1 = Cosmo(extra_parameters={
        "halo_model": {"mass_def": "200m",
                       "mass_def_strict": False,
                       "mass_function": "Tinker10",
                       "halo_bias": "Tinker10",
                       "concentration": "Duffy08"}})
    cosmo1.compute_nonlin_power()

    # only some parameters defined
    cosmo2 = Cosmo(extra_parameters={
        "halo_model": {"mass_function": "Tinker10",
                       "concentration": "Duffy08"}})
    cosmo2.compute_nonlin_power()

    # no parameters
    cosmo3 = Cosmo()
    with pytest.warns(ccl.CCLWarning):
        cosmo3.compute_nonlin_power()

    # deprecated parameters
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo4 = Cosmo(mass_function="tinker10",
                       halo_concentration="duffy2008")
    with pytest.warns(ccl.CCLWarning):
        cosmo4.compute_nonlin_power()

    # some deprecated parameters
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo5 = Cosmo(mass_function="tinker10")
    with pytest.warns(ccl.CCLWarning):
        cosmo5.compute_nonlin_power()

    def F(cosmo):
        return cosmo.nonlin_matter_power(k_arr, a)

    assert all([np.allclose(F(cosmo1), F(x), rtol=0)
                for x in [cosmo2, cosmo3, cosmo4, cosmo5]])


def test_cosmology_halomodel_deprecated():
    """This test tries a different mass function and checks that the
    corresponding halo bias is used, if available.
    """
    k_arr = np.logspace(-1, 2, 64)
    a = 0.8

    def Cosmo(**pars):
        return ccl.CosmologyVanillaLCDM(
            transfer_function="bbks",
            matter_power_spectrum="halo_model",
            **pars)

    # old behavior
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo1 = Cosmo(mass_function="shethtormen",
                       halo_concentration="constant_concentration")
    with pytest.warns(ccl.CCLWarning):
        cosmo1.compute_nonlin_power()

    # new behavior
    cosmo2 = Cosmo(extra_parameters={
        "halo_model": {"mass_function": "Sheth99",
                       "concentration": "Constant"}})
    cosmo2.compute_nonlin_power()

    # fully specified new behavior
    cosmo3 = Cosmo(extra_parameters={
        "halo_model": {"mass_def": "200m",
                       "mass_def_strict": False,
                       "mass_function": "Sheth99",
                       "halo_bias": "Sheth99",
                       "concentration": "Constant"}})
    cosmo3.compute_nonlin_power()

    def F(cosmo):
        return cosmo.nonlin_matter_power(k_arr, a)

    assert all([np.allclose(F(cosmo1), F(x), rtol=0)
                for x in [cosmo2, cosmo3]])


def test_pyccl_default_params():
    """Check that Python-layer for setting the gsl and spline parameters
    works on par with the C-layer."""
    HM_MMIN = ccl.gsl_params["HM_MMIN"]

    # we will test with this parameter
    assert HM_MMIN == 1e7

    # can be accessed as an attribute and as a dictionary item
    assert ccl.gsl_params.HM_MMIN == ccl.gsl_params["HM_MMIN"]

    # can be assigned as an attribute
    ccl.gsl_params.HM_MMIN = 1e5
    assert ccl.gsl_params["HM_MMIN"] == 1e5  # cross-check

    ccl.gsl_params["HM_MMIN"] = 1e6
    assert ccl.gsl_params.HM_MMIN == 1e6

    # does not accept extra assignment
    with pytest.raises(KeyError):
        ccl.gsl_params.test = "hello_world"
    with pytest.raises(KeyError):
        ccl.gsl_params["test"] = "hallo_world"

    # verify that this has changed
    assert ccl.gsl_params.HM_MMIN != HM_MMIN

    # but now we reload it, so it should be the default again
    ccl.gsl_params.reload()
    assert ccl.gsl_params.HM_MMIN == HM_MMIN

    # complains when we try to set A_SPLINE_MAX != 1.0
    ccl.spline_params.A_SPLINE_MAX = 1.
    with pytest.raises(RuntimeError):
        ccl.spline_params.A_SPLINE_MAX = 0.9

    # complains when we try to change the spline type
    ccl.spline_params.A_SPLINE_TYPE = None
    with pytest.raises(RuntimeError):
        ccl.spline_params.A_SPLINE_TYPE = "something_else"

    # check that dict properties work fine
    dic = dict(zip(ccl.gsl_params.keys(), ccl.gsl_params.values()))
    assert dic.items() == ccl.gsl_params.items()

    # check that copying works fine
    ccl.gsl_params.reload()
    dic = ccl.gsl_params.copy()
    dic.HM_MMIN = 1e6
    assert dic.HM_MMIN != ccl.gsl_params.HM_MMIN


def test_cosmology_default_params():
    """Check that the default params within Cosmology work as intended."""
    cosmo1 = ccl.CosmologyVanillaLCDM()
    v1 = cosmo1.cosmo.gsl_params.HM_MMIN

    ccl.gsl_params.HM_MMIN = 1e6
    cosmo2 = ccl.CosmologyVanillaLCDM()
    v2 = cosmo2.cosmo.gsl_params.HM_MMIN
    assert v2 == 1e6
    assert v2 != v1

    ccl.gsl_params.reload()
    cosmo3 = ccl.CosmologyVanillaLCDM()
    v3 = cosmo3.cosmo.gsl_params.HM_MMIN
    assert v3 == v1


def test_ccl_physical_constants_smoke():
    assert ccl.physical_constants.CLIGHT == ccl.ccllib.cvar.constants.CLIGHT

    # constants are immutable
    with pytest.raises(NotImplementedError):
        ccl.physical_constants.CLIGHT = 3e8


def test_CCLParams_raises():
    with pytest.raises(ValueError):
        ccl.physical_constants.locked = False
