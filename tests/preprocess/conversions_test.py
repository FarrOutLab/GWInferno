import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import beta

from gwinferno.preprocess import conversions


def test_chieff_from_q_component_spins():
    test_qs = np.array([0.1, 0.5, 0.9, 1.0])
    test_mags = np.linspace(0, 1, 10)
    test_tilts = np.linspace(-1, 1, 10)

    # Check non-spinning case
    qq, tt = np.meshgrid(test_qs, test_tilts)
    chi_effs = conversions.chieff_from_q_component_spins(qq, 0.0, 0.0, tt, tt)
    assert_allclose(chi_effs, 0.0, err_msg="Non-spinning case failed")

    # Check stability
    qq, aa, tt = np.meshgrid(test_qs, test_mags, test_tilts)
    chi_effs = conversions.chieff_from_q_component_spins(qq, aa, aa, tt, tt)
    assert np.all(np.isfinite(chi_effs)), "Non-finite values found"

    # Point checks
    test_cases = [
        ([1.0, 1.0, 1.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 1.0, 1.0, 1.0], 1.0),
        ([1.0, 1.0, 1.0, -1.0, -1.0], -1.0),
        ([0.5, 0.8, 0.2, -0.4, 0.3], -0.1933333333),
        ([0.2, 0.3, 0.4, 0.5, 0.6], 0.165),
    ]
    for inputs, expected in test_cases:
        chi_eff = conversions.chieff_from_q_component_spins(*inputs)
        assert_allclose(chi_eff, expected, err_msg="Point check failed")


def test_chip_from_q_component_spins():
    test_qs = np.array([0.1, 0.5, 0.9, 1.0])
    test_mags = np.linspace(0, 1, 10)
    test_tilts = np.linspace(-1, 1, 10)

    # Check non-spinning case
    qq, tt = np.meshgrid(test_qs, test_tilts)
    chi_ps = conversions.chip_from_q_component_spins(qq, 0.0, 0.0, tt, tt)
    assert_allclose(chi_ps, 0.0, err_msg="Non-spinning case failed")

    # Check stability
    qq, aa, tt = np.meshgrid(test_qs, test_mags, test_tilts)
    chi_ps = conversions.chip_from_q_component_spins(qq, aa, aa, tt, tt)
    assert np.all(np.isfinite(chi_ps)), "Non-finite values found"


def test_mu_var_from_alpha_beta():
    test_alphas = np.linspace(1, 50, 100)
    test_betas = np.linspace(1, 50, 100)
    test_xmaxs = [1.0, 0.8, 0.5, 0.2]
    for test_xmax in test_xmaxs:
        αα, ββ = np.meshgrid(test_alphas, test_betas)

        expected_means = beta.mean(αα, ββ, scale=test_xmax)
        expected_vars = beta.var(αα, ββ, scale=test_xmax)
        means, vars = conversions.mu_var_from_alpha_beta(αα, ββ, xmax=test_xmax)

        assert_allclose(means, expected_means, err_msg="Means do not match")
        assert_allclose(vars, expected_vars, err_msg="Vars do not match")


def test_alpha_beta_from_mu_var():
    expected_alphas = np.linspace(1, 50, 100)
    expected_betas = np.linspace(1, 50, 100)
    test_xmaxs = [1.0, 0.8, 0.5, 0.2]
    for test_xmax in test_xmaxs:
        αα, ββ = np.meshgrid(expected_alphas, expected_betas)

        test_means = beta.mean(αα, ββ, scale=test_xmax)
        test_vars = beta.var(αα, ββ, scale=test_xmax)
        alphas, betas = conversions.alpha_beta_from_mu_var(test_means, test_vars, xmax=test_xmax)
        assert_allclose(alphas, αα, err_msg="αs do not match")
        assert_allclose(betas, ββ, err_msg="βs do not match")
