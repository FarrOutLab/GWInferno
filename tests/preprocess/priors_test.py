import numpy as np
from numpy.testing import assert_allclose
from gwinferno.preprocess import priors


def test_chi_effective_prior_from_aligned_spins():
    eps = .01
    chi_eff_bounds = np.array([-1.0, 1.0])
    chi_eff = np.linspace(-1+eps, 1-eps, 1000)

    # Should be 0. at the bounds
    p_chi_eff = priors.chi_effective_prior_from_aligned_spins(chi_eff_bounds, q=1)
    assert_allclose(p_chi_eff, 0., err_msg="chi_eff prior not 0. at the bounds for q=1")
    p_chi_eff = priors.chi_effective_prior_from_aligned_spins(chi_eff_bounds, q=0.5)
    assert_allclose(p_chi_eff, 0., err_msg="chi_eff prior not 0. at the bounds for q=0.5")

    # Should be uniform in high mass ratio limit
    p_chi_eff = priors.chi_effective_prior_from_aligned_spins(chi_eff, q=1e-5)
    assert_allclose(p_chi_eff, 0.5, err_msg="chi_eff prior not uniform in high mass ratio limit")

    # Should be triangular in equal mass limit
    chi_eff = np.array([-1., -.5, 0., .5, 1.])
    p_chi_eff_expected = np.array([0., 0.5, 1., 0.5, 0.])
    p_chi_eff = priors.chi_effective_prior_from_aligned_spins(chi_eff, q=1)
    assert_allclose(p_chi_eff, p_chi_eff_expected, err_msg="chi_eff prior not triangular in equal mass limit")
