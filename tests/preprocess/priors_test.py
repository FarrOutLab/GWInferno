import itertools

import numpy as np
from numpy.testing import assert_allclose

from gwinferno.preprocess import priors


def test_chi_effective_prior_from_aligned_spins():
    test_q = np.array([1e-6, 0.1, 0.5, 1.0])

    # Should be 0. at the bounds
    test_chi_eff_bounds = np.array([-1.0, 1.0])
    χχ, qq = np.meshgrid(test_chi_eff_bounds, test_q)
    pp = priors.chi_effective_prior_from_aligned_spins(χχ, qq, a_max=1.0)
    assert np.all(np.isfinite(pp)), "encountered non-finite prior density"
    assert_allclose(pp, 0.0, err_msg="chi_eff prior not 0. at the bounds for all q")

    # Should be uniform in high mass ratio limit
    eps = 0.01
    safe_test_chi_eff = np.linspace(-1 + eps, 1 - eps, 1000)
    p_chi_eff = priors.chi_effective_prior_from_aligned_spins(safe_test_chi_eff, q=test_q[0])
    assert np.all(np.isfinite(p_chi_eff)), "encountered non-finite prior density"
    assert_allclose(p_chi_eff, 0.5, rtol=1e-5, err_msg="chi_eff prior not uniform in high mass ratio limit")

    # Should be triangular in equal mass limit
    test_chi_eff = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    p_chi_eff_expected = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    p_chi_eff = priors.chi_effective_prior_from_aligned_spins(test_chi_eff, q=test_q[-1])
    assert np.all(np.isfinite(p_chi_eff)), "encountered non-finite prior density"
    assert_allclose(p_chi_eff, p_chi_eff_expected, err_msg="chi_eff prior not triangular in equal mass limit")

    # Test support and normalization over a_max's and q's
    test_q = np.array([0.1, 0.5, 1.0])
    test_chi_eff = np.linspace(-1.0, 1.0, 5000)
    χχ, qq = np.meshgrid(test_chi_eff, test_q)

    for a_max in [0.3, 0.6, 1.0]:
        pp = priors.chi_effective_prior_from_aligned_spins(χχ, qq, a_max=a_max)
        assert np.all(np.isfinite(pp)), "encountered non-finite prior density"
        expected_nulls = (χχ < -a_max) | (χχ > a_max)
        assert_allclose(
            pp[expected_nulls],
            0.0,
            err_msg="chi_eff prior not 0 for |χ| > {}".format(a_max),
        )

        prob = np.trapz(pp, χχ, axis=1)
        assert_allclose(
            prob,
            1.0,
            rtol=1e-5,
            err_msg="chi_eff prior not normalized over χ for {}".format(a_max),
        )


def test_chi_effective_prior_from_isotropic_spins():
    test_q = np.array([1e-6, 0.1, 0.5, 1.0])

    # Should be 0. at the bounds
    test_chi_eff_bounds = np.array([-1.0, 1.0])
    χχ, qq = np.meshgrid(test_chi_eff_bounds, test_q)
    pp = priors.chi_effective_prior_from_aligned_spins(χχ, qq, a_max=1.0)
    assert np.all(np.isfinite(pp)), "encountered non-finite prior density"
    assert_allclose(pp, 0.0, err_msg="chi_eff prior not 0. at the bounds for all q")

    # Test support and normalization over a_max's and q's
    test_q = np.array([0.1, 0.5, 1.0])
    test_chi_eff = np.linspace(-1.0, 1.0, 5000)
    χχ, qq = np.meshgrid(test_chi_eff, test_q)

    for a_max in [0.3, 0.6, 1.0]:
        pp = priors.chi_effective_prior_from_aligned_spins(χχ, qq, a_max=a_max)
        assert np.all(np.isfinite(pp)), "encountered non-finite prior density"
        expected_nulls = (χχ < -a_max) | (χχ > a_max)
        assert_allclose(
            pp[expected_nulls],
            0.0,
            err_msg="chi_eff prior not 0 for |χ| > {}".format(a_max),
        )

        prob = np.trapz(pp, χχ, axis=1)
        assert_allclose(
            prob,
            1.0,
            rtol=1e-5,
            err_msg="chi_eff prior not normalized over χ for {}".format(a_max),
        )


def test_chi_p_prior_from_isotropic_spins():
    test_q = np.array([1e-6, 0.1, 0.5, 1.0])

    # Should be 0. at the bounds
    test_chi_p_bounds = np.array([0.0, 1.0])
    χχ, qq = np.meshgrid(test_chi_p_bounds, test_q)
    pp = priors.chi_p_prior_from_isotropic_spins(χχ, qq, a_max=1.0)
    assert np.all(np.isfinite(pp)), "encountered non-finite prior density"
    assert_allclose(pp, 0.0, atol=1e-10, err_msg="chi_p prior not 0. at the bounds for all q")

    # Test support and normalization over a_max's and q's
    test_q = np.array([0.1, 0.5, 1.0])
    test_chi_p = np.linspace(0.0, 1.0, 5000)
    χχ, qq = np.meshgrid(test_chi_p, test_q)

    for a_max in [0.3, 0.6, 1.0]:
        pp = priors.chi_p_prior_from_isotropic_spins(χχ, qq, a_max=a_max)
        assert np.all(np.isfinite(pp)), "encountered non-finite prior density"
        expected_nulls = (χχ < -a_max) | (χχ > a_max)
        assert_allclose(
            pp[expected_nulls],
            0.0,
            err_msg="chi_p prior not 0 for |χ| > {}".format(a_max),
        )

        prob = np.trapz(pp, χχ, axis=1)
        assert_allclose(
            prob,
            1.0,
            rtol=1e-5,
            err_msg="chi_p prior not normalized over χ for {}".format(a_max),
        )


def test_chi_p_prior_given_chi_eff_q():
    chi_p_prior_given_chi_eff_q_vectorized = np.vectorize(
        priors.chi_p_prior_given_chi_eff_q,
        excluded=["a_max", "ndraws", "bw_method"],
    )
    eps = 0.1  # avoid extremal values where internal monte carlo has trouble drawing physical tilts
    safe_test_chi_eff = np.linspace(-1 + eps, 1 - eps, 5)
    safe_test_chi_p = np.linspace(0 + eps, 1 - eps, 5)
    ee, pp = np.meshgrid(safe_test_chi_eff, safe_test_chi_p)

    a_max = 1.0
    q = 1.0
    pdf = chi_p_prior_given_chi_eff_q_vectorized(pp, ee, q, a_max=a_max)
    assert np.all(np.isfinite(pdf)), "encountered non-finite prior density"

    # Should be 0. at the bounds
    test_q = np.array([0.1, 0.5, 1.0])
    test_chi_eff = np.array([-0.4, 0.0, 0.6])
    test_chi_p_bounds = np.array([0.0, 1.0])
    pp, ee, qq = np.meshgrid(test_chi_p_bounds, test_chi_eff, test_q)
    pdf = chi_p_prior_given_chi_eff_q_vectorized(pp, ee, qq, a_max=1.0)
    assert np.all(np.isfinite(pdf)), "encountered non-finite prior density"
    assert_allclose(pdf, 0.0, atol=1e-10, err_msg="chi_p prior not 0. at the bounds for all q")

    # Test support and normalization over chi_eff's, q's
    test_qs = [0.5, 1.0]
    test_chi_effs = [0.0, 0.3]
    a_max = 0.9
    test_chi_p = np.linspace(0.0, 1.0, 5000)

    for test_chi_eff, test_q in itertools.product(test_chi_effs, test_qs):
        pdf = priors.chi_p_prior_given_chi_eff_q(test_chi_p, test_chi_eff, test_q, a_max=a_max)
        assert np.all(np.isfinite(pdf)), "encountered non-finite prior density"
        expected_nulls = (test_chi_p < -a_max) | (test_chi_p > a_max)
        assert_allclose(
            pdf[expected_nulls],
            0.0,
            err_msg="chi_p prior not 0 for |χ| > {}".format(a_max),
        )

        prob = np.trapz(pdf, test_chi_p)
        assert_allclose(
            prob,
            1.0,
            rtol=1e-5,
            err_msg="chi_p prior not normalized over χ for {}".format(a_max),
        )


def test_joint_prior_from_isotropic_spins():
    eps = 0.1  # avoid extremal values where internal monte carlo has trouble drawing physical tilts
    safe_test_chi_eff = np.linspace(-1 + eps, 1 - eps, 5)
    safe_test_chi_p = np.linspace(0 + eps, 1 - eps, 5)
    ee, pp = np.meshgrid(safe_test_chi_eff, safe_test_chi_p)

    a_max = 1.0
    q = 1.0
    probprob = priors.joint_prior_from_isotropic_spins(pp, ee, q, a_max=a_max)
    assert np.all(np.isfinite(probprob)), "encountered non-finite prior density"
