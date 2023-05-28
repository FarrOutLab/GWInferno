from .conversions import chip_from_q_component_spins

"""
**********************************************************************************************************
THESE FUNCTIONS ARE DIRECTLY TAKEN FROM TOM CALLISTER AT:

https://github.com/tcallister/effective-spin-priors/blob/main/priors.py

**************************************************************************************************************
"""
import numpy as np
from scipy.special import spence as PL
from scipy.stats import gaussian_kde


def Di(z):

    """
    Wrapper for the scipy implmentation of Spence's function.
    Note that we adhere to the Mathematica convention as detailed in:
    https://reference.wolfram.com/language/ref/PolyLog.html
    Inputs
    z: A (possibly complex) scalar or array
    Returns
    Array equivalent to PolyLog[2,z], as defined by Mathematica
    """

    return PL(1.0 - z + 0j)


def chi_effective_prior_from_aligned_spins(q, aMax, xs):

    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, aligned component spin priors.
    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior
    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(xs, -1)

    # Set up various piecewise cases
    pdfs = np.zeros(xs.size)
    caseA = (xs > aMax * (1.0 - q) / (1.0 + q)) * (xs <= aMax)
    caseB = (xs < -aMax * (1.0 - q) / (1.0 + q)) * (xs >= -aMax)
    caseC = (xs >= -aMax * (1.0 - q) / (1.0 + q)) * (xs <= aMax * (1.0 - q) / (1.0 + q))

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    # x_C = xs[caseC]

    pdfs[caseA] = (1.0 + q) ** 2.0 * (aMax - x_A) / (4.0 * q * aMax**2)
    pdfs[caseB] = (1.0 + q) ** 2.0 * (aMax + x_B) / (4.0 * q * aMax**2)
    pdfs[caseC] = (1.0 + q) / (2.0 * aMax)

    return pdfs


def chi_effective_prior_from_isotropic_spins(q, aMax, xs):

    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors.
    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior
    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(np.abs(xs), -1)

    # Set up various piecewise cases
    pdfs = np.ones(xs.size, dtype=complex) * (-1.0)
    caseZ = xs == 0
    caseA = (xs > 0) * (xs < aMax * (1.0 - q) / (1.0 + q)) * (xs < q * aMax / (1.0 + q))
    caseB = (xs < aMax * (1.0 - q) / (1.0 + q)) * (xs > q * aMax / (1.0 + q))
    caseC = (xs > aMax * (1.0 - q) / (1.0 + q)) * (xs < q * aMax / (1.0 + q))
    caseD = (xs > aMax * (1.0 - q) / (1.0 + q)) * (xs < aMax / (1.0 + q)) * (xs >= q * aMax / (1.0 + q))
    caseE = (xs > aMax * (1.0 - q) / (1.0 + q)) * (xs > aMax / (1.0 + q)) * (xs < aMax)
    caseF = xs >= aMax

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]
    x_D = xs[caseD]
    x_E = xs[caseE]

    pdfs[caseZ] = (1.0 + q) / (2.0 * aMax) * (2.0 - np.log(q))

    pdfs[caseA] = (
        (1.0 + q)
        / (4.0 * q * aMax**2)
        * (
            q * aMax * (4.0 + 2.0 * np.log(aMax) - np.log(q**2 * aMax**2 - (1.0 + q) ** 2 * x_A**2))
            - 2.0 * (1.0 + q) * x_A * np.arctanh((1.0 + q) * x_A / (q * aMax))
            + (1.0 + q) * x_A * (Di(-q * aMax / ((1.0 + q) * x_A)) - Di(q * aMax / ((1.0 + q) * x_A)))
        )
    )

    pdfs[caseB] = (
        (1.0 + q)
        / (4.0 * q * aMax**2)
        * (
            4.0 * q * aMax
            + 2.0 * q * aMax * np.log(aMax)
            - 2.0 * (1.0 + q) * x_B * np.arctanh(q * aMax / ((1.0 + q) * x_B))
            - q * aMax * np.log((1.0 + q) ** 2 * x_B**2 - q**2 * aMax**2)
            + (1.0 + q) * x_B * (Di(-q * aMax / ((1.0 + q) * x_B)) - Di(q * aMax / ((1.0 + q) * x_B)))
        )
    )

    pdfs[caseC] = (
        (1.0 + q)
        / (4.0 * q * aMax**2)
        * (
            2.0 * (1.0 + q) * (aMax - x_C)
            - (1.0 + q) * x_C * np.log(aMax) ** 2.0
            + (aMax + (1.0 + q) * x_C * np.log((1.0 + q) * x_C)) * np.log(q * aMax / (aMax - (1.0 + q) * x_C))
            - (1.0 + q) * x_C * np.log(aMax) * (2.0 + np.log(q) - np.log(aMax - (1.0 + q) * x_C))
            + q * aMax * np.log(aMax / (q * aMax - (1.0 + q) * x_C))
            + (1.0 + q) * x_C * np.log((aMax - (1.0 + q) * x_C) * (q * aMax - (1.0 + q) * x_C) / q)
            + (1.0 + q) * x_C * (Di(1.0 - aMax / ((1.0 + q) * x_C)) - Di(q * aMax / ((1.0 + q) * x_C)))
        )
    )

    pdfs[caseD] = (
        (1.0 + q)
        / (4.0 * q * aMax**2)
        * (
            -x_D * np.log(aMax) ** 2
            + 2.0 * (1.0 + q) * (aMax - x_D)
            + q * aMax * np.log(aMax / ((1.0 + q) * x_D - q * aMax))
            + aMax * np.log(q * aMax / (aMax - (1.0 + q) * x_D))
            - x_D * np.log(aMax) * (2.0 * (1.0 + q) - np.log((1.0 + q) * x_D) - q * np.log((1.0 + q) * x_D / aMax))
            + (1.0 + q) * x_D * np.log((-q * aMax + (1.0 + q) * x_D) * (aMax - (1.0 + q) * x_D) / q)
            + (1.0 + q) * x_D * np.log(aMax / ((1.0 + q) * x_D)) * np.log((aMax - (1.0 + q) * x_D) / q)
            + (1.0 + q) * x_D * (Di(1.0 - aMax / ((1.0 + q) * x_D)) - Di(q * aMax / ((1.0 + q) * x_D)))
        )
    )

    pdfs[caseE] = (
        (1.0 + q)
        / (4.0 * q * aMax**2)
        * (
            2.0 * (1.0 + q) * (aMax - x_E)
            - (1.0 + q) * x_E * np.log(aMax) ** 2
            + np.log(aMax) * (aMax - 2.0 * (1.0 + q) * x_E - (1.0 + q) * x_E * np.log(q / ((1.0 + q) * x_E - aMax)))
            - aMax * np.log(((1.0 + q) * x_E - aMax) / q)
            + (1.0 + q) * x_E * np.log(((1.0 + q) * x_E - aMax) * ((1.0 + q) * x_E - q * aMax) / q)
            + (1.0 + q) * x_E * np.log((1.0 + q) * x_E) * np.log(q * aMax / ((1.0 + q) * x_E - aMax))
            - q * aMax * np.log(((1.0 + q) * x_E - q * aMax) / aMax)
            + (1.0 + q) * x_E * (Di(1.0 - aMax / ((1.0 + q) * x_E)) - Di(q * aMax / ((1.0 + q) * x_E)))
        )
    )

    pdfs[caseF] = 0.0

    # Deal with spins on the boundary between cases
    if np.any(pdfs == -1):
        boundary = pdfs == -1
        pdfs[boundary] = 0.5 * (
            chi_effective_prior_from_isotropic_spins(q, aMax, xs[boundary] + 1e-6)
            + chi_effective_prior_from_isotropic_spins(q, aMax, xs[boundary] - 1e-6)
        )

    return np.real(pdfs)


def chi_p_prior_from_isotropic_spins(q, aMax, xs):

    """
    Function defining the conditional priors p(chi_p|q) corresponding to
    uniform, isotropic component spin priors.
    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_p value or values at which we wish to compute prior
    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(xs, -1)

    # Set up various piecewise cases
    pdfs = np.zeros(xs.size)
    caseA = xs < q * aMax * (3.0 + 4.0 * q) / (4.0 + 3.0 * q)
    caseB = (xs >= q * aMax * (3.0 + 4.0 * q) / (4.0 + 3.0 * q)) * (xs < aMax)

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]

    pdfs[caseA] = (
        (1.0 / (aMax**2 * q))
        * ((4.0 + 3.0 * q) / (3.0 + 4.0 * q))
        * (
            np.arccos((4.0 + 3.0 * q) * x_A / ((3.0 + 4.0 * q) * q * aMax)) * (aMax - np.sqrt(aMax**2 - x_A**2) + x_A * np.arccos(x_A / aMax))
            + np.arccos(x_A / aMax)
            * (
                aMax * q * (3.0 + 4.0 * q) / (4.0 + 3.0 * q)
                - np.sqrt(aMax**2 * q**2 * ((3.0 + 4.0 * q) / (4.0 + 3.0 * q)) ** 2 - x_A**2)
                + x_A * np.arccos((4.0 + 3.0 * q) * x_A / ((3.0 + 4.0 * q) * aMax * q))
            )
        )
    )

    pdfs[caseB] = (1.0 / aMax) * np.arccos(x_B / aMax)

    return pdfs


def joint_prior_from_isotropic_spins(q, aMax, xeffs, xps, ndraws=10000, bw_method="scott"):

    """
    Function to calculate the conditional priors p(xp|xeff,q) on a set of {xp,xeff,q} posterior samples.
    INPUTS
    q: Mass ratio
    aMax: Maximimum spin magnitude considered
    xeffs: Effective inspiral spin samples
    xps: Effective precessing spin values
    ndraws: Number of draws from the component spin priors used in numerically building interpolant
    RETURNS
    p_chi_p: Array of priors on xp, conditioned on given effective inspiral spins and mass ratios
    """

    # Convert to arrays for safety
    xeffs = np.reshape(xeffs, -1)
    xps = np.reshape(xps, -1)

    # Compute marginal prior on xeff, conditional prior on xp, and multiply to get joint prior!
    p_chi_eff = chi_effective_prior_from_isotropic_spins(q, aMax, xeffs)
    p_chi_p_given_chi_eff = np.array([chi_p_prior_given_chi_eff_q(q, aMax, xeffs[i], xps[i], ndraws, bw_method) for i in range(len(xeffs))])
    joint_p_chi_p_chi_eff = p_chi_eff * p_chi_p_given_chi_eff

    return joint_p_chi_p_chi_eff


def chi_p_prior_given_chi_eff_q(q, aMax, xeff, xp, ndraws=10000, bw_method="scott"):

    """
    Function to calculate the conditional prior p(xp|xeff,q) on a single {xp,xeff,q} posterior sample.
    Called by `joint_prior_from_isotropic_spins`.
    INPUTS
    q: Single posterior mass ratio sample
    aMax: Maximimum spin magnitude considered
    xeff: Single effective inspiral spin sample
    xp: Single effective precessing spin value
    ndraws: Number of draws from the component spin priors used in numerically building interpolant
    RETURNS
    p_chi_p: Prior on xp, conditioned on given effective inspiral spin and mass ratio
    """

    # Draw random spin magnitudes.
    # Note that, given a fixed chi_eff, a1 can be no larger than (1+q)*chi_eff,
    # and a2 can be no larger than (1+q)*chi_eff/q
    a1 = np.random.random(ndraws) * aMax
    a2 = np.random.random(ndraws) * aMax

    # Draw random tilts for spin 2
    cost2 = 2.0 * np.random.random(ndraws) - 1.0

    # Finally, given our conditional value for chi_eff, we can solve for cost1
    # Note, though, that we still must require that the implied value of cost1 be *physical*
    cost1 = (xeff * (1.0 + q) - q * a2 * cost2) / a1

    # While any cost1 values remain unphysical, redraw a1, a2, and cost2, and recompute
    # Repeat as necessary
    while np.any(cost1 < -1) or np.any(cost1 > 1):
        to_replace = np.where((cost1 < -1) | (cost1 > 1))[0]
        a1[to_replace] = np.random.random(to_replace.size) * aMax
        a2[to_replace] = np.random.random(to_replace.size) * aMax
        cost2[to_replace] = 2.0 * np.random.random(to_replace.size) - 1.0
        cost1 = (xeff * (1.0 + q) - q * a2 * cost2) / a1

    # Compute precessing spins and corresponding weights, build KDE
    # See `Joint-ChiEff-ChiP-Prior.ipynb` for a discussion of these weights
    Xp_draws = chip_from_q_component_spins(q, a1, a2, cost1, cost2)
    jacobian_weights = (1.0 + q) / a1
    prior_kde = gaussian_kde(Xp_draws, weights=jacobian_weights, bw_method=bw_method)

    # Compute maximum chi_p
    if (1.0 + q) * np.abs(xeff) / q < aMax:
        max_Xp = aMax
    else:
        max_Xp = np.sqrt(aMax**2 - ((1.0 + q) * np.abs(xeff) - q) ** 2.0)

    # Set up a grid slightly inside (0,max chi_p) and evaluate KDE
    reference_grid = np.linspace(0.05 * max_Xp, 0.95 * max_Xp, 50)
    reference_vals = prior_kde(reference_grid)

    # Manually prepend/append zeros at the boundaries
    reference_grid = np.concatenate([[0], reference_grid, [max_Xp]])
    reference_vals = np.concatenate([[0], reference_vals, [0]])
    norm_constant = np.trapz(reference_vals, reference_grid)

    # Interpolate!
    p_chi_p = np.interp(xp, reference_grid, reference_vals / norm_constant)
    return p_chi_p
