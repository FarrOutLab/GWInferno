"""
a module that stores useful prior functions to evaluate spin priors in terms of effective spin paramters
"""

import numpy as np
from scipy.special import spence as PL
from scipy.stats import gaussian_kde

from .conversions import chip_from_q_component_spins

"""
**********************************************************************************************************
THESE FUNCTIONS ARE DIRECTLY TAKEN FROM TOM CALLISTER AT:

https://github.com/tcallister/effective-spin-priors/blob/main/priors.py

**************************************************************************************************************
"""


def Di(z):
    """Wrapper for the scipy implementation of Spence's function.
    Note that we adhere to the Mathematica convention as detailed in:
    https://reference.wolfram.com/language/ref/PolyLog.html

    Parameters
    ==========
    z : float, complex, array-like
        Input scalar or array-like value(s) at which to evaluate the dilogarithm.

    Returns
    =======
    array-like
        Array equivalent to PolyLog[2,z], as defined by Mathematica.
    """
    return PL(1.0 - z + 0j)


def chi_effective_prior_from_aligned_spins(chi_eff, q, a_max=1.0):
    r"""Calculate the conditional prior density for effective spin :math:`p(\chi_\mathrm{eff}\mid q)`
    corresponding to uniform and aligned component spin priors.

    Parameters
    ==========
    chi_eff : float, array-like
        Effective spin value(s) at which to compute the prior density.
    q : float, array-like
        Mass ratio value(s) to condition on. Expected convention :math:`q<1`.
    a_max : float, default=1.0
        Maximum allowed dimensionless component spin magnitude.

    Returns
    =======
    array-like
        Prior density at given :math:`\chi_\mathrm{eff}` value(s).
    """
    # Ensure that `chi_eff` is an array and take absolute value
    chi_eff = np.reshape(chi_eff, -1)

    # Set up various piecewise cases
    pdfs = np.zeros(chi_eff.size)
    caseA = (chi_eff > a_max * (1.0 - q) / (1.0 + q)) * (chi_eff <= a_max)
    caseB = (chi_eff < -a_max * (1.0 - q) / (1.0 + q)) * (chi_eff >= -a_max)
    caseC = (chi_eff >= -a_max * (1.0 - q) / (1.0 + q)) * (chi_eff <= a_max * (1.0 - q) / (1.0 + q))

    # Select relevant effective spins
    chi_eff_A = chi_eff[caseA]
    chi_eff_B = chi_eff[caseB]
    # chi_eff_C = chi_eff[caseC]

    pdfs[caseA] = (1.0 + q) ** 2.0 * (a_max - chi_eff_A) / (4.0 * q * a_max**2)
    pdfs[caseB] = (1.0 + q) ** 2.0 * (a_max + chi_eff_B) / (4.0 * q * a_max**2)
    pdfs[caseC] = (1.0 + q) / (2.0 * a_max)

    return pdfs


def chi_effective_prior_from_isotropic_spins(chi_eff, q, a_max=1.0):
    r"""Calculate the conditional prior density for effective spin :math:`p(\chi_\mathrm{eff}\mid q)`
    corresponding to uniform and isotropic component spin priors.

    Parameters
    ==========
    chi_eff : float, array-like
        Effective spin value(s) at which to compute the prior density.
    q : float, array-like
        Mass ratio value(s) to condition on. Expected convention :math:`q<1`.
    a_max : float, default=1.0
        Maximum allowed dimensionless component spin magnitude.

    Returns
    =======
    array-like
        Prior density at given effective spin value(s).
    """
    # Ensure that `chi_eff` is an array and take absolute value
    chi_eff = np.reshape(np.abs(chi_eff), -1)

    # Set up various piecewise cases
    pdfs = np.ones(chi_eff.size, dtype=complex) * (-1.0)
    caseZ = chi_eff == 0
    caseA = (chi_eff > 0) * (chi_eff < a_max * (1.0 - q) / (1.0 + q)) * (chi_eff < q * a_max / (1.0 + q))
    caseB = (chi_eff < a_max * (1.0 - q) / (1.0 + q)) * (chi_eff > q * a_max / (1.0 + q))
    caseC = (chi_eff > a_max * (1.0 - q) / (1.0 + q)) * (chi_eff < q * a_max / (1.0 + q))
    caseD = (chi_eff > a_max * (1.0 - q) / (1.0 + q)) * (chi_eff < a_max / (1.0 + q)) * (chi_eff >= q * a_max / (1.0 + q))
    caseE = (chi_eff > a_max * (1.0 - q) / (1.0 + q)) * (chi_eff > a_max / (1.0 + q)) * (chi_eff < a_max)
    caseF = chi_eff >= a_max

    # Select relevant effective spins
    chi_eff_A = chi_eff[caseA]
    chi_eff_B = chi_eff[caseB]
    chi_eff_C = chi_eff[caseC]
    chi_eff_D = chi_eff[caseD]
    chi_eff_E = chi_eff[caseE]

    pdfs[caseZ] = (1.0 + q) / (2.0 * a_max) * (2.0 - np.log(q))

    pdfs[caseA] = (
        (1.0 + q)
        / (4.0 * q * a_max**2)
        * (
            q * a_max * (4.0 + 2.0 * np.log(a_max) - np.log(q**2 * a_max**2 - (1.0 + q) ** 2 * chi_eff_A**2))
            - 2.0 * (1.0 + q) * chi_eff_A * np.arctanh((1.0 + q) * chi_eff_A / (q * a_max))
            + (1.0 + q) * chi_eff_A * (Di(-q * a_max / ((1.0 + q) * chi_eff_A)) - Di(q * a_max / ((1.0 + q) * chi_eff_A)))
        )
    )

    pdfs[caseB] = (
        (1.0 + q)
        / (4.0 * q * a_max**2)
        * (
            4.0 * q * a_max
            + 2.0 * q * a_max * np.log(a_max)
            - 2.0 * (1.0 + q) * chi_eff_B * np.arctanh(q * a_max / ((1.0 + q) * chi_eff_B))
            - q * a_max * np.log((1.0 + q) ** 2 * chi_eff_B**2 - q**2 * a_max**2)
            + (1.0 + q) * chi_eff_B * (Di(-q * a_max / ((1.0 + q) * chi_eff_B)) - Di(q * a_max / ((1.0 + q) * chi_eff_B)))
        )
    )

    pdfs[caseC] = (
        (1.0 + q)
        / (4.0 * q * a_max**2)
        * (
            2.0 * (1.0 + q) * (a_max - chi_eff_C)
            - (1.0 + q) * chi_eff_C * np.log(a_max) ** 2.0
            + (a_max + (1.0 + q) * chi_eff_C * np.log((1.0 + q) * chi_eff_C)) * np.log(q * a_max / (a_max - (1.0 + q) * chi_eff_C))
            - (1.0 + q) * chi_eff_C * np.log(a_max) * (2.0 + np.log(q) - np.log(a_max - (1.0 + q) * chi_eff_C))
            + q * a_max * np.log(a_max / (q * a_max - (1.0 + q) * chi_eff_C))
            + (1.0 + q) * chi_eff_C * np.log((a_max - (1.0 + q) * chi_eff_C) * (q * a_max - (1.0 + q) * chi_eff_C) / q)
            + (1.0 + q) * chi_eff_C * (Di(1.0 - a_max / ((1.0 + q) * chi_eff_C)) - Di(q * a_max / ((1.0 + q) * chi_eff_C)))
        )
    )

    pdfs[caseD] = (
        (1.0 + q)
        / (4.0 * q * a_max**2)
        * (
            -chi_eff_D * np.log(a_max) ** 2
            + 2.0 * (1.0 + q) * (a_max - chi_eff_D)
            + q * a_max * np.log(a_max / ((1.0 + q) * chi_eff_D - q * a_max))
            + a_max * np.log(q * a_max / (a_max - (1.0 + q) * chi_eff_D))
            - chi_eff_D * np.log(a_max) * (2.0 * (1.0 + q) - np.log((1.0 + q) * chi_eff_D) - q * np.log((1.0 + q) * chi_eff_D / a_max))
            + (1.0 + q) * chi_eff_D * np.log((-q * a_max + (1.0 + q) * chi_eff_D) * (a_max - (1.0 + q) * chi_eff_D) / q)
            + (1.0 + q) * chi_eff_D * np.log(a_max / ((1.0 + q) * chi_eff_D)) * np.log((a_max - (1.0 + q) * chi_eff_D) / q)
            + (1.0 + q) * chi_eff_D * (Di(1.0 - a_max / ((1.0 + q) * chi_eff_D)) - Di(q * a_max / ((1.0 + q) * chi_eff_D)))
        )
    )

    pdfs[caseE] = (
        (1.0 + q)
        / (4.0 * q * a_max**2)
        * (
            2.0 * (1.0 + q) * (a_max - chi_eff_E)
            - (1.0 + q) * chi_eff_E * np.log(a_max) ** 2
            + np.log(a_max) * (a_max - 2.0 * (1.0 + q) * chi_eff_E - (1.0 + q) * chi_eff_E * np.log(q / ((1.0 + q) * chi_eff_E - a_max)))
            - a_max * np.log(((1.0 + q) * chi_eff_E - a_max) / q)
            + (1.0 + q) * chi_eff_E * np.log(((1.0 + q) * chi_eff_E - a_max) * ((1.0 + q) * chi_eff_E - q * a_max) / q)
            + (1.0 + q) * chi_eff_E * np.log((1.0 + q) * chi_eff_E) * np.log(q * a_max / ((1.0 + q) * chi_eff_E - a_max))
            - q * a_max * np.log(((1.0 + q) * chi_eff_E - q * a_max) / a_max)
            + (1.0 + q) * chi_eff_E * (Di(1.0 - a_max / ((1.0 + q) * chi_eff_E)) - Di(q * a_max / ((1.0 + q) * chi_eff_E)))
        )
    )

    pdfs[caseF] = 0.0

    # Deal with spins on the boundary between cases
    if np.any(pdfs == -1):
        boundary = pdfs == -1
        pdfs[boundary] = 0.5 * (
            chi_effective_prior_from_isotropic_spins(chi_eff[boundary] + 1e-6, q, a_max=a_max)
            + chi_effective_prior_from_isotropic_spins(chi_eff[boundary] - 1e-6, q, a_max=a_max)
        )

    return np.real(pdfs)


def chi_p_prior_from_isotropic_spins(chi_eff, q, a_max=1.0):
    r"""Calculate the conditional prior density for effective precession :math:`p(\chi_\mathrm{p}\mid q)`
    corresponding to uniform and isotropic component spin priors.

    Parameters
    ==========
    chi_eff : float, array-like
        Effective spin value(s) at which to compute the prior density.
    q : float, array-like
        Mass ratio value(s) to condition on. Expected convention :math:`q<1`.
    a_max : float, default=1.0
        Maximum allowed dimensionless component spin magnitude.

    Returns
    =======
    array-like
        Prior density at given effective spin value(s).
    """
    # Ensure that `chi_eff` is an array and take absolute value
    chi_eff = np.reshape(chi_eff, -1)

    # Set up various piecewise cases
    pdfs = np.zeros(chi_eff.size)
    caseA = chi_eff < q * a_max * (3.0 + 4.0 * q) / (4.0 + 3.0 * q)
    caseB = (chi_eff >= q * a_max * (3.0 + 4.0 * q) / (4.0 + 3.0 * q)) * (chi_eff < a_max)

    # Select relevant effective spins
    chi_eff_A = chi_eff[caseA]
    chi_eff_B = chi_eff[caseB]

    pdfs[caseA] = (
        (1.0 / (a_max**2 * q))
        * ((4.0 + 3.0 * q) / (3.0 + 4.0 * q))
        * (
            np.arccos((4.0 + 3.0 * q) * chi_eff_A / ((3.0 + 4.0 * q) * q * a_max))
            * (a_max - np.sqrt(a_max**2 - chi_eff_A**2) + chi_eff_A * np.arccos(chi_eff_A / a_max))
            + np.arccos(chi_eff_A / a_max)
            * (
                a_max * q * (3.0 + 4.0 * q) / (4.0 + 3.0 * q)
                - np.sqrt(a_max**2 * q**2 * ((3.0 + 4.0 * q) / (4.0 + 3.0 * q)) ** 2 - chi_eff_A**2)
                + chi_eff_A * np.arccos((4.0 + 3.0 * q) * chi_eff_A / ((3.0 + 4.0 * q) * a_max * q))
            )
        )
    )

    pdfs[caseB] = (1.0 / a_max) * np.arccos(chi_eff_B / a_max)

    return pdfs


def joint_prior_from_isotropic_spins(chi_p, chi_eff, q, a_max=1.0, **kwargs):
    r"""Calculate the conditional prior density for effective precession conditioned on effective
    spin and mass ratio, :math:`p(\chi_\mathrm{p}\mid \chi_\mathrm{eff}, q)`
    corresponding to uniform and isotropic component spin priors.

    Parameters
    ==========
    chi_p : float, array-like
        Effective precession value(s) at which to compute the prior density.
    chi_eff : float, array-like
        Effective spin value(s) to condition on.
    q : float, array-like
        Mass ratio value(s) to condition on. Expected convention :math:`q<1`.
    a_max : float, default=1.0
        Maximum allowed dimensionless component spin magnitude.
    **kwargs: dict, optional
        Keyword arguments to pass to :func:`chi_p_prior_given_chi_eff_q`.

    See Also
    ========
    chi_effective_prior_from_isotropic_spins, chi_p_prior_given_chi_eff_q

    Returns
    =======
    array-like
        Prior density at given effective precession value(s).
    """
    # Convert to arrays for safety
    chi_p = np.reshape(chi_p, -1)
    chi_eff = np.reshape(chi_eff, -1)

    # Compute marginal prior on chi_eff, conditional prior on chi_p, and multiply to get joint prior!
    p_chi_eff = chi_effective_prior_from_isotropic_spins(chi_eff, q, a_max=a_max)
    p_chi_p_given_chi_eff = np.array([chi_p_prior_given_chi_eff_q(chi_p[i], chi_eff[i], q, a_max=a_max, **kwargs) for i in range(len(chi_eff))])
    joint_p_chi_p_chi_eff = p_chi_eff * p_chi_p_given_chi_eff

    return joint_p_chi_p_chi_eff


def chi_p_prior_given_chi_eff_q(chi_p, chi_eff, q, a_max=1.0, ndraws=10000, bw_method="scott"):
    r"""Calculate the conditional prior density for effective precession conditioned on effective
    spin and mass ratio, :math:`p(\chi_\mathrm{p}\mid \chi_\mathrm{eff}, q)`, for a single
    sample in :math:`\{\chi_\mathrm{p},\chi_\mathrm{eff},q\}`, corresponding to uniform
    and isotropic component spin priors.

    The prior density is computed numerically by:
        1. drawing random component spins and tilts,
        2. computing the corresponding effective precession values,
        3. training a kernel density estimate (KDE),
        4. building an interpolant using the KDE's density evaluated over a grid.

    Parameters
    ==========
    chi_p : float
        Effective precession value at which to compute the prior density.
    chi_eff : float
        Effective spin value to condition on.
    q : float
        Mass ratio value to condition on. Expected convention :math:`q<1`.
    a_max : float, default=1.0
        Maximum allowed dimensionless component spin magnitude.
    ndraws : int, default=10000
        Number of draws from the component spin priors used to train KDE.
    **kwargs: dict, optional
        Keyword arguments to pass to :func:`chi_p_prior_given_chi_eff_q`.

    See Also
    ========
    joint_prior_from_isotropic_spins

    Returns
    =======
    array-like
        Prior density at given effective precession value.
    """
    # Draw random spin magnitudes.
    # Note that, given a fixed chi_eff, a1 can be no larger than (1+q)*chi_eff,
    # and a2 can be no larger than (1+q)*chi_eff/q
    a1 = np.random.random(ndraws) * a_max
    a2 = np.random.random(ndraws) * a_max

    # Draw random tilts for spin 2
    cost2 = 2.0 * np.random.random(ndraws) - 1.0

    # Finally, given our conditional value for chi_eff, we can solve for cost1
    # Note, though, that we still must require that the implied value of cost1 be *physical*
    cost1 = (chi_eff * (1.0 + q) - q * a2 * cost2) / a1

    # While any cost1 values remain unphysical, redraw a1, a2, and cost2, and recompute
    # Repeat as necessary
    while np.any(cost1 < -1) or np.any(cost1 > 1):
        to_replace = np.where((cost1 < -1) | (cost1 > 1))[0]
        a1[to_replace] = np.random.random(to_replace.size) * a_max
        a2[to_replace] = np.random.random(to_replace.size) * a_max
        cost2[to_replace] = 2.0 * np.random.random(to_replace.size) - 1.0
        cost1 = (chi_eff * (1.0 + q) - q * a2 * cost2) / a1

    # Compute precessing spins and corresponding weights, build KDE
    # See `Joint-ChiEff-ChiP-Prior.ipynb` for a discussion of these weights
    chi_p_draws = chip_from_q_component_spins(q, a1, a2, cost1, cost2)
    jacobian_weights = (1.0 + q) / a1
    prior_kde = gaussian_kde(chi_p_draws, weights=jacobian_weights, bw_method=bw_method)

    # Compute maximum chi_p
    if (1.0 + q) * np.abs(chi_eff) / q < a_max:
        max_chi_p = a_max
    else:
        max_chi_p = np.sqrt(a_max**2 - ((1.0 + q) * np.abs(chi_eff) - q) ** 2.0)

    # Set up a grid slightly inside (0,max chi_p) and evaluate KDE
    reference_grid = np.linspace(0.05 * max_chi_p, 0.95 * max_chi_p, 50)
    reference_vals = prior_kde(reference_grid)

    # Manually prepend/append zeros at the boundaries
    reference_grid = np.concatenate([[0], reference_grid, [max_chi_p]])
    reference_vals = np.concatenate([[0], reference_vals, [0]])
    norm_constant = np.trapz(reference_vals, reference_grid)

    # Interpolate!
    p_chi_p = np.interp(chi_p, reference_grid, reference_vals / norm_constant)
    return p_chi_p
