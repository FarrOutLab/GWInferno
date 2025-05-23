import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

from gwinferno.cosmology import PLANCK_2015_LVK_Cosmology as Planck15

from ...distributions import betadist
from ...distributions import powerlaw_logit_pdf
from ...distributions import powerlaw_pdf
from ...distributions import smooth
from ...distributions import truncnorm_pdf

# subset Of Models From https://github.com/ColmTalbot/gwpopulation

"""
=============================================
This file contains a small subset of functions/models from https://github.com/ColmTalbot/gwpopulation re-implemented with jax.numpy
=============================================
"""

"""
***************************************
MASS MODELS
***************************************
"""


def powerlaw_primary_ratio_pdf(m1, q, alpha, beta, mmin, mmax):
    p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
    p_m1 = powerlaw_pdf(m1, alpha, mmin, mmax)
    return p_q * p_m1


def powerlaw_primary_ratio_falloff_pdf(m1, q, alpha, beta, mmin, mmax, fall_off):
    p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
    p_m1 = powerlaw_logit_pdf(m1, alpha, mmin, mmax, fall_off)
    return p_q * p_m1


def plpeak_primary_ratio_pdf(m1, q, alpha, beta, mmin, mmax, mpp, sigpp, lam, delta=None):
    p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
    p_m1 = plpeak_primary_pdf(m1, alpha, mmin, mmax, mpp, sigpp, lam, delta=delta)

    if delta is None:
        return p_q * p_m1
    else:
        return p_q * smooth(delta, q * m1, mmin) * p_m1


def plpeak_primary_pdf(m1, alpha, mmin, mmax, mpp, sigpp, lam, delta=None):
    if delta is None:
        return (1 - lam) * powerlaw_pdf(m1, alpha, mmin, mmax) + lam * truncnorm_pdf(m1, mpp, sigpp, mmin, mmax)
    else:
        return (1 - lam) * powerlaw_pdf(m1, alpha, mmin, mmax) * smooth(delta, m1, mmin) + lam * truncnorm_pdf(m1, mpp, sigpp, mmin, mmax)


"""
***************************************
SPIN MODELS
***************************************
"""


def beta_spin_magnitude(a, alpha, beta, amax=1):
    return betadist(a, alpha, beta, scale=amax)


def iid_spin_magnitude(a1, a2, alpha_mag, beta_mag, amax=1):
    return betadist(a1, alpha_mag, beta_mag, scale=amax) * betadist(a2, alpha_mag, beta_mag, scale=amax)


def independent_spin_magnitude_beta_dist(
    a1,
    a2,
    alpha_mag1,
    beta_mag1,
    alpha_mag2,
    beta_mag2,
    amax1=1,
    amax2=1,
):
    return betadist(a1, alpha_mag1, beta_mag1, scale=amax1) * betadist(a2, alpha_mag2, beta_mag2, scale=amax2)


def mixture_isoalign_spin_tilt(ct, xi_tilt, sigma_tilt):
    cut = jnp.where(jnp.greater(ct, 1) | jnp.less(ct, -1), 0, 1)
    return cut * (1 - xi_tilt) / 2 + xi_tilt * truncnorm_pdf(ct, 1, sigma_tilt, -1, 1)


def iid_spin_tilt(ct1, ct2, xi_tilt, sigma_tilt):
    return mixture_isoalign_spin_tilt(ct1, xi_tilt, sigma_tilt) * mixture_isoalign_spin_tilt(ct2, xi_tilt, sigma_tilt)


def independent_spin_tilt(ct1, ct2, xi_tilt_1, xi_tilt_2, sigma_tilt1, sigma_tilt2):
    return mixture_isoalign_spin_tilt(ct1, xi_tilt_1, sigma_tilt1) * mixture_isoalign_spin_tilt(ct2, xi_tilt_2, sigma_tilt2)


def default_spin_tilt(ct1, ct2, xi_tilt, sigma_tilt):
    iso1 = jnp.where(jnp.greater(ct1, 1) | jnp.less(ct1, -1), 0, 0.5)
    iso2 = jnp.where(jnp.greater(ct2, 1) | jnp.less(ct2, -1), 0, 0.5)
    ali1 = truncnorm_pdf(ct1, 1, sigma_tilt, -1, 1)
    ali2 = truncnorm_pdf(ct2, 1, sigma_tilt, -1, 1)
    return (1 - xi_tilt) * iso1 * iso2 + xi_tilt * ali1 * ali2


"""
***************************************
REDSHIFT MODELS
***************************************
"""


class PowerlawRedshiftModel(object):
    def __init__(self, z_pe, z_inj):
        self.zmin = jnp.max(jnp.array([jnp.min(z_pe), jnp.min(z_inj)]))
        self.zmax = jnp.min(jnp.array([jnp.max(z_pe), jnp.max(z_inj)]))
        self.zs = jnp.linspace(self.zmin, self.zmax, 1000)
        self.dVdz_ = jnp.array(Planck15.dVcdz(self.zs))
        self.dVdzs = [
            jnp.array(Planck15.dVcdz(z_inj)),
            jnp.array(Planck15.dVcdz(z_pe)),
        ]

    def normalization(self, lamb):
        return trapezoid(self.prob(self.zs, self.dVdz_, lamb), self.zs)

    def prob(self, z, dVdz, lamb):
        return dVdz * jnp.power(1.0 + z, lamb - 1.0)

    def log_prob(self, z, lamb):
        ndim = len(z.shape)
        dVdz = self.dVdzs[ndim - 1]
        return jnp.where(
            jnp.less_equal(z, self.zmax),
            jnp.log(dVdz) + (lamb - 1.0) * jnp.log(1.0 + z) - jnp.log(self.normalization(lamb)),
            jnp.nan_to_num(-jnp.inf),
        )

    def __call__(self, z, lamb):
        ndim = len(z.shape)
        dVdz = self.dVdzs[ndim - 1]
        return jnp.where(
            jnp.less_equal(z, self.zmax),
            self.prob(z, dVdz, lamb) / self.normalization(lamb),
            0,
        )
