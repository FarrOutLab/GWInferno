import jax.numpy as jnp

from ..distributions import powerlaw_pdf
from .gwpopulation.gwpopulation import plpeak_primary_pdf


# """log of logistic function (see https://git.ligo.org/will-farr/pisnmassfunctions.jl/-/blob/master/src/model.jl#L10 )"""
def logistic_unit(x):
    return 1 / (1 + jnp.exp(-x))


def powerlaw_gap_primary_ratio_pdf(m1, q, alpha, beta, mmin, mmax, gapmin):

    gapmax = gapmin + 50
    p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
    p_m1 = powerlaw_pdf(m1, -alpha, mmin, mmax)
    return jnp.where(jnp.less_equal(m1, gapmax) & jnp.greater_equal(m1, gapmin), 0, p_m1) * jnp.where(
        jnp.less_equal(m1 * q, gapmax) & jnp.greater_equal(m1 * q, gapmin), 0, p_q
    )


def powerlaw_gap_logit_primary_pdf(m1, q, alpha, beta, mmin, mmax, gapmin, fall_off, gapwidth):
    gapmax = gapmin + gapwidth
    floor_p = -10.0  # jnp.where(jnp.less(alpha, 0), alpha*2 - 1, -8)

    pl_alpha_1 = jnp.power(m1, -alpha) * 1 / (1 + jnp.exp(fall_off * (m1 - gapmin)))
    pl_alpha_2 = jnp.power(m1, -alpha) * 1 / (1 + jnp.exp(-fall_off * (m1 - gapmax))) * 1 / (1 + jnp.exp(fall_off * (m1 - mmax)))
    floor = jnp.power(10.0, floor_p)
    p_m1 = pl_alpha_1 + pl_alpha_2 + floor

    p_q = powerlaw_pdf(q, beta, mmin / m1, 1.0)

    return p_m1 * p_q


def powerlaw_gap_logit_ben_primary_pdf(m1, q, alpha, beta, mmin, mmax, gapmin):
    m_gap_high = 50.0 + gapmin
    k = 100
    x_gl = k * (m1 - gapmin) / gapmin
    x_gh = k * (m1 - m_gap_high) / m_gap_high
    x_max = k * (m1 - mmax) / mmax

    A = 1
    B = 1
    C = 1e-8
    D = 1e-8
    p_m1 = m1**-alpha * (A * logistic_unit(-x_gl) + B * logistic_unit(x_gh) * logistic_unit(-x_max))
    p_m1 += m1**-beta * (C * logistic_unit(x_gl) * logistic_unit(-x_gh) + D * logistic_unit(x_max))

    p_q = powerlaw_pdf(q, beta, mmin / m1, 1.0)

    return p_m1 * p_q


def powerlaw_gap_cutoff_primary_ratio_pdf(m1, q, alpha, beta, mmin, mmax, gapmin):
    gapmax = gapmin + 50

    p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
    gapmin_norm = powerlaw_pdf(gapmin, -alpha, mmin, mmax) / jnp.power(gapmin, -10)
    gapmax_norm = powerlaw_pdf(gapmax, -alpha, mmin, mmax) / jnp.power(gapmax, 10)
    p_m1_gap = jnp.where(
        jnp.less_equal(m1, gapmin + 25),
        gapmin_norm * jnp.power(m1, -10),
        gapmax_norm * jnp.power(m1, 10),
    )
    p_m1 = jnp.where(
        jnp.less_equal(m1, gapmin) | jnp.greater_equal(m1, gapmax),
        powerlaw_pdf(m1, -alpha, mmin, mmax),
        p_m1_gap,
    )
    return p_q * p_m1


def plpeak_gap_primary_ratio_pdf(m1, q, alpha, beta, mmin, mmax, mpp, sigpp, lam, gapmin, gapmax):
    p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
    p_m1 = plpeak_primary_pdf(m1, alpha, mmin, mmax, mpp, sigpp, lam)
    return jnp.where(jnp.less_equal(m1, gapmax) & jnp.greater_equal(m1, gapmin), 0, p_m1) * jnp.where(
        jnp.less_equal(m1 * q, gapmax) & jnp.greater_equal(m1 * q, gapmin), 0, p_q
    )
