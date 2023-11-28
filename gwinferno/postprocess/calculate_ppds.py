"""
a module with utilities for calculating population posterior distributions (i.e. Rates on Grids)
"""

import jax.numpy as jnp
import numpy as np
from jax import jit
from tqdm import trange

from gwinferno.interpolation import LogXLogYBSpline


def calculate_m1q_ppds_plbspline_model(posterior, mass_model, nknots, rate=None, mmin=5, m2min=3, mmax=100, **model_kwargs):
    ms = jnp.linspace(mmin, mmax, 1000)
    qs = jnp.linspace(mmin / mmax, 1, 1000)
    mm, qq = jnp.meshgrid(ms, qs)
    mpdfs = np.zeros((posterior["alpha"].shape[0], len(ms)))
    qpdfs = np.zeros((posterior["alpha"].shape[0], len(qs)))
    masspdf = mass_model(nknots, mm, ms, mmin=mmin, m2min=m2min, mmax=mmax, **model_kwargs)
    if rate is None:
        rate = np.ones_like(posterior["alpha"])

    @jit
    def calc_pdf(a, b, mi, ma, fs, r):
        p_mq = masspdf(mm, qq, alpha=a, beta=b, mmin=mi, mmax=ma, cs=fs)
        p_mq = jnp.where(jnp.isinf(p_mq) | jnp.isnan(p_mq), 0, p_mq)
        p_m = jnp.trapz(p_mq, qs, axis=0)
        p_q = jnp.trapz(p_mq, ms, axis=1)
        return p_m * r, p_q * r

    try:
        # loop through hyperposterior samples
        for ii in trange(posterior["alpha"].shape[0]):
            mpdfs[ii], qpdfs[ii] = calc_pdf(
                posterior["alpha"][ii],
                posterior["beta"][ii],
                posterior["mmin"][ii],
                posterior["mmax"][ii],
                posterior["mass_cs"][ii],
                rate[ii],
            )
    except KeyError:
        try:
            for ii in trange(posterior["alpha"].shape[0]):
                mpdfs[ii], qpdfs[ii] = calc_pdf(
                    posterior["alpha"][ii],
                    posterior["beta"][ii],
                    mmin,
                    posterior["mmax"][ii],
                    posterior["mass_cs"][ii],
                    rate[ii],
                )
        except KeyError:
            for ii in trange(posterior["alpha"].shape[0]):
                mpdfs[ii], qpdfs[ii] = calc_pdf(
                    posterior["alpha"][ii],
                    posterior["beta"][ii],
                    mmin,
                    85.0,
                    posterior["mass_cs"][ii],
                    rate[ii],
                )
    return mpdfs, qpdfs, ms, qs


def calculate_powerlaw_rate_of_z_ppds(lamb, rate, model):
    zs = model.zs
    rs = np.zeros((len(lamb), len(zs)))

    def calc_rz(la, r):
        return r * jnp.power(1.0 + zs, la)

    calc_rz = jit(calc_rz)
    _ = calc_rz(lamb[0], rate[0])
    for ii in trange(lamb.shape[0]):
        rs[ii] = calc_rz(lamb[ii], rate[ii])
    return rs, zs


def calculate_powerbspline_rate_of_z_ppds(lamb, z_cs, rate, model):
    zs = model.zs
    rs = np.zeros((len(lamb), len(zs)))

    def calc_rz(cs, la, r):
        return r * jnp.power(1.0 + zs, la) * jnp.exp(model.interpolator.project(model.norm_design_matrix, (model.nknots, 1), cs))

    calc_rz = jit(calc_rz)
    _ = calc_rz(z_cs[0], lamb[0], rate[0])
    for ii in trange(lamb.shape[0]):
        rs[ii] = calc_rz(z_cs[ii], lamb[ii], rate[ii])
    return rs, zs


def calculate_iid_spin_bspline_ppds(coefs, model, nknots, rate=None, xmin=0, xmax=1, k=4, ngrid=500, pop_frac=None, pop_num=None, **model_kwargs):
    xs = np.linspace(xmin, xmax, ngrid)
    pdf = model(nknots, xs, xs, xs, xs, degree=k - 1, **model_kwargs)
    pdfs = np.zeros((coefs.shape[0], len(xs)))
    if rate is None:
        rate = jnp.ones(coefs.shape[0])

    def calc_pdf(cs, r):
        return pdf.primary_model(cs)  # * r

    calc_pdf = jit(calc_pdf)
    _ = calc_pdf(coefs[0], rate[0])
    # loop through hyperposterior samples

    if pop_frac is None:
        for ii in trange(coefs.shape[0]):
            pdfs[ii] = calc_pdf(coefs[ii], rate[ii])
    else:
        for ii in trange(coefs.shape[0]):
            pdfs[ii] = calc_pdf(coefs[ii], rate[ii]) * pop_frac[ii][pop_num]
    return pdfs, xs


def calculate_ind_spin_bspline_ppds(coefs, scoefs, model, nknots, rate=None, xmin=0, xmax=1, k=4, ngrid=750, **model_kwargs):
    xs = jnp.linspace(xmin, xmax, ngrid)
    pdf = model(nknots, xs, xs, xs, xs, degree=k - 1, **model_kwargs)
    ppdfs = np.zeros((coefs.shape[0], len(xs)))
    spdfs = np.zeros((coefs.shape[0], len(xs)))
    if rate is None:
        rate = jnp.ones(coefs.shape[0])

    def calc_pdf(pcs, scs, r):
        return pdf.primary_model(pcs), pdf.secondary_model(scs)  # * r

    calc_pdf = jit(calc_pdf)
    _, _ = calc_pdf(coefs[0], scoefs[0], rate[0])
    # loop through hyperposterior samples
    for ii in trange(coefs.shape[0]):
        ppdfs[ii], spdfs[ii] = calc_pdf(coefs[ii], scoefs[ii], rate[ii])
    return ppdfs, spdfs, xs


def calculate_chieff_bspline_ppds(coefs, model, nknots, rate=None, xmin=-1, xmax=1, k=4, ngrid=750, **model_kwargs):
    xs = jnp.linspace(xmin, xmax, ngrid)
    pdf = model(nknots, xs, xs, degree=k - 1, **model_kwargs)
    pdfs = np.zeros((coefs.shape[0], len(xs)))
    if rate is None:
        rate = jnp.ones(coefs.shape[0])

    def calc_pdf(cs, r):
        return pdf(cs) * r

    calc_pdf = jit(calc_pdf)
    _ = calc_pdf(coefs[0], rate[0])
    # loop through hyperposterior samples
    for ii in trange(coefs.shape[0]):
        pdfs[ii] = calc_pdf(coefs[ii], rate[ii])
    return pdfs, xs


def calculate_m1q_bspline_ppds(
    mcoefs, qcoefs, mass_model, nknots, qknots, rate=None, mmin=3.0, m1mmin=3.0, mmax=100.0, pop_frac=1, num=None, **model_kwargs
):
    ms = np.linspace(mmin, mmax, 800)
    qs = np.linspace(mmin / mmax, 1, 800)
    mm, qq = np.meshgrid(ms, qs)
    mass_pdf = mass_model(nknots, qknots, mm, ms, qq, qs, m1min=mmin, m2min=mmin, mmax=mmax, **model_kwargs)
    mpdfs = np.zeros((mcoefs.shape[0], len(ms)))
    qpdfs = np.zeros((qcoefs.shape[0], len(qs)))
    if rate is None:
        rate = jnp.ones(mcoefs.shape[0])

    def calc_pdf(mcs, qcs, r, pop_frac):
        p_mq = mass_pdf(mcs, qcs)
        p_mq = jnp.where(jnp.less(mm, m1mmin) | jnp.less(mm * qq, mmin), 0, p_mq)
        p_m = jnp.trapz(p_mq, qs, axis=0)
        p_q = jnp.trapz(p_mq, ms, axis=1)
        return r * p_m * pop_frac / jnp.trapz(p_m, ms), r * p_q * pop_frac / jnp.trapz(p_q, qs)

    calc_pdf = jit(calc_pdf)
    # _ = calc_pdf(mcoefs[0], qcoefs[0], rate[0], pop_frac[0][0])
    # loop through hyperposterior samples
    if isinstance(pop_frac, int):
        for ii in trange(mcoefs.shape[0]):
            mpdfs[ii], qpdfs[ii] = calc_pdf(mcoefs[ii], qcoefs[ii], rate[ii], pop_frac)
    else:
        for ii in trange(mcoefs.shape[0]):
            mpdfs[ii], qpdfs[ii] = calc_pdf(mcoefs[ii], qcoefs[ii], rate[ii], pop_frac[ii][num - 1])
    return mpdfs, qpdfs, ms, qs


def calculate_m1q_bspline_weights(
    m, q, mcoefs, qcoefs, mass_model, nknots, qknots, rate=None, mmin=3.0, m1mmin=3.0, mmax=100.0, num=None, **model_kwargs
):
    mass_pdf = mass_model(nknots, qknots, m, m, q, q, m1min=mmin, m2min=mmin, mmax=mmax, **model_kwargs)
    rate = 1

    def calc_pdf(mcs, qcs, r):
        p_mq = mass_pdf(mcs, qcs)
        p_mq = jnp.where(jnp.less(m, m1mmin) | jnp.less(m * q, mmin), 0, p_mq)
        return r * p_mq

    calc_pdf = jit(calc_pdf)
    # loop through hyperposterior samples

    return calc_pdf(mcoefs, qcoefs, rate)


def calculate_iid_spin_bspline_weights(xs, coefs, model, nknots, rate=1, xmin=0, xmax=1, k=4, ngrid=500, pop_frac=None, pop_num=None, **model_kwargs):
    pdf = model(nknots, xs, xs, xs, xs, degree=k - 1, **model_kwargs)
    pdfs = np.zeros((coefs.shape[0], len(xs)))

    def calc_pdf(cs, r):
        return pdf.primary_model(cs)  # * r

    calc_pdf = jit(calc_pdf)
    # loop through hyperposterior samples
    try:
        for ii in range(coefs.shape[0]):
            pdfs[ii] = calc_pdf(coefs[ii], rate[ii])
    except IndexError:
        for ii in range(coefs.shape[0]):
            pdfs[ii] = calc_pdf(coefs[ii], rate)

    return pdfs


def calculate_m1_bspline_q_powerlaw_ppds(
    mcoefs, mass_model, nknots, rate=None, mmin=3.0, m1mmin=3.0, mmax=100.0, pop_frac=1, pop_num=2, basis=LogXLogYBSpline, **model_kwargs
):
    ms = np.linspace(m1mmin, mmax, 800)
    qs = np.linspace(mmin / mmax, 1, 800)
    mm, qq = np.meshgrid(ms, qs)
    mass_pdf = mass_model(
        nknots,
        mm,
        ms,
        mmin=mmin,
        mmax=mmax,
        basis=basis,
    )
    mpdfs = np.zeros((mcoefs.shape[0], len(ms)))
    qpdfs = np.zeros((mcoefs.shape[0], len(qs)))
    if rate is None:
        rate = jnp.ones(mcoefs.shape[0])

    def calc_pdf(mcs, r, pop_frac, beta):
        p_mq = mass_pdf(mm, qq, beta, mmin, mcs)
        p_mq = jnp.where(jnp.less(mm, m1mmin) | jnp.less(mm * qq, mmin), 0, p_mq)
        p_m = jnp.trapz(p_mq, qs, axis=0)
        p_q = jnp.trapz(p_mq, ms, axis=1)
        return r * p_m * pop_frac / jnp.trapz(p_m, ms), r * p_q * pop_frac / jnp.trapz(p_q, qs)

    calc_pdf = jit(calc_pdf)
    # loop through hyperposterior samples
    for ii in trange(mcoefs.shape[0]):
        mpdfs[ii], qpdfs[ii] = calc_pdf(mcoefs[ii], rate[ii], pop_frac[ii][pop_num], model_kwargs["beta"][ii])
    return mpdfs, qpdfs, ms, qs


def calculate_m1m2_bspline_ppds(
    mcoefs, mass_model, nknots, rate=None, mmin=5.0, mmax=100.0, pop_frac=1, basis=LogXLogYBSpline, pop_num=2, **model_kwargs
):
    ms1 = np.linspace(mmin, mmax, 800)
    ms2 = np.linspace(mmin, mmax, 800)
    mm1, mm2 = np.meshgrid(ms1, ms2)
    mass_pdf = mass_model(
        nknots,
        mm1,
        mm2,
        ms1,
        ms2,
        mmin=mmin,
        mmax=mmax,
        basis=basis,
    )
    mp1dfs = np.zeros((mcoefs.shape[0], len(ms1)))
    mp2dfs = np.zeros((mcoefs.shape[0], len(ms2)))
    if rate is None:
        rate = jnp.ones(mcoefs.shape[0])

    def calc_pdf(mcs1, r, pop_frac, beta):
        p_m1m2 = mass_pdf(mcs1, beta=beta)
        p_m1m2 = jnp.where(jnp.less(mm1, mmin) | jnp.less(mm2, mmin), 0, p_m1m2)
        p_m1 = jnp.trapz(p_m1m2, ms2, axis=0)
        p_m2 = jnp.trapz(p_m1m2, ms1, axis=1)
        return r * p_m1 * pop_frac / jnp.trapz(p_m1, ms1), r * p_m2 * pop_frac / jnp.trapz(p_m2, ms2)

    calc_pdf = jit(calc_pdf)
    # loop through hyperposterior samples
    for ii in trange(mcoefs.shape[0]):
        mp1dfs[ii], mp2dfs[ii] = calc_pdf(mcoefs[ii], rate[ii], pop_frac[ii][pop_num], model_kwargs["beta"][ii])
    return mp1dfs, mp2dfs, ms1, ms2
