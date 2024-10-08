import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.integrate import trapezoid
from tqdm import trange

from gwinferno.distributions import betadist
from gwinferno.distributions import truncnorm_pdf
from gwinferno.interpolation import LogYBSpline
from gwinferno.models.bsplines.separable import BSplineIIDSpinMagnitudes
from gwinferno.models.bsplines.separable import BSplineIIDSpinTilts
from gwinferno.models.bsplines.separable import BSplineIndependentSpinMagnitudes
from gwinferno.models.bsplines.separable import BSplineIndependentSpinTilts
from gwinferno.models.bsplines.separable import BSplinePrimaryBSplineRatio
from gwinferno.models.bsplines.single import BSplineRatio
from gwinferno.models.parametric.parametric import mixture_isoalign_spin_tilt
from gwinferno.models.parametric.parametric import plpeak_primary_ratio_pdf


def calculate_bspline_mass_ppds(m_cs, q_cs, nspline_dict, mmin, mmax, rate=None, pop_frac=None):

    ms = jnp.linspace(mmin, mmax, 800)
    qs = jnp.linspace(mmin / mmax, 1, 800)
    M, Q = jnp.meshgrid(ms, qs)

    if rate is None:
        rate = jnp.ones(m_cs.shape[0])
    if pop_frac is None:
        pop_frac = jnp.ones(m_cs.shape[0])

    model = BSplinePrimaryBSplineRatio(
        nspline_dict["m1"],
        nspline_dict["q"],
        M,
        ms,
        Q,
        qs,
        m1min=mmin,
        m2min=mmin,
        mmax=mmax,
    )

    mpdfs = np.zeros((m_cs.shape[0], len(ms)))
    qpdfs = np.zeros((q_cs.shape[0], len(qs)))

    def calc_pdf(m_cs, q_cs, r, frac):
        p_MQ = model(m_cs, q_cs, pe_samples=True)
        p_mq = jnp.where(jnp.greater(Q, mmin / M), p_MQ, 0.0)
        p_m = trapezoid(p_mq, qs, axis=0)
        p_q = trapezoid(p_mq, ms, axis=1)
        P_m = r * p_m * frac / trapezoid(p_m, ms)
        P_q = r * p_q * frac / trapezoid(p_q, qs)
        return P_m, P_q

    calc_pdf = jit(calc_pdf)

    for i in trange(mpdfs.shape[0]):
        mpdfs[i], qpdfs[i] = calc_pdf(m_cs[i], q_cs[i], rate[i], pop_frac[i])

    return mpdfs, ms, qpdfs, qs


def calculate_powerlaw_peak_mass_ppds(alpha, beta, mu_peak, sig_peak, lamb, mmin, mmax, rate=None, pop_frac=None):

    ms = jnp.linspace(mmin, mmax, 800)
    qs = jnp.linspace(mmin / mmax, 1, 800)
    M, Q = jnp.meshgrid(ms, qs)

    if rate is None:
        rate = jnp.ones(alpha.shape[0])
    if pop_frac is None:
        pop_frac = jnp.ones(alpha.shape[0])

    mpdfs = np.zeros((alpha.shape[0], len(ms)))
    qpdfs = np.zeros((alpha.shape[0], len(qs)))

    def calc_pdf(a, b, mp, sigp, lam, r, frac):
        p_MQ = plpeak_primary_ratio_pdf(M, Q, a, b, mmin, mmax, mp, sigp, lam)
        p_mq = jnp.where(jnp.greater(Q, mmin / M), p_MQ, 0.0)
        p_m = trapezoid(p_mq, qs, axis=0)
        p_q = trapezoid(p_mq, ms, axis=1)
        P_m = r * p_m * frac / trapezoid(p_m, ms)
        P_q = r * p_q * frac / trapezoid(p_q, qs)
        return P_m, P_q

    calc_pdf = jit(calc_pdf)

    for i in trange(mpdfs.shape[0]):
        mpdfs[i], qpdfs[i] = calc_pdf(alpha[i], beta[i], mu_peak[i], sig_peak[i], lamb[i], rate[i], pop_frac[i])

    return mpdfs, ms, qpdfs, qs


def calculate_peak_logm1_bspline_q_ppds(logmp, logsigp, q_cs, nspline_dict, mmin, mmax, rate=None, pop_frac=None):

    ms = jnp.linspace(mmin, mmax, 800)
    qs = jnp.linspace(mmin / mmax, 1, 800)
    M, Q = jnp.meshgrid(ms, qs)

    if rate is None:
        rate = jnp.ones(q_cs.shape[0])
    if pop_frac is None:
        pop_frac = jnp.ones(q_cs.shape[0])

    q_model = BSplineRatio(
        nspline_dict["q"],
        Q,
        qs,
        mmin / mmax,
        basis=LogYBSpline,
    )

    mpdfs = np.zeros((q_cs.shape[0], len(ms)))
    qpdfs = np.zeros((q_cs.shape[0], len(qs)))

    def calc_pdf(logmp, logsigp, q_cs, r, frac):
        p_mq = q_model(q_cs, pe_samples=True) * truncnorm_pdf(M, logmp, logsigp, mmin, mmax, log=True)
        p_mq = jnp.where(jnp.less(M, mmin) | jnp.less(M * Q, mmin), 0, p_mq)
        p_m = trapezoid(p_mq, qs, axis=0)
        p_q = trapezoid(p_mq, ms, axis=1)
        P_m = r * p_m * frac / trapezoid(p_m, ms)
        P_q = r * p_q * frac / trapezoid(p_q, qs)
        return P_m, P_q

    calc_pdf = jit(calc_pdf)

    for i in trange(mpdfs.shape[0]):
        mpdfs[i], qpdfs[i] = calc_pdf(logmp[i], logsigp[i], q_cs[i], rate[i], pop_frac[i])

    return mpdfs, ms, qpdfs, qs


def calculate_beta_spin_mag(alpha_a, beta_a, amax=1, rate=None, pop_frac=None):

    aa = jnp.linspace(0, amax, 800)

    if rate is None:
        rate = jnp.ones(alpha_a.shape[0])
    if pop_frac is None:
        pop_frac = jnp.ones(alpha_a.shape[0])

    apdfs = np.zeros((alpha_a.shape[0], len(aa)))

    def calc_pdf(a_a1, b_a1, r, f):
        p_a = betadist(aa, a_a1, b_a1, scale=amax)
        P_a = r * f * p_a / trapezoid(p_a, aa)
        return P_a

    calc_pdf = jit(calc_pdf)

    for i in trange(alpha_a.shape[0]):
        apdfs[i] = calc_pdf(alpha_a[i], beta_a[i], rate[i], pop_frac[i])

    return apdfs, aa


def calculate_mixture_iso_aligned_spin_tilt(sig_ct, lambda_ct, rate=None, pop_frac=None):

    ct = jnp.linspace(-1, 1, 800)

    if rate is None:
        rate = jnp.ones(sig_ct.shape[0])
    if pop_frac is None:
        pop_frac = jnp.ones(sig_ct.shape[0])

    ctpdfs = np.zeros((sig_ct.shape[0], len(ct)))

    def calc_pdf(s_ct, l_ct, r, f):
        p_ct = mixture_isoalign_spin_tilt(ct, l_ct, s_ct)
        P_ct = r * f * p_ct / trapezoid(p_ct, ct)
        return P_ct

    calc_pdf = jit(calc_pdf)

    for i in trange(sig_ct.shape[0]):
        ctpdfs[i] = calc_pdf(sig_ct[i], lambda_ct[i], rate[i], pop_frac[i])

    return ctpdfs, ct


def calculate_bspline_spin_ppds(a1_cs, tilt1_cs, nspline_dict, a2_cs=None, tilt2_cs=None, rate=None, pop_frac=None):

    aa = jnp.linspace(0, 1, 800)
    cc = jnp.linspace(-1, 1, 800)

    if rate is None:
        rate = jnp.ones(a1_cs.shape[0])
    if pop_frac is None:
        pop_frac = jnp.ones(a1_cs.shape[0])

    if a2_cs is None:

        mag_model = BSplineIIDSpinMagnitudes(nspline_dict["a"], aa, aa, aa, aa, basis=LogYBSpline, normalize=True)

        tilt_model = BSplineIIDSpinTilts(nspline_dict["tilt"], cc, cc, cc, cc, basis=LogYBSpline, normalize=True)

        apdfs = np.zeros((a1_cs.shape[0], len(aa)))
        ctpdfs = np.zeros((tilt1_cs.shape[0], len(cc)))

        def calc_pdf(a_cs, ct_cs, r, f):
            p_a = mag_model.primary_model(a_cs)
            p_ct = tilt_model.primary_model(ct_cs)
            P_a = r * f * p_a / trapezoid(p_a, aa)
            P_ct = r * f * p_ct / trapezoid(p_ct, cc)
            return P_a, P_ct

        calc_pdf = jit(calc_pdf)

        for i in trange(apdfs.shape[0]):
            apdfs[i], ctpdfs[i] = calc_pdf(a1_cs[i], tilt1_cs[i], rate[i], pop_frac[i])

        return apdfs, aa, ctpdfs, cc

    else:
        mag_model = BSplineIndependentSpinMagnitudes(nspline_dict["a1"], nspline_dict["a2"], aa, aa, aa, aa, normalize=True)

        tilt_model = BSplineIndependentSpinTilts(nspline_dict["tilt1"], nspline_dict["tilt2"], cc, cc, cc, cc, normalize=True)

        apdfs_1 = np.zeros((a1_cs.shape[0], len(aa)))
        ctpdfs_1 = np.zeros((tilt1_cs.shape[0], len(cc)))
        apdfs_2 = np.zeros((a2_cs.shape[0], len(aa)))
        ctpdfs_2 = np.zeros((tilt2_cs.shape[0], len(cc)))

        def calc_pdf(a_cs_1, ct_cs_1, a_cs_2, ct_cs_2, r, f):
            p_a1 = mag_model.primary_model(a_cs_1)
            p_ct1 = tilt_model.primary_model(ct_cs_1)
            p_a2 = mag_model.secondary_model(a_cs_2)
            p_ct2 = tilt_model.secondary_model(ct_cs_2)

            P_a1 = r * f * p_a1 / trapezoid(p_a1, aa)
            P_ct1 = r * f * p_ct1 / trapezoid(p_ct1, cc)
            P_a2 = r * f * p_a2 / trapezoid(p_a2, aa)
            P_ct2 = r * f * p_ct2 / trapezoid(p_ct2, cc)
            return P_a1, P_ct1, P_a2, P_ct2

        calc_pdf = jit(calc_pdf)

        for i in trange(apdfs_1.shape[0]):
            apdfs_1[i], ctpdfs_1[i], apdfs_2[i], ctpdfs_2[i] = calc_pdf(a1_cs[i], tilt1_cs[i], a2_cs[i], tilt2_cs[i], rate[i], pop_frac[i])

        return apdfs_1, apdfs_2, aa, ctpdfs_1, ctpdfs_2, cc


def calculate_powerlaw_rate_of_z_ppds(lamb, rate, z_model, pop_frac=None):

    if pop_frac is None:
        pop_frac = jnp.ones(lamb.shape[0])

    zs = z_model.zs
    rs = np.zeros((len(lamb), len(zs)))

    def calc_rz(la, r, f):
        return r * f * jnp.power(1.0 + zs, la)

    calc_rz = jit(calc_rz)
    for ii in trange(lamb.shape[0]):
        rs[ii] = calc_rz(lamb[ii], rate[ii], pop_frac[ii])
    return rs, zs


def calculate_powerlaw_spline_rate_of_z_ppds(lamb, z_cs, rate, z_model, pop_frac=None):

    if pop_frac is None:
        pop_frac = jnp.ones(z_cs.shape[0])

    zs = z_model.zs
    rs = np.zeros((len(lamb), len(zs)))

    def calc_rz(cs, la, r, f):
        cs = jnp.concatenate([jnp.array([0]), cs])
        return r * f * jnp.power(1.0 + zs, la) * jnp.exp(z_model.interpolator.project(z_model.norm_design_matrix, cs))

    calc_rz = jit(calc_rz)
    for ii in trange(lamb.shape[0]):
        rs[ii] = calc_rz(z_cs[ii], lamb[ii], rate[ii], pop_frac[ii])
    return rs, zs
