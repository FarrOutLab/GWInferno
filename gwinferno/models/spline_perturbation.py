"""
a module that stores spline perturbation related population models
"""

import jax.numpy as jnp
import numpy as np

from ..distributions import powerlaw_pdf
from ..interpolation import BSpline
from ..interpolation import LogXBSpline
from .gwpopulation.gwpopulation import PowerlawRedshiftModel


class PowerlawBasisSplinePrimaryPowerlawRatio(object):
    def __init__(
        self,
        nknots: int,
        m1pe: dict,
        m1inj: dict,
        mmin: float = 3.0,
        m2min: float = 3.0,
        mmax: float = 100.0,
        k: int = 4,
        basis: BSpline = BSpline,
        **kwargs
    ):
        """
        __init__

        Args:
            nknots (int): Number of knots used to create the B-Splines.
            m1pe (dict): Dictionary with m1's parameter estimation.
            m1inj (dict): Dictionary with m1's injection samples.
            mmin (float, optional): Minimum primary mass distribution cutoff. Defaults to 3.
            m2min (float, optional): Minimum secondary mass. Defaults to 3.
            mmax (float, optional): Maximum primary mass distribution cutoff. Defaults to 100.
            k (int, optional): Power of the polynomials used in the B-Spline. Defaults to 4.
            basis (object, optional): The type of basis class you wish to use. Defaults to BSpline.
        """
        self.m2min = m2min
        self.nknots = nknots
        self.mmin = mmin
        self.mmax = mmax
        self.ms = jnp.linspace(mmin, mmax, 1000)
        self.nknots = nknots
        interior_knots = np.linspace(np.log(mmin), np.log(mmax), nknots - k + 2)
        dx = interior_knots[1] - interior_knots[0]
        knots = np.concatenate(
            [
                np.log(mmin) - dx * np.arange(1, k)[::-1],
                interior_knots,
                np.log(mmax) + dx * np.arange(1, k),
            ]
        )
        self.knots = knots
        self.interpolator = basis(nknots, knots=knots, interior_knots=interior_knots, xrange=(np.log(mmin), np.log(mmax)), k=4, **kwargs)
        self.pe_design_matrix = jnp.array(self.interpolator.bases(np.log(m1pe)))
        self.inj_design_matrix = jnp.array(self.interpolator.bases(np.log(m1inj)))
        self.dmats = [self.inj_design_matrix, self.pe_design_matrix]
        self.norm_design_matrix = jnp.array(self.interpolator.bases(np.log(self.ms)))

    def smoothing(self, ms: jnp.ndarray, mmin: float, delta_m: float):
        """
        smoothing

        Args:
            ms (jnp.ndarray): Black hole masses
            mmin (float): minimum black hole mass
            delta_m (float): size of BH grid

        Returns:
            _type_:
        """
        sm = ms - mmin
        smoothing_region = jnp.greater(sm, 0) & jnp.less(sm, delta_m)
        window = jnp.where(
            smoothing_region,
            1.0 / (jnp.exp(delta_m / sm + delta_m / (sm - delta_m)) + 1.0),
            1,
        )
        window = jnp.where(jnp.isinf(window) | jnp.isnan(window), 1, window)
        return jnp.where(jnp.less_equal(ms, mmin), 0, window)

    def p_m1(self, m1: jnp.ndarray, alpha: float, mmin: float, mmax: float, cs: jnp.ndarray):
        """
        p_m1 Probability distribution of primary masses

        Args:
            m1 (jnp.ndarray): Ndarray of primary (m1) masses
            alpha (float): Power-law index
            mmin (float): Minimum primary mass cutoff
            mmax (float): Maximum primary mass cutoff
            cs (jnp.ndarray): B-spline coefficients

        Returns:
            _type_: Probability of primary mass
        """
        p_m = powerlaw_pdf(m1, alpha=-alpha, low=mmin, high=mmax)
        ndim = len(m1.shape)
        perturbation = jnp.exp(self.interpolator.project(self.dmats[ndim - 1], cs))
        norm = self.norm_p_m1(alpha=alpha, mmin=mmin, mmax=mmax, cs=cs)
        return p_m * perturbation / norm

    def norm_p_m1(self, alpha: float, mmin: float, mmax: float, cs: jnp.ndarray):
        """
        norm_p_m1 Normalized probability distribution of primary mass

        Args:
            alpha (float): Power of the powerlaw
            mmin (float): Minimum primary mass cutoff
            mmax (float): Maximum primary mass cutoff
            cs (jnp.ndarray): B-spline coefficients

        Returns:
            _type_: Normalized probability of primary mass
        """
        p_m = powerlaw_pdf(self.ms, alpha=-alpha, low=mmin, high=mmax)
        perturbation = jnp.exp(self.interpolator.project(self.norm_design_matrix, cs))
        return jnp.trapz(y=p_m * perturbation, x=self.ms)

    def p_q(self, q: jnp.ndarray, m1: jnp.ndarray, beta: float):
        """
        p_q Probability of mass ratio

        Args:
            q (jnp.ndarray): Mass ratio
            m1 (jnp.ndarray): Primary mass
            beta (float): Power-law index

        Returns:
            _type_: Probability of mass ratio
        """
        p_q = powerlaw_pdf(q, alpha=beta, low=self.m2min / m1, high=1)
        return p_q

    def __call__(self, m1: jnp.ndarray, q: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        __call__
        Args:
            m1 (jnp.ndarray): Primary masses
            q (jnp.ndarray): Mass ratio

        Returns:
            jnp.ndarray: _description_
        """
        beta = kwargs.pop("beta")
        p_m1 = self.p_m1(m1, **kwargs)
        p_q = self.p_q(q, m1, beta=beta)
        return p_m1 * p_q


class PowerlawBasisSplinePrimaryRatio(object):
    def __init__(self, nknots: int, qknots: int, m1pe: dict, qpe: dict, m1inj: dict, qinj: dict, mmin: float = 2.0, mmax: float = 100.0, k: int = 4):
        """
        __init__

        Args:
            nknots (int): Number of knots used to create the B-Splines.
            qknots (int): Number of knots used to create the B-Spline for the mass ratio.
            m1pe (dict): Dictionary with m1's parameter estimation.
            qpe (dict): Dictionary with mass ratio parameter estimation.
            m1inj (dict): Dictionary with m1's injection samples.
            qinj (dict): Dictionary with mass ratio injection samples.
            mmin (float, optional): Minimum primary mass cutoff. Defaults to 2.
            mmax (float, optional): Maximum primary mass cutoff. Defaults to 100.
            k (int, optional): Power of the polynomials used in the B-Spline. Defaults to 4.
        """
        self.nknots = nknots
        self.mmin = mmin
        self.mmax = mmax
        self.ms = jnp.linspace(mmin, mmax, 1000)
        self.qs = jnp.linspace(mmin / mmax, 1, 500)
        self.nknots = nknots
        self.qknots = qknots
        self.mm, self.qq = jnp.meshgrid(self.ms, self.qs)
        interior_knots = np.linspace(np.log(mmin), np.log(mmax), nknots - k + 2)
        dx = interior_knots[1] - interior_knots[0]
        knots = np.concatenate(
            [
                np.log(mmin) - dx * np.arange(1, k)[::-1],
                interior_knots,
                np.log(mmax) + dx * np.arange(1, k),
            ]
        )
        self.knots = knots
        interior_qknots = np.linspace(0, 1, qknots - k + 2)
        dxq = interior_qknots[1] - interior_qknots[0]
        knotsq = np.concatenate(
            [
                -dxq * np.arange(1, k)[::-1],
                interior_qknots,
                1 + dxq * np.arange(1, k),
            ]
        )
        self.knotsq = knotsq

        self.interpolator = BSpline(
            nknots,
            knots=knots,
            interior_knots=interior_knots,
            xrange=(np.log(mmin), np.log(mmax)),
            k=4,
        )
        self.pe_design_matrix = jnp.array(self.interpolator.bases(np.log(m1pe)))
        self.inj_design_matrix = jnp.array(self.interpolator.bases(np.log(m1inj)))
        self.dmats = [self.inj_design_matrix, self.pe_design_matrix]
        self.qinterpolator = BSpline(
            qknots,
            knots=knotsq,
            interior_knots=interior_qknots,
            xrange=(0, 1),
            k=4,
        )
        self.qpe_design_matrix = jnp.array(self.qinterpolator.bases(qpe))
        self.qinj_design_matrix = jnp.array(self.qinterpolator.bases(qinj))
        self.qdmats = [self.qinj_design_matrix, self.qpe_design_matrix]
        self.qshapes = [(self.qknots, 1), (self.qknots, 1, 1)]
        self.norm_design_matrix = jnp.array(self.interpolator.bases(np.log(self.mm)))
        self.qnorm_design_matrix = jnp.array(self.qinterpolator.bases(self.qq))

    def p_m1(self, m1: jnp.ndarray, alpha: float, mmin: float, mmax: float, cs: jnp.ndarray):
        """
        p_m1 Probability distribution of primary masses

        Args:
            m1 (jnp.ndarray): Ndarray of primary (m1) masses
            alpha (float): Power-law index
            mmin (float): Minimum primary mass cutoff
            mmax (float): Maximum primary mass cutoff
            cs (jnp.ndarray): B-Spline coefficients

        Returns:
            _type_: Probability of primary mass
        """
        p_m = powerlaw_pdf(m1, alpha=-alpha, low=mmin, high=mmax)
        ndim = len(m1.shape)
        perturbation = jnp.exp(self.interpolator.project(self.dmats[ndim - 1], cs))
        return p_m * perturbation

    def norm_pm1q(self, alpha: float, mmin: float, mmax: float, cs: jnp.ndarray, beta: float, vs: jnp.ndarray):
        """
        norm_pm1q Normalized (primary mass/mass ratio) distribution

        Args:
            alpha (_type_): Power of the power-law
            mmin (_type_): Minimum primary mass cutoff
            mmax (_type_): Maximum primary mass cutoff
            cs (_type_): B-Spline coefficients
            beta (_type_): Mass ratio power-law index
            vs (_type_): B-Spline coefficients for the mass ratio

        Returns:
            _type_: Normalized probability of (primary mass/ mass ratio)
        """
        p_m = powerlaw_pdf(self.mm, alpha=-alpha, low=mmin, high=mmax)
        perturbation = jnp.exp(self.interpolator.project(self.norm_design_matrix, cs))
        p_q = powerlaw_pdf(self.qq, alpha=beta, low=mmin / self.mm, high=1)
        qperturbation = jnp.exp(self.qinterpolator.project(self.qnorm_design_matrix, vs))
        p_mq = p_m * perturbation * p_q * qperturbation
        return jnp.trapz(jnp.trapz(p_mq, self.qs, axis=0), self.ms)

    def p_q(self, q: jnp.ndarray, m1: jnp.ndarray, beta: float, mmin: float, vs: jnp.ndarray):
        """
        p_q Probability of mass ratio

        Args:
            q (jnp.ndarray): Mass ratio
            m1 (jnp.ndarray): Primary mass
            beta (float): Mass ratio power-law index
            mmin (float): Minimum primary mass cutoff
            vs (jnp.ndarray): B-Spline coefficients

        Returns:
            _type_: Probability of mass ratio
        """
        p_q = powerlaw_pdf(q, alpha=beta, low=mmin / m1, high=1)
        ndim = len(q.shape)
        perturbation = jnp.exp(self.qinterpolator.project(self.qdmats[ndim - 1], vs))
        return p_q * perturbation

    def __call__(self, m1: jnp.ndarray, q: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        __call__

        Args:
            m1 (jnp.ndarray): Primary mass
            q (jnp.ndarray): Mass ratio

        Returns:
            jnp.ndarray:
        """
        beta = kwargs.pop("beta")
        mmin = kwargs.pop("mmin", self.mmin)
        vs = kwargs.pop("vs")
        p_m1 = self.p_m1(m1, mmin=mmin, **kwargs)
        p_q = self.p_q(q, m1, beta=beta, mmin=mmin, vs=vs)
        norm = self.norm_pm1q(beta=beta, mmin=mmin, vs=vs, **kwargs)
        return p_m1 * p_q / norm


class PowerlawSplineRedshiftModel(PowerlawRedshiftModel):
    def __init__(self, nknots: int, z_pe: dict, z_inj: dict, basis: LogXBSpline = LogXBSpline):
        """
        __init__

        Args:
            nknots (int): Number of knots used to create B-Spline
            z_pe (dict): Redshift parameter estimation
            z_inj (dict): Redshift injections
            basis (LogXBSpline, optional): Bases to be used in the spline perturbation. Defaults to LogXBSpline.
        """
        super().__init__(z_pe=z_pe, z_inj=z_inj)
        self.nknots = nknots
        self.interpolator = basis(nknots, xrange=(self.zmin, self.zmax), k=4, normalize=False)
        self.pe_design_matrix = jnp.array(self.interpolator.bases(z_pe))
        self.inj_design_matrix = jnp.array(self.interpolator.bases(z_inj))
        self.dmats = [self.inj_design_matrix, self.pe_design_matrix]
        self.norm_design_matrix = jnp.array(self.interpolator.bases(self.zs))

    def normalization(self, lamb: float, cs: jnp.ndarray):
        """
        normalization

        Args:
            lamb (float): Power-law exponent for the redshift model
            cs (jnp.ndarray): B-Spline coefficients

        Returns:
            _type_:
        """
        pz = self.dVdz_ * jnp.power(1.0 + self.zs, lamb - 1)
        pz *= jnp.exp(self.interpolator.project(self.norm_design_matrix, cs))
        return jnp.trapz(pz, self.zs)

    def prob(self, z: jnp.ndarray, dVdz: jnp.ndarray, lamb: float, cs: jnp.ndarray):
        """
        prob Returns probability

        Args:
            z (jnp.ndarray): Redshift
            dV_cdz (jnp.ndarray): Differential co-moving volume element with respect to redshift.
            lamb (float): Power-law exponent for redshift model
            cs (jnp.ndarray): B-Spline coefficients

        Returns:
            _type_:
        """
        ndim = len(z.shape)
        return dVdz * jnp.power(1.0 + z, lamb - 1.0) * jnp.exp(self.interpolator.project(self.dmats[ndim - 1], cs))

    def __call__(self, z: jnp.ndarray, lamb: float, cs: jnp.ndarray) -> jnp.ndarray:
        """
        __call__

        Args:
            z (jnp.ndarray): Redshift
            lamb (float): Power-law exponent for redshift model
            cs (jnp.ndarray): B-Spline coefficients

        Returns:
            jnp.ndarray:
        """
        ndim = len(z.shape)
        dVdz = self.dVdzs[ndim - 1]
        return jnp.where(
            jnp.less_equal(z, self.zmax),
            self.prob(z, dVdz, lamb, cs) / self.normalization(lamb, cs),
            0,
        )
