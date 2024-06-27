"""
a module that stores 2D seperable (i.e. independent) population models constructed from bsplines
"""

import jax.numpy as jnp

from ...distributions import powerlaw_pdf
from ...interpolation import BSpline
from ...models.gwpopulation.gwpopulation import plpeak_primary_pdf
from .single import BSplineChiEffective
from .single import BSplineChiPrecess
from .single import BSplineMass
from .single import BSplineRatio
from .single import BSplineSpinMagnitude
from .single import BSplineSpinTilt


class BSplineIIDSpinMagnitudes(object):
    """Class to construct a spin magnitude B-Spline model for both binary components, assuming they
        are independently and identically distributed (IID).

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        a1 (array_like): primary component spin magntiude pe samples to evaluate the basis spline at
        a2 (array_like): secondary component spin magntiude pe samples to evaluate the basis spline at
        a1_inj (array_like): primary component spin magnitude injection samples to evalute the basis spline at
        a2_inj (array_like): secondary component spin magnitude injection samples to evalute the basis spline at
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        a1,
        a2,
        a1_inj,
        a2_inj,
        knots=None,
        degree=3,
        **kwargs,
    ):
        self.primary_model = BSplineSpinMagnitude(
            n_splines=n_splines,
            a=a1,
            a_inj=a1_inj,
            knots=knots,
            degree=degree,
            **kwargs,
        )
        self.secondary_model = BSplineSpinMagnitude(
            n_splines=n_splines,
            a=a2,
            a_inj=a2_inj,
            knots=knots,
            degree=degree,
            **kwargs,
        )

    def __call__(self, coefs, pe_samples=True):
        """will evaluate the joint probability density distribution for the primary and secondary spin
        magnitude along the posterior or injection samples. Use flag `pe_samples` to specify which type
        of samples are being evaluated (pe or injection).

        Args:
            coefs (array_like): basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary and secondary spin magnitude, p(a1, a2)
        """
        p_a1 = self.primary_model(coefs, pe_samples=pe_samples)
        p_a2 = self.secondary_model(coefs, pe_samples=pe_samples)
        return p_a1 * p_a2


class BSplineIndependentSpinMagnitudes(object):
    """Class to construct a spin magnitude B-Spline model for both binary components, assuming they are indipendently distributed.

    Args:
        n_splines1 (int): number of degrees of freedom of basis, i.e. number of basis splines, for the primary binary component
        n_splines2 (int): number of degrees of freedom of basis, i.e. number of basis splines, for the secondary binary component
        a1 (array_like): primary component spin magntiude pe samples to evaluate the basis spline at
        a2 (array_like): secondary component spin magntiude pe samples to evaluate the basis spline at
        a1_inj (array_like): primary component spin magnitude injection samples to evalute the basis spline at
        a2_inj (array_like): secondary component spin magnitude injection samples to evalute the basis spline at
        knots1 (array_like, optional): array of knots for the primary component, if non-uniform knot placing is preferred.
                Defaults to None.
        degree1 (int, optional): degree of primary component spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
        knots2 (array_like, optional): array of knots for the secondary component, if non-uniform knot placing is preferred.
                Defaults to None.
        degree2 (int, optional): degree of secondary component spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines1,
        n_splines2,
        a1,
        a2,
        a1_inj,
        a2_inj,
        knots1=None,
        degree1=3,
        knots2=None,
        degree2=3,
        **kwargs,
    ):
        self.primary_model = BSplineSpinMagnitude(
            n_splines=n_splines1,
            a=a1,
            a_inj=a1_inj,
            knots=knots1,
            degree=degree1,
            **kwargs,
        )
        self.secondary_model = BSplineSpinMagnitude(
            n_splines=n_splines2,
            a=a2,
            a_inj=a2_inj,
            knots=knots2,
            degree=degree2,
            **kwargs,
        )

    def __call__(self, pcoefs, scoefs, pe_samples=True):
        """will evaluate the joint probability density distribution for the primary and secondary spin
        magnitude along the posterior or injection samples. Use flag `pe_samples` to specify which type
        of samples are being evaluated (pe or injection).

        Args:
            pcoefs (array_like): primary component spin magnitude basis spline coefficients
            scoefs (array_like): secondary component spin magnitude basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary and secondary spin magnitude, p(a1, a2)
        """
        p_a1 = self.primary_model(pcoefs, pe_samples=pe_samples)
        p_a2 = self.secondary_model(scoefs, pe_samples=pe_samples)
        return p_a1 * p_a2


class BSplineIIDSpinTilts(object):
    """
    Class to construct a cosine tilt (cos(theta)) B-Spline model for both binary components,
    assuming they are indipendently and identically distributed (IID).

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis splines
        ct1 (array_like): primary component cosine tilt pe samples to evaluate the basis spline at
        ct2 (array_like): secondary component cosine tilt pe samples to evaluate the basis spline at
        ct1_inj (array_like): primary component cosine tilt injection samples to evalute the basis spline at
        ct2_inj (array_like): secondary component cosine tilt injection samples to evalute the basis spline at
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        ct1,
        ct2,
        ct1_inj,
        ct2_inj,
        knots=None,
        degree=3,
        **kwargs,
    ):
        self.primary_model = BSplineSpinTilt(
            n_splines=n_splines,
            ct=ct1,
            ct_inj=ct1_inj,
            knots=knots,
            degree=degree,
            **kwargs,
        )
        self.secondary_model = BSplineSpinTilt(
            n_splines=n_splines,
            ct=ct2,
            ct_inj=ct2_inj,
            knots=knots,
            degree=degree,
            **kwargs,
        )

    def __call__(self, coefs, pe_samples=True):
        """will evalute the joint probability density distribution for the primary and secondary cosine tilt along the posterior or injection samples.
        Use flag `pe_samples` to specify which type of samples are being evaluated (pe or injection).

        Args:
            coefs (array_like): basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary and secondary spin magnitude, p(ct1, ct2)
        """
        p_ct1 = self.primary_model(coefs, pe_samples=pe_samples)
        p_ct2 = self.secondary_model(coefs, pe_samples=pe_samples)
        return p_ct1 * p_ct2


class BSplineIndependentSpinTilts(object):
    """Class to construct a cosine tilt (cos(theta)) B-Spline model for both binary components, assuming they are indipendently distributed.

    Args:
        n_splines1 (int): number of degrees of freedom of basis, i.e. number of basis splines, for the primary binary component
        n_splines2 (int): number of degrees of freedom of basis, i.e. number of basis splines, for the secondary binary component
        ct1 (array_like): primary component cosine tilt pe samples to evaluate the basis spline at
        ct2 (array_like): secondary component cosine tilt pe samples to evaluate the basis spline at
        ct1_inj (array_like): primary component cosine tilt injection samples to evalute the basis spline at
        ct2_inj (array_like): secondary component cosine tilt injection samples to evalute the basis spline at
        knots1 (array_like, optional): array of knots for the primary binary component, if non-uniform knot placing is preferred.
                Defaults to None.
        degree1 (int, optional): degree of the spline for the primary binary component, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
        knots2 (array_like, optional): array of knots for the secondary binary component, if non-uniform knot placing is preferred.
                Defaults to None.
        degree2 (int, optional): degree of the spline for the secondary binary component, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines1,
        n_splines2,
        ct1,
        ct2,
        ct1_inj,
        ct2_inj,
        knots1=None,
        degree1=3,
        knots2=None,
        degree2=3,
        **kwargs,
    ):
        self.primary_model = BSplineSpinTilt(
            n_splines=n_splines1,
            ct=ct1,
            ct_inj=ct1_inj,
            knots=knots1,
            degree=degree1,
            **kwargs,
        )
        self.secondary_model = BSplineSpinTilt(
            n_splines=n_splines2,
            ct=ct2,
            ct_inj=ct2_inj,
            knots=knots2,
            degree=degree2,
            **kwargs,
        )

    def __call__(self, pcoefs, scoefs, pe_samples=True):
        """will evalute the joint probability density distribution for the primary and secondary cosine tilt along the posterior or injection samples.
        Use flag `pe_samples` to specify which type of samples are being evaluated (pe or injection).

        Args:
            pcoefs (array_like): primary component cosine tilt basis spline coefficients
            scoefs (array_like): secondary component cosine tilt basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary and secondary cosine tilt, p(ct1, ct2)
        """
        p_ct1 = self.primary_model(pcoefs, pe_samples=pe_samples)
        p_ct2 = self.secondary_model(scoefs, pe_samples=pe_samples)
        return p_ct1 * p_ct2


class BSplinePrimaryPowerlawRatio(object):
    """Class to construct a B-Spline model in primary mass and a powerlaw model in mass ratio

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis splines
        m1 (array_like): primary component mass pe samples to evaluate the basis spline at
        m1_inj(array_like): primary component mass injection samples to evaluate the basis spline at
        mmin (float, optional): minimum mass value. Spline is truncated below this minimum mass. Defaults to 2.
        mmax (float, optional): maximum mass value. Spline is truncated above this maximum mass. Defaults to 100.
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
        basis (class, optional): type of basis to use (ex. LogYBSpline). Defaults to Bspline.
    """

    def __init__(
        self,
        n_splines,
        m1,
        m1_inj,
        mmin=2,
        mmax=100,
        knots=None,
        degree=3,
        basis=BSpline,
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            n_splines,
            m1,
            m1_inj,
            knots=knots,
            mmin=mmin,
            mmax=mmax,
            degree=degree,
            basis=basis,
            **kwargs,
        )

    def __call__(self, m1, q, beta, mmin, coefs, pe_samples=True):
        """will evalute the joint probability density distribution for the primary mass and mass ratio along the posterior or injection samples.
        Use flag `pe_samples` to specify which type of samples are being evaluated (pe or injection).

        Args:
            m1 (array_like): primary mass samples to evaluate pdf at
            q (array_like): mass ratio samples to evaluate pdf at
            mmin (float): minimum mass. Pdf will be truncated below this value.
            coefs (array_like): primary component mass basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary mass and mass ratio, p(m1, q)
        """
        # p_m1 = jnp.where(jnp.greater_equal(m1, mmin), self.primary_model(len(m1.shape), coefs), 0)
        # norm_factor = self.primary_model.norm_mmin_cut(mmin, coefs)
        # p_m1 = p_m1 / norm_factor
        p_m1 = self.primary_model(coefs, pe_samples=pe_samples)
        p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
        return p_m1 * p_q


class PLPeakPrimaryBSplineRatio(object):
    """Class to construct a powerlaw + gaussian peak primary mass model and B-Spline mass ratio model.

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis splines
        q (array_like): mass ratio pe samples to evaluate the basis spline at
        q_inj(array_like): mass ratio injection samples to evaluate the basis spline at
        m1 (float, optional): primary mass pe samples
        m1_inj (float, optional): primary mass injection samples
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        q,
        q_inj,
        knots=None,
        degree=3,
    ):
        self.ratio_model = BSplineRatio(
            n_splines,
            q,
            q_inj,
            knots=knots,
            degree=degree,
        )

    def __call__(self, m1, coefs, pe_samples=True, **kwargs):
        """will evalute the joint probability density distribution for the primary mass and mass ratio along the posterior or injection samples.
        Use flag `pe_samples` to specify which type of samples are being evaluated (pe or injection).

        Args:
            m1 (array_like): primary mass samples to evaluate pdf at
            q (array_like): mass ratio samples to evaluate pdf at
            mmin (float): minimum mass. Pdf will be truncated below this value.
            coefs (array_like): primary component mass basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary mass and mass ratio, p(m1, q)
        """
        p_q = self.ratio_model(coefs, pe_samples=pe_samples)
        p_m1 = plpeak_primary_pdf(m1, **kwargs)
        return p_m1 * p_q


class BSplinePrimaryBSplineRatio(object):
    """Class to construct a B-Spline model in primary mass and a B-Spline model in mass ratio

    Args:
        n_splines_m (int): number of degrees of freedom of basis, i.e. number of basis splines, for primary mass model
        n_splines_q (int): number of degrees of freedom of basis, i.e. number of basis splines, for mass ratio model
        m1 (array_like): primary component mass pe samples to evaluate the basis spline at
        m1_inj(array_like): primary component mass injection samples to evaluate the basis spline at
        q1 (array_like): mass ratio pe samples to evaluate the basis spline at
        q1_inj(array_like): mass ratio injection samples to evaluate the basis spline at
        knots_m (array_like, optional): primary mass array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        knots_q (array_like, optional): mass ratio array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree_m (int, optional): degree of the primary mass spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline)
        degree_q (int, optional): degree of the mass ratio spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline)
        m1min (float, optional): minimum primary mass value. Primary mass spline is truncated below this minimum mass. Defaults to 3.
        m2min (float, optional): minimum secondary mass value. Mass ratio spline is truncated below m2min/mmax. Defaults to 3.
        mmax (float, optional): maximum mass value. Primary mass spline is truncated above this maximum mass. Defaults to 100.
        basis_m (class, optional): type of basis to use (ex. LogYBSpline) for primary mass spline. Defaults to Bspline.
        basis_q (class, optional): type of basis to use (ex. LogYBSpline) for mass ratio spline. Defaults to Bspline.
    """

    def __init__(
        self,
        n_splines_m,
        n_splines_q,
        m1,
        m1_inj,
        q,
        q_inj,
        knots_m=None,
        knots_q=None,
        degree_m=3,
        degree_q=3,
        m1min=3.0,
        m2min=3.0,
        mmax=100.0,
        basis_m=BSpline,
        basis_q=BSpline,
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            n_splines_m,
            m1,
            m1_inj,
            knots=knots_m,
            mmin=m1min,
            mmax=mmax,
            degree=degree_m,
            basis=basis_m,
            **kwargs,
        )
        self.ratio_model = BSplineRatio(
            n_splines_q,
            q,
            q_inj,
            qmin=m2min / mmax,
            knots=knots_q,
            degree=degree_q,
            basis=basis_q,
            **kwargs,
        )

    def __call__(self, mcoefs, qcoefs, pe_samples=True):
        """will evalute the joint probability density distribution for the primary mass and mass ratio along the posterior or injection samples.
        Use flag `pe_samples` to specify which type of samples are being evaluated (pe or injection).

        Args:
            mcoefs (array_like): primary component mass basis spline coefficients
            qcoefs (array_like): mass ratio basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary mass and mass ratio, p(m1, q)
        """
        return self.ratio_model(qcoefs, pe_samples=pe_samples) * self.primary_model(mcoefs, pe_samples=pe_samples)


class BSplineIIDComponentMasses(object):
    """
    Class to construct a B-Spline model in primary mass and secondary mass,
    assuming the two binary mass components are independently and identically distributed (IID).

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis splines, for primary mass model
        m1 (array_like): primary component mass pe samples to evaluate the basis spline at
        m2 (array_like): secondary component mass pe samples to evaluate the basis spline at
        m1_inj(array_like): primary component mass injection samples to evaluate the basis spline at
        m2_inj(array_like): secondary component mass injection samples to evaluate the basis spline at
        knots(array_like, optional): array of knots, if non-uniform knot placing is preferred. Defaults to None.
        mmin (float, optional): minimum mass value. Splines are truncated below this minimum mass. Defaults to 2.
        mmax (float, optional): maximum mass value. Splines are truncated above this maximum mass. Defaults to 100.
    """

    def __init__(
        self,
        n_splines,
        m1,
        m2,
        m1_inj,
        m2_inj,
        knots=None,
        mmin=2,
        mmax=100,
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            n_splines=n_splines,
            m=m1,
            m_inj=m1_inj,
            knots=knots,
            mmin=mmin,
            mmax=mmax,
            **kwargs,
        )
        self.secondary_model = BSplineMass(
            n_splines=n_splines,
            m=m2,
            m_inj=m2_inj,
            knots=knots,
            mmin=mmin,
            mmax=mmax,
            **kwargs,
        )
        self.qs = [m2_inj / m1_inj, m2 / m1]

    def __call__(self, coefs, beta=0, pe_samples=True):
        """will evalute the joint probability density distribution for the primary and secondary masses along the posterior or injection samples.
        Use flag `pe_samples` to specify which type of samples are being evaluated (pe or injection).

        Args:
            coefs (array_like): mass basis spline coefficients
            beta (float, optional): mass ratio powerlaw slope. Defaults to 0.
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary and secondary mass and mass ratio, p(m1, m2, q)
        """
        p_m1 = self.primary_model(coefs, pe_samples=pe_samples)
        p_m2 = self.secondary_model(coefs, pe_samples=pe_samples)
        dim = 1 if pe_samples else 0
        return jnp.where(
            jnp.less(self.qs[dim], 0) | jnp.greater(self.qs[dim], 1),
            0,
            p_m1 * p_m2,
        ) * jnp.power(self.qs[dim], beta)


class BSplineIndependentComponentMasses(object):
    """Class to construct a B-Spline model in primary mass and secondary mass, assuming the two binary mass components are idependently distributed.

    Args:
        n_splines1 (int): number of degrees of freedom of primary mass basis, i.e. number of basis splines
        n_splines2 (int): number of degrees of freedom of secondary mass basis, i.e. number of basis splines
        m1 (array_like): primary component mass pe samples to evaluate the basis spline at
        m2 (array_like): secondary component mass pe samples to evaluate the basis spline at
        m1_inj(array_like): primary component mass injection samples to evaluate the basis spline at
        m2_inj(array_like): secondary component mass injection samples to evaluate the basis spline at
        knots1 (array_like, optional): array of primary mass knots, if non-uniform knot placing is preferred. Defaults to None.
        knots2 (array_like, optional): array of secondary mass knots, if non-uniform knot placing is preferred. Defaults to None.
        mmin1 (float, optional): minimum primary mass value. Spline is truncated below this minimum mass. Defaults to 2.
        mmax1 (float, optional): maximum primary mass value. Spline is truncated above this maximum mass. Defaults to 100.
        mmin2 (float, optional): minimum secondary mass value. Spline is truncated below this minimum mass. Defaults to 2.
        mmax2 (float, optional): maximum secondary mass value. Spline is truncated above this maximum mass. Defaults to 100.
        degree1 (int, optional): degree of the spline for the primary mass, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
        degree2 (int, optional): degree of the spline for the secondary mass, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines1,
        n_splines2,
        m1,
        m2,
        m1_inj,
        m2_inj,
        knots1=None,
        knots2=None,
        mmin1=2,
        mmax1=100,
        mmin2=2,
        mmax2=100,
        degree1=3,
        degree2=3,
    ):
        self.primary_model = BSplineMass(
            n_splines=n_splines1,
            m=m1,
            m_inj=m1_inj,
            knots=knots1,
            mmin=mmin1,
            mmax=mmax1,
            degree=degree1,
        )
        self.secondary_model = BSplineMass(
            n_splines=n_splines2,
            m=m2,
            m_inj=m2_inj,
            knots=knots2,
            mmin=mmin2,
            mmax=mmax2,
            degree=degree2,
        )

        self.qs = [m2_inj / m1_inj, m2 / m1]

    def __call__(self, pcoefs, scoefs, beta, pe_samples=True):
        """will evalute the joint probability density distribution for the primary and secondary masses along the posterior or injection samples.
        Use flag `pe_samples` to specify which type of samples are being evaluated (pe or injection).

        Args:
            m1 (array_like): primary component mass pe samples to evaluate the basis spline at
            m2 (array_like): secondary component mass pe samples to evaluate the basis spline at
            pcoefs (array_like): primary mass basis spline coefficients
            scoefs (array_like): secondary mass basis spline coefficients
            beta (float, optional): mass ratio powerlaw slope.
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary and secondary mass and mass ratio, p(m1, m2, q)
        """
        p_m1 = self.primary_model(pcoefs, pe_samples=pe_samples)
        p_m2 = self.secondary_model(scoefs, pe_samples=pe_samples)
        dim = 1 if pe_samples else 0
        return p_m1 * p_m2 * self.qs[dim] ** beta


class BSplineEffectiveSpinDims(object):
    """Class to construct a B-Spline model in the chi effective and effective precession (chi_p) parameters

    Args:
        n_splines1 (int): number of degrees of freedom of primary mass basis, i.e. number of basis splines
        n_splines2 (int): number of degrees of freedom of secondary mass basis, i.e. number of basis splines
        chieff(array_like): chi effective pe samples to evalute the basis spline at
        chip (array_like): chi_p pe samples to evalute the basis spline at
        chieff_inj (array_like): chi effective injection samples to evalute the basis spline at
        chip_inj (array_like): chi_p injection samples to evalute the basis spline at
        knotse (array_like, optional): array of chi effective knots, if non-uniform knot placing is preferred. Defaults to None.
        knotsp (array_like, optional): array of chi_p knots, if non-uniform knot placing is preferred. Defaults to None.
        degree_e (int, optional): degree of the spline for chi effective, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
        degree_p (int, optional): degree of the spline for chi_p, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines_e,
        n_splines_p,
        chieff,
        chip,
        chieff_inj,
        chip_inj,
        knotse=None,
        knotsp=None,
        degree_e=3,
        degree_p=3,
    ):
        self.chi_eff_model = BSplineChiEffective(
            n_splines_e,
            chieff,
            chieff_inj,
            knots=knotse,
            degree=degree_e,
        )

        self.chi_p_model = BSplineChiPrecess(
            n_splines_p,
            chip,
            chip_inj,
            knots=knotsp,
            degree=degree_p,
        )

    def __call__(self, ecoefs, pcoefs, pe_samples=True):
        """will evalute the joint probability density distribution for chi effective and chi_p along the posterior or injection samples.
        Use flag `pe_samples` to specify which type of samples are being evaluated (pe or injection).

        Args:
            m1 (array_like): primary component mass pe samples to evaluate the basis spline at
            m2 (array_like): secondary component mass pe samples to evaluate the basis spline at
            ecoefs (array_like): chi_eff basis spline coefficients
            pcoefs (array_like): chi_p basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: the joint probability density distribution for the primary and secondary mass and mass ratio, p(m1, m2, q)
        """
        p_chieff = self.chi_eff_model(ecoefs, pe_samples=pe_samples)
        p_chip = self.chi_p_model(pcoefs, pe_samples=pe_samples)
        return p_chieff * p_chip
