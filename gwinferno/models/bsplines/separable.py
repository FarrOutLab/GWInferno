"""
A collection of 2-D separable (i.e. independent) population models involving basis splines
"""

import jax.numpy as jnp

from ...distributions import powerlaw_pdf
from ..parametric.parametric import plpeak_primary_pdf
from .single import BSplineChiEffective
from .single import BSplineChiPrecess
from .single import BSplineMass
from .single import BSplineRatio
from .single import BSplineSpinMagnitude
from .single import BSplineSpinTilt


class BSplineIIDSpinMagnitudes(object):
    r"""A B-Spline model for the spin magnitude of both binary components assuming
    they are independently and identically distributed (IID),

    .. math::
        p(a_1, a_2 \mid \mathbf{c}) = p(a_1 \mid \mathbf{c}) p(a_2 \mid \mathbf{c}),

    where :math:`\mathbf{c}` is a vector of the ``n_splines`` basis spline coefficients.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    a1, a2 : array_like
        Primary and secondary component spin magnitude parameter estimation samples for basis evaluation.
    a1_inj, a2_inj : array_like
        Primary and secondary component spin magnitude injection samples for basis evaluation.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the basis spline model.
    """

    def __init__(
        self,
        n_splines,
        a1,
        a2,
        a1_inj,
        a2_inj,
        **kwargs,
    ):
        self.primary_model = BSplineSpinMagnitude(
            n_splines=n_splines,
            a=a1,
            a_inj=a1_inj,
            **kwargs,
        )
        self.secondary_model = BSplineSpinMagnitude(
            n_splines=n_splines,
            a=a2,
            a_inj=a2_inj,
            **kwargs,
        )

    def __call__(self, coefs, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        coefs : array_like
            Spline coefficients.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Joint probability density for parameter estimation or injection samples.
        """
        p_a1 = self.primary_model(coefs, pe_samples=pe_samples)
        p_a2 = self.secondary_model(coefs, pe_samples=pe_samples)
        return p_a1 * p_a2


class BSplineIndependentSpinMagnitudes(object):
    r"""A B-Spline model for the spin magnitudes of the primary and secondary components assuming
    they are independently distributed,

    .. math::
        p(a_1, a_2 \mid \mathbf{c}_1, \mathbf{c}_2) = p(a_1 \mid \mathbf{c}_1) p(a_2 \mid \mathbf{c}_2),

    where :math:`\mathbf{c}_1, \mathbf{c}_2` are vectors of the ``n_splines1``, ``n_splines2`` basis
    spline coefficients for the primary and secondary component spin magnitudes, respectively.

    Parameters
    ----------
    n_splines1, n_splines2 : int
        Number of basis functions, i.e., the number of degrees of freedom, of the
        primary and secondary component spline models.
    a1, a2 : array_like
        Primary and secondary component spin magnitude parameter estimation samples for basis evaluation.
    a1_inj, a2_inj : array_like
        Primary and secondary component spin magnitude injection samples for basis evaluation.
    kwargs1, kwargs2 : dict, optional
        Additional keyword arguments to pass to the basis spline model for the primary and secondary components.
    **kwargs : dict, optional
        Additional keyword arguments to pass to both basis spline models.
    """

    def __init__(
        self,
        n_splines1,
        n_splines2,
        a1,
        a2,
        a1_inj,
        a2_inj,
        kwargs1={},
        kwargs2={},
        **kwargs,
    ):
        self.primary_model = BSplineSpinMagnitude(
            n_splines=n_splines1,
            a=a1,
            a_inj=a1_inj,
            **kwargs1,
            **kwargs,
        )
        self.secondary_model = BSplineSpinMagnitude(
            n_splines=n_splines2,
            a=a2,
            a_inj=a2_inj,
            **kwargs2,
            **kwargs,
        )

    def __call__(self, pcoefs, scoefs, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        pcoefs, scoefs : array_like
            Spline coefficients for the (p)rimary and (s)econdary components.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Joint probability density for parameter estimation or injection samples.
        """
        p_a1 = self.primary_model(pcoefs, pe_samples=pe_samples)
        p_a2 = self.secondary_model(scoefs, pe_samples=pe_samples)
        return p_a1 * p_a2


class BSplineIIDSpinTilts(object):
    """A B-Spline model for the (cosine of) spin tilts of both binary components assuming
    they are independently and identically distributed (IID),

    .. math::
        p(\cos{t_1}, \cos{t_2} \mid \mathbf{c}) = p(\cos{t_1} \mid \mathbf{c}) p(\cos{t_2} \mid \mathbf{c}),

    where :math:`\mathbf{c}` is a vector of the ``n_splines`` basis spline coefficients.

    Parameters
    ----------
    n_splines1 : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    ct1, ct2 : array_like
        Primary and secondary component spin cosine tilt parameter estimation samples for basis evaluation.
    ct1_inj, ct2_inj : array_like
        Primary and secondary component spin cosine tilt injection samples for basis evaluation.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the basis spline model.
    """

    def __init__(
        self,
        n_splines,
        ct1,
        ct2,
        ct1_inj,
        ct2_inj,
        **kwargs,
    ):
        self.primary_model = BSplineSpinTilt(
            n_splines=n_splines,
            ct=ct1,
            ct_inj=ct1_inj,
            **kwargs,
        )
        self.secondary_model = BSplineSpinTilt(
            n_splines=n_splines,
            ct=ct2,
            ct_inj=ct2_inj,
            **kwargs,
        )

    def __call__(self, coefs, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        coefs : array_like
            Spline coefficients.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like:
            Joint probability density for parameter estimation or injection samples.
        """
        p_ct1 = self.primary_model(coefs, pe_samples=pe_samples)
        p_ct2 = self.secondary_model(coefs, pe_samples=pe_samples)
        return p_ct1 * p_ct2


class BSplineIndependentSpinTilts(object):
    """A B-Spline model for the (cosine of) spin tilts of the primary and secondary components assuming
    they are independently distributed,

    .. math::
        p(\cos{t_1}, \cos{t_2} \mid \mathbf{c}_1, \mathbf{c}_2) = p(\cos{t_1} \mid \mathbf{c}_1) p(\cos{t_2} \mid \mathbf{c}_2),

    where :math:`\mathbf{c}_1, \mathbf{c}_2` are vectors of the ``n_splines1``, ``n_splines2`` basis
    spline coefficients for the primary and secondary component cosine spin tilts, respectively.

    Parameters
    ----------
    n_splines1, n_splines2 : int
        Number of basis functions, i.e., the number of degrees of freedom, of the
        primary and secondary component spline models.
    ct1, ct2 : array_like
        Primary and secondary component spin cosine tilt parameter estimation samples for basis evaluation.
    ct1_inj, ct2_inj : array_like
        Primary and secondary component spin cosine tilt injection samples for basis evaluation.
    kwargs1, kwargs2 : dict, optional
        Additional keyword arguments to pass to the basis spline model for the primary and secondary components.
    **kwargs : dict, optional
        Additional keyword arguments to pass to both basis spline models.
    """

    def __init__(
        self,
        n_splines1,
        n_splines2,
        ct1,
        ct2,
        ct1_inj,
        ct2_inj,
        kwargs1={},
        kwargs2={},
        **kwargs,
    ):
        self.primary_model = BSplineSpinTilt(
            n_splines=n_splines1,
            ct=ct1,
            ct_inj=ct1_inj,
            **kwargs1,
            **kwargs,
        )
        self.secondary_model = BSplineSpinTilt(
            n_splines=n_splines2,
            ct=ct2,
            ct_inj=ct2_inj,
            **kwargs2,
            **kwargs,
        )

    def __call__(self, pcoefs, scoefs, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        pcoefs, scoefs : array_like
            Spline coefficients for the (p)rimary and (s)econdary components.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Joint probability density for parameter estimation or injection samples.
        """
        p_ct1 = self.primary_model(pcoefs, pe_samples=pe_samples)
        p_ct2 = self.secondary_model(scoefs, pe_samples=pe_samples)
        return p_ct1 * p_ct2


class BSplinePrimaryPowerlawRatio(object):
    r"""A B-Spline model in primary mass and a powerlaw model in mass ratio,

    .. math::
        p(m_1, q \mid \mathbf{c}, \beta) = p(m_1 \mid \mathbf{c}) p(q \mid \beta, m_1, m_{\mathrm{min}}),

    where :math:`\mathbf{c}` is a vector of the ``n_splines`` basis spline coefficients, and
    :math:`\beta` is the powerlaw slope of the mass ratio distribution.

    See Also
    --------
    gwinferno.distributions.powerlaw_pdf : Powerlaw probability density function.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    m1 : array_like
        Primary component mass parameter estimation samples for basis evaluation.
    m1_inj : array_like
        Primary component mass injection samples for basis evaluation.
    mmin : float, default=2
        Minimum component mass, setting lower bounds on the primary mass and mass ratio (:math:`q>m_\mathrm{min}/m_1`).
    mmax : float, default=100
        Maximum component mass, setting the upper bound on the primary mass.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the basis spline model.
    """

    def __init__(
        self,
        n_splines,
        m1,
        m1_inj,
        mmin=2,
        mmax=100,
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            n_splines,
            m1,
            m1_inj,
            mmin=mmin,
            mmax=mmax,
            **kwargs,
        )

    def __call__(self, m1, q, beta, mmin, coefs, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        m1, q : array_like
            Primary masses and mass ratios for computing joint probability density.
        mmin : float
            Minimum component mass, setting lower bounds on the primary mass and mass ratio (:math:`q>m_\mathrm{min}/m_1`).
        coefs (array_like):
            Spline coefficients.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Joint probability density for parameter estimation or injection samples.
        """
        p_m1 = self.primary_model(coefs, pe_samples=pe_samples)
        p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
        return p_m1 * p_q


class PLPeakPrimaryBSplineRatio(object):
    r"""A powerlaw + gaussian peak primary mass model and B-Spline model in mass ratio.

    .. math::
        p(m_1, q \mid \mathbf{c}, \alpha, \mu_\mathrm{peak}, \sigma_\mathrm{peak}, f_\mathrm{peak}) =
        p(m_1 \mid \alpha, \mu_\mathrm{peak}, \sigma_\mathrm{peak}, f_\mathrm{peak}) p(q \mid \mathbf{c}, m_1, m_{\mathrm{min}}),

    where :math:`\mathbf{c}` is a vector of the ``n_splines`` basis spline coefficients,
    :math:`\alpha` is the powerlaw slope of the primary component mass distribution, :math:`\mu_\mathrm{peak}` and
    :math:`\sigma_\mathrm{peak}` the mean and standard deviation of the peak in mass, and :math:`f_\mathrm{peak}`
    is the mixing fraction between the powerlaw and peak in mass.

    See Also
    --------
    gwinferno.models.parametric.parametric.plpeak_primary_pdf : Powerlaw+Peak primary mass model density.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    q : array_like
        Mass ratio parameter estimation samples for basis evaluation.
    q_inj : array_like
        Mass ratio injection samples for basis evaluation.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the basis spline model.
    """

    def __init__(
        self,
        n_splines,
        q,
        q_inj,
        **kwargs,
    ):
        self.ratio_model = BSplineRatio(
            n_splines,
            q,
            q_inj,
            **kwargs,
        )

    def __call__(self, m1, alpha, mmin, mmax, peak_mean, peak_sd, peak_frac, coefs, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        m1 : array_like
            Primary masses for computing joint probability density.
        alpha : float
            Powerlaw slope of the primary mass distribution.
        mmin : float
            Minimum component mass, the lower bound on the primary mass.
        mmax : float
            Maximum component mass, the upper bound on the primary mass.
        peak_mean : float
            Mean of the peak in mass.
        peak_sd : float
            Standard deviation of the peak in mass.
        peak_frac : float
            Fraction of binaries in the peak in mass.
        coefs : array_like
            Spline coefficients.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Joint probability density for parameter estimation or injection samples.
        """
        p_q = self.ratio_model(coefs, pe_samples=pe_samples)
        p_m1 = plpeak_primary_pdf(m1, alpha, mmin, mmax, peak_mean, peak_sd, peak_frac)
        return p_m1 * p_q


class BSplinePrimaryBSplineRatio(object):
    r"""B-Spline models for the primary mass and mass ratio,

    .. math::
        p(m_1, q \mid \mathbf{c}_m, \mathbf{c}_q) = p(m_1 \mid \mathbf{c}_m) p(q \mid \mathbf{c}_q),

    where :math:`\mathbf{c}_m` and :math:`\mathbf{c}_q` are vectors of the ``n_splines_m`` and ``n_splines_q``
    basis spline coefficients for the primary mass and mass ratio, respectively.

    Parameters
    ----------
    n_splines_m, n_splines_q : int
        Number of basis functions, i.e., the number of degrees of freedom, of the
        primary component mass and mass ratio spline models.
    m1 : array_like
        Primary component mass parameter estimation samples for basis evaluation.
    m1_inj : array_like
        Primary component mass injection samples for basis evaluation.
    q : array_like
        Mass ratio parameter estimation samples for basis evaluation.
    q_inj : array_like
        Mass ratio injection samples for basis evaluation.
    mmax : float, default=100
        Maximum component mass.
    m1min : float, default=3
        Minimum primary component mass.
    m2min : float, default=3
        Minimum secondary component mass, setting lower bound on the mass ratio (:math:`q>m_{2,\mathrm{min}}/m_\mathrm{max}`).
    kwargs_m, kwargs_q : dict, optional
        Additional keyword arguments to pass to the basis spline models for the primary component mass and mass ratio.
    **kwargs : dict, optional
        Additional keyword arguments to pass to both basis spline models.
    """

    def __init__(
        self,
        n_splines_m,
        n_splines_q,
        m1,
        m1_inj,
        q,
        q_inj,
        mmax=100.0,
        m1min=3.0,
        m2min=3.0,
        kwargs_m={},
        kwargs_q={},
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            n_splines_m,
            m1,
            m1_inj,
            mmin=m1min,
            mmax=mmax,
            **kwargs_m,
            **kwargs,
        )
        self.ratio_model = BSplineRatio(
            n_splines_q,
            q,
            q_inj,
            qmin=m2min / mmax,
            **kwargs_q,
            **kwargs,
        )

    def __call__(self, mcoefs, qcoefs, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        mcoefs, qcoefs : array_like
            Spline coefficients for the primary component mass and mass ratio.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Joint probability density for parameter estimation or injection samples.
        """
        return self.ratio_model(qcoefs, pe_samples=pe_samples) * self.primary_model(mcoefs, pe_samples=pe_samples)


class BSplineIIDComponentMasses(object):
    r"""B-Spline model for the masses of both binary components assuming they are independently and identically distributed (IID),
    with an optional pairing term as a powerlaw in mass ratio.

    .. math::
        p(m_1, m_2 \mid \mathbf{c}, \beta) = p(m_1 \mid \mathbf{c}) p(m_2 \mid \mathbf{c}) \left(\frac{m_2}{m_1}\right)^\beta,

    where :math:`\mathbf{c}` is a vector of the ``n_splines`` basis spline coefficients.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom, of the spline model.
    m1, m2 : array_like
        Primary and secondary component mass parameter estimation samples for basis evaluation.
    m1_inj, m2_inj : array_like
        Primary and secondary component mass injection samples for basis evaluation.
    mmin : float, default=2
        Minimum component mass.
    mmax : float, default=100
        Maximum component mass.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the basis spline model.
    """

    def __init__(
        self,
        n_splines,
        m1,
        m2,
        m1_inj,
        m2_inj,
        mmin=2,
        mmax=100,
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            n_splines=n_splines,
            m=m1,
            m_inj=m1_inj,
            mmin=mmin,
            mmax=mmax,
            **kwargs,
        )
        self.secondary_model = BSplineMass(
            n_splines=n_splines,
            m=m2,
            m_inj=m2_inj,
            mmin=mmin,
            mmax=mmax,
            **kwargs,
        )
        self.qs = [m2_inj / m1_inj, m2 / m1]

    def __call__(self, coefs, beta=0, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        coefs : array_like
            Spline coefficients.
        beta : float, default=0
            Mass ratio powerlaw slope.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Joint probability density for parameter estimation or injection samples.
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
    r"""A B-Spline model for the masses of the primary and secondary components assuming
    they are independently distributed, with an optional pairing term as a powerlaw in mass ratio,

    .. math::
        p(m_1, m_2 \mid \mathbf{c}_1, \mathbf{c}_2, \beta) = p(m_1 \mid \mathbf{c}_1) p(m_2 \mid \mathbf{c}_2) \left(\frac{m_2}{m_1}\right)^\beta,

    where :math:`\mathbf{c}_1` and :math:`\mathbf{c}_2` are vectors of the ``n_splines1`` and ``n_splines2``
    basis spline coefficients for the primary and secondary component masses, respectively.

    Parameters
    ----------
    n_splines1, n_splines2 : int
        Number of basis functions, i.e., the number of degrees of freedom, of the
        primary and secondary component spline models.
    m1, m2 : array_like
        Primary and secondary component mass parameter estimation samples for basis evaluation.
    m1_inj, m2_inj : array_like
        Primary and secondary component mass injection samples for basis evaluation.
    mmin1, mmax1 : float, default=2, 100
        Minimum and maximum primary component mass.
    mmin2, mmax2 : float, default=2, 100
        Minimum and maximum secondary component mass.
    kwargs1, kwargs2 : dict, optional
        Additional keyword arguments to pass to the basis spline model for the primary and secondary components.
    **kwargs : dict, optional
        Additional keyword arguments to pass to both basis spline models.
    """

    def __init__(
        self,
        n_splines1,
        n_splines2,
        m1,
        m2,
        m1_inj,
        m2_inj,
        mmin1=2,
        mmax1=100,
        mmin2=2,
        mmax2=100,
        kwargs1={},
        kwargs2={},
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            n_splines=n_splines1,
            m=m1,
            m_inj=m1_inj,
            mmin=mmin1,
            mmax=mmax1,
            **kwargs1,
            **kwargs,
        )
        self.secondary_model = BSplineMass(
            n_splines=n_splines2,
            m=m2,
            m_inj=m2_inj,
            mmin=mmin2,
            mmax=mmax2,
            **kwargs2,
            **kwargs,
        )
        self.qs = [m2_inj / m1_inj, m2 / m1]

    def __call__(self, pcoefs, scoefs, beta=0, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        pcoefs, scoefs : array_like
            Spline coefficients for the (p)rimary and (s)econdary components.
        beta : float, default=0
            Mass ratio powerlaw slope.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Joint probability density for parameter estimation or injection samples.
        """
        p_m1 = self.primary_model(pcoefs, pe_samples=pe_samples)
        p_m2 = self.secondary_model(scoefs, pe_samples=pe_samples)
        dim = 1 if pe_samples else 0
        return p_m1 * p_m2 * self.qs[dim] ** beta


class BSplineEffectiveSpinDims(object):
    r"""B-Spline models for the effective spin (:math:`\chi_\mathrm{eff}`) and
    effective precession (:math:`\chi_\mathrm{p}`) of binaries,

    .. math::
        p(\chi_\mathrm{eff}, \chi_\mathrm{p} \mid \mathbf{c}_\mathrm{eff}, \mathbf{c}_\mathrm{p}) = p(\chi_\mathrm{eff} \mid \mathbf{c}_\mathrm{eff}) p(\chi_\mathrm{p} \mid \mathbf{c}_\mathrm{p}),

    where :math:`\mathbf{c}_\mathrm{eff}` and :math:`\mathbf{c}_\mathrm{p}` are vectors of the ``n_splines_e``
    and ``n_splines_p`` basis spline coefficients for the effective spin and effective precession, respectively.

    Parameters
    ----------
    n_splines_e, n_splines_p : int
        Number of basis functions, i.e., the number of degrees of freedom, of the
        (e)ffective spin and effective (p)recession spline models.
    chieff, chip : array_like
        Effective spin and effective precession parameter estimation samples for basis evaluation.
    chieff_inj, chip_inj : array_like
        Effective spin and effective precession injection samples for basis evaluation.
    kwargs_e, kwargs_p : dict, optional
        Additional keyword arguments to pass to the basis spline models for the effective spin and effective precession.
    **kwargs : dict, optional
        Additional keyword arguments to pass to both basis spline models.
    """

    def __init__(
        self,
        n_splines_e,
        n_splines_p,
        chieff,
        chip,
        chieff_inj,
        chip_inj,
        kwargs_e={},
        kwargs_p={},
        **kwargs,
    ):
        self.chi_eff_model = BSplineChiEffective(
            n_splines_e,
            chieff,
            chieff_inj,
            **kwargs_e,
            **kwargs,
        )

        self.chi_p_model = BSplineChiPrecess(
            n_splines_p,
            chip,
            chip_inj,
            **kwargs_p,
            **kwargs,
        )

    def __call__(self, ecoefs, pcoefs, pe_samples=True):
        """Evaluate the joint probability density over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        ecoefs, pcoefs : array_like
            Spline coefficients for the effective spin and effective precession.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Joint probability density for parameter estimation or injection samples.
        """
        p_chieff = self.chi_eff_model(ecoefs, pe_samples=pe_samples)
        p_chip = self.chi_p_model(pcoefs, pe_samples=pe_samples)
        return p_chieff * p_chip
