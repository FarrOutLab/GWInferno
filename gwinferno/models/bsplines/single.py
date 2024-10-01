"""
a module that stores 1D population models constructed from BSplines
"""

import jax.numpy as jnp
import numpy as np
from jax.scipy.integrate import trapezoid

from gwinferno.cosmology import PLANCK_2015_Cosmology as Planck15

from ...interpolation import BSpline
from ...interpolation import LogXBSpline
from ...interpolation import LogXLogYBSpline
from ...interpolation import LogYBSpline


class Base1DBSplineModel(object):
    """Base class for basis splines for population inference

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    xx : array_like
        Parameter estimation samples for basis evaluation.
    xx_inj : array_like
        Injection samples for basis evaluation.
    degree : int, default=3
        Degree of the spline, i.e., `3` for cubic splines.
    xrange : tuple, default=(0.0, 1.0)
        Domain of the spline.
    basis : class, default=BSpline
        Type of basis to use, e.g., `LogYBSpline`.
    """

    def __init__(
        self,
        n_splines,
        xx,
        xx_inj,
        xrange=(0.0, 1.0),
        degree=3,
        basis=BSpline,
        **kwargs,
    ):
        self.n_splines = n_splines
        self.xmin, self.xmax = xrange
        self.degree = degree
        self.interpolator = basis(
            n_splines,
            xrange=xrange,
            k=degree + 1,
            **kwargs,
        )
        self._valid_xx = (xx >= self.xmin) & (xx <= self.xmax)
        self._valid_xx_inj = (xx_inj >= self.xmin) & (xx_inj <= self.xmax)
        self.pe_design_matrix = self.interpolator.bases(xx[self._valid_xx])
        self.inj_design_matrix = self.interpolator.bases(xx_inj[self._valid_xx_inj])
        self.funcs = [self.inj_pdf, self.pe_pdf]

    def eval_spline(self, bases, coefs):
        """Given design matrix ``bases`` and coefficients ``coefs``, project coefficients onto the basis.

        Parameters
        ----------
        bases : array_like
            Design matrix of the spline, i.e., basis functions evaluated at samples.
        coefs : array_like
            Basis spline coefficients.

        Returns
        -------
        array_like
            The linear combination of the basis components given the coefficients.
        """
        return self.interpolator.project(bases, coefs)

    def pe_pdf(self, coefs):
        """Project the coefficients ``coefs`` onto the design matrix evaluated at the parameter estimation samples.

        Parameters
        ----------
        coefs : array_like
            Basis spline coefficients.

        Returns
        -------
        array_like
            The linear combination of the basis components evaluated at the parameter estimation samples given the coefficients.
        """
        pdf = jnp.zeros(self._valid_xx.shape)
        pdf = pdf.at[self._valid_xx].set(self.eval_spline(self.pe_design_matrix, coefs))
        return pdf

    def inj_pdf(self, coefs):
        """Project the coefficients ``coefs`` onto the design matrix evaluated at the injection samples.

        Parameters
        ----------
        coefs : array_like
            Basis spline coefficients.

        Returns
        -------
        array_like
            The linear combination of the basis components evaluated at the injection samples given the coefficients.
        """
        pdf = jnp.zeros(self._valid_xx_inj.shape)
        pdf = pdf.at[self._valid_xx_inj].set(self.eval_spline(self.inj_design_matrix, coefs))
        return pdf

    def __call__(self, coefs, pe_samples=True):
        """Evaluate the projection of the coefficients along the design matrix over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        coefs : array_like
            Basis spline coefficients.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            The linear combination of the basis components evaluated at the parameter estimation or injection samples given the coefficients.
        """
        return self.funcs[1](coefs) if pe_samples else self.funcs[0](coefs)


class BSplineSpinMagnitude(Base1DBSplineModel):
    """A B-Spline model for the spin magnitude of a single binary component.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    a1 : array_like
        Component spin magnitude parameter estimation samples for basis evaluation.
    a1_inj : array_like
        Component spin magnitude injection samples for basis evaluation.
    basis : class, default=LogYBSpline
        Type of basis to use.
    """

    def __init__(
        self,
        n_splines,
        a,
        a_inj,
        basis=LogYBSpline,
        **kwargs,
    ):
        xrange = kwargs.pop("xrange", (0.0, 1.0))
        super().__init__(
            n_splines,
            a,
            a_inj,
            basis=basis,
            xrange=xrange,
            **kwargs,
        )


class BSplineSpinTilt(Base1DBSplineModel):
    """A B-Spline model for the (cosine of) spin tilt of a single binary component.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    ct : array_like
        Component spin cosine tilt parameter estimation samples for basis evaluation.
    ct_inj : array_like
        Component spin cosine tilt injection samples for basis evaluation.
    basis : class, default=LogYBSpline
        Type of basis to use.
    """

    def __init__(
        self,
        n_splines,
        ct,
        ct_inj,
        basis=LogYBSpline,
        **kwargs,
    ):
        xrange = kwargs.pop("xrange", (-1.0, 1.0))
        super().__init__(
            n_splines,
            ct,
            ct_inj,
            basis=basis,
            xrange=xrange,
            **kwargs,
        )


class BSplineChiEffective(Base1DBSplineModel):
    r"""A B-Spline model for the binary effective spin :math:`\chi_\mathrm{eff}`.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    chieff : array_like
        Effective spin parameter estimation samples for basis evaluation.
    chieff_inj : array_like
        Effective spin injection samples for basis evaluation.
    basis : class, default=BSpline
        Type of basis to use.
    """

    def __init__(
        self,
        n_splines,
        chieff,
        chieff_inj,
        basis=BSpline,
        **kwargs,
    ):
        xrange = kwargs.pop("xrange", (-1.0, 1.0))
        super().__init__(
            n_splines,
            chieff,
            chieff_inj,
            basis=basis,
            xrange=xrange,
            **kwargs,
        )


class BSplineSymmetricChiEffective(Base1DBSplineModel):
    """A B-Spline model for the binary effective spin :math:`\chi_\mathrm{eff}`
    that is symmetric about :math:`\chi_\mathrm{eff} = 0`.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    chieff : array_like
        Effective spin parameter estimation samples for basis evaluation.
    chieff_inj : array_like
        Effective spin injection samples for basis evaluation.
    basis : class, default=BSpline
        Type of basis to use.
    """

    def __init__(
        self,
        n_splines,
        chieff,
        chieff_inj,
        basis=BSpline,
        **kwargs,
    ):
        xrange = kwargs.pop("xrange", (0.0, 1.0))
        super().__init__(
            n_splines,
            jnp.abs(chieff),
            jnp.abs(chieff_inj),
            basis=basis,
            xrange=xrange,
            **kwargs,
        )

    def __call__(self, coefs, pe_samples=True):
        """Evaluate the projection of the coefficients along the design matrix over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        coefs : array_like
            Basis spline coefficients.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            The linear combination of the basis components evaluated at the parameter estimation or injection samples given the coefficients.
        """
        return 0.5 * self.funcs[1](coefs) if pe_samples else 0.5 * self.funcs[0](coefs)


class BSplineChiPrecess(Base1DBSplineModel):
    r"""A B-Spline model for the binary effective precession :math:`\chi_\mathrm{p}`.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    chip : array_like
        Effective precession parameter estimation samples for basis evaluation.
    chip_inj : array_like
        Effective precession injection samples for basis evaluation.
    basis : class, default=BSpline
        Type of basis to use.
    """

    def __init__(
        self,
        n_splines,
        chip,
        chip_inj,
        basis=BSpline,
        **kwargs,
    ):
        xrange = kwargs.pop("xrange", (0.0, 1.0))
        super().__init__(
            n_splines,
            chip,
            chip_inj,
            basis=basis,
            xrange=xrange,
            **kwargs,
        )


class BSplineRatio(Base1DBSplineModel):
    """A B-Spline model for the binary mass ratio.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    q : array_like
        Mass ratio parameter estimation samples for basis evaluation.
    q_inj : array_like
        Mass ratio injection samples for basis evaluation.
    qmin : float, default=0
        Minimum mass ratio. Ignored if ``xrange`` is provided.
    basis : class, default=LogYBSpline
        Type of basis to use.
    """

    def __init__(
        self,
        n_splines,
        q,
        q_inj,
        qmin=0,
        basis=LogYBSpline,
        **kwargs,
    ):
        xrange = kwargs.pop("xrange", (qmin, 1))
        super().__init__(
            n_splines,
            q,
            q_inj,
            basis=basis,
            xrange=xrange,
            **kwargs,
        )


class BSplineMass(Base1DBSplineModel):
    """A B-Spline model for the mass of a single binary component.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    m : array_like
        Component mass parameter estimation samples for basis evaluation.
    m_inj : array_like
        Component mass injection samples for basis evaluation.
    mmin : float, default=2
        Minimum component mass. Ignored if ``xrange`` is provided.
    mmax : float, default=100
        Maximum component mass. Ignored if ``xrange`` is provided.
    basis : class, default=LogXLogYBSpline
        Type of basis to use.
    """

    def __init__(
        self,
        n_splines,
        m,
        m_inj,
        mmin=2,
        mmax=100,
        basis=LogXLogYBSpline,
        **kwargs,
    ):
        xrange = kwargs.pop("xrange", (mmin, mmax))
        super().__init__(
            n_splines,
            m,
            m_inj,
            basis=basis,
            xrange=xrange,
            **kwargs,
        )


class BSplineRedshift(Base1DBSplineModel):
    r"""A B-Spline model for redshift. The B-Spline will define the *volumetric* rate density :math:`r`, which relates to the
    merger-rate per unit redshift in the detector-frame :math:`R` by

    .. math::
        R(z) = \frac{r(z)}{1+z} \frac{dV_c}{dz},

    where :math:`dV_c/dz` is the co-moving volume element and :math:`r(z)` is the merger rate per unit co-moving volume.

    Parameters
    ----------
    n_splines : int
        Number of basis functions, i.e., the number of degrees of freedom of the spline model.
    z : array_like
        Redshift parameter estimation samples for basis evaluation.
    z_inj : array_like
        Redshift injection samples for basis evaluation.
    dVdc : array_like
        Differential co-moving volume for each parameter estimation sample.
    dVdc_inj : array_like
        Differential co-moving volume for each injection sample.
    zmax : float, default=2.3
        Maximum redshift. Ignored if ``xrange`` is provided.
    basis : class, default=LogXBSpline
        Type of basis to use.
    """

    def __init__(
        self,
        n_splines,
        z,
        z_inj,
        dVdc,
        dVdc_inj,
        zmax=2.3,
        basis=LogXBSpline,
        **kwargs,
    ):
        xrange = kwargs.pop("xrange", (1e-4, zmax))
        super().__init__(
            n_splines,
            z,
            z_inj,
            xrange=xrange,
            basis=basis,
            **kwargs,
        )
        self.zmax = zmax
        self.dVcdzgrid = jnp.array(Planck15.dVcdz(np.linspace(1e-4, zmax, 2500)).value)
        self.differential_comov_vols = [dVdc_inj, dVdc]
        self.zs = [z_inj, z]

    def norm(self, coefs):
        """Compute the normalization coefficient for the redshift basis spline; useful for computing the merger rate.

        Parameters
        ----------
        coefs : array_like
            Basis spline coefficients.

        Returns
        -------
        float
            The redshift normalization constant.
        """
        return trapezoid(
            self.dVcdzgrid / (1 + self.grid) * jnp.einsum("i...,i->...", self.grid_bases, coefs),
            self.grid,
        )

    def __call__(self, coefs, pe_samples=True):
        """Evaluate the merger-rate per unit redshift in the detector-frame :math:`R`. Use flag
        `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Parameters
        ----------
        coefs : array_like
            Basis spline coefficients.
        pe_samples : bool, default=True
            If `True`, design matrix is evaluated across parameter estimation samples.
            If `False`, design matrix is evaluated across injection samples.

        Returns
        -------
        array_like
            Merger-rate per unit redshift in the detector-frame :math:`R`.
        """
        return (
            self.funcs[1](coefs) * self.differential_comov_vols[1] / (1 + self.zs[1])
            if pe_samples
            else self.funcs[0](coefs) * self.differential_comov_vols[0] / (1 + self.zs[0])
        )
