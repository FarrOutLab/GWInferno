"""
a module that stores 1D population models constructed from bsplines
"""

import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck15
from jax.scipy.integrate import trapezoid

from ...interpolation import BSpline


class Base1DBSplineModel(object):
    """Class to construct basis spline for population inference

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        xx (array_like): posterior samples to evaluate design matrix at
        xx_inj (array_like): injection samples to evaluate design matrix at
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        xrange (tuple, optional): domain of spline. Defaults to (0, 1).
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
        basis (class, optional): type of basis to use (ex. LogYBSpline). Defaults to Bspline.

    """

    def __init__(
        self,
        n_splines,
        xx,
        xx_inj,
        knots=None,
        xrange=(0, 1),
        degree=3,
        basis=BSpline,
        **kwargs,
    ):
        self.n_splines = n_splines
        self.xmin, self.xmax = xrange
        self.degree = degree
        self.interpolator = basis(
            n_splines,
            knots=knots,
            xrange=xrange,
            k=degree + 1,
            **kwargs,
        )
        self.pe_design_matrix = jnp.array(self.truncate_dmat(xx, self.interpolator.bases(xx)))
        self.inj_design_matrix = jnp.array(self.truncate_dmat(xx_inj, self.interpolator.bases(xx_inj)))
        self.funcs = [self.inj_pdf, self.pe_pdf]

    def truncate_dmat(self, x, dmat):
        """ensures the design matrix is truncated outside of the basis interval [xmin, xmax].

        Args:
            x (array_like): domain values
            dmat (array_like): design matrix

        Returns:
            array_like: truncated design matrix
        """
        return jnp.where(jnp.less(x, self.xmin) | jnp.greater(x, self.xmax), 0, dmat)

    def eval_spline(self, bases, coefs):
        """given design matrix and coefficients, project coefficients onto the basis.

        Args:
            bases (array_like): design matrix of of spline
            coefs (array_like): basis spline coefficients

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        return self.interpolator.project(bases, coefs)

    def pe_pdf(self, coefs):
        """projects the coefficients onto the design matrix evaluated at the posterior samples

        Args:
            coefs (array_like): basis spline coefficients

        Returns:
            array_like: The linear combination of the basis components evaluated at the posterior samples given the coefficients
        """
        return self.eval_spline(self.pe_design_matrix, coefs)

    def inj_pdf(self, coefs):
        """projects the coefficients onto the design matrix evaluated at the injection samples

        Args:
            coefs (array_like): basis spline coefficients

        Returns:
            array_like: The linear combination of the basis components evaluated at the injection samples given the coefficients
        """
        return self.eval_spline(self.inj_design_matrix, coefs)

    def __call__(self, coefs, pe_samples=True):
        """will evalute the projection of the coefficients along the design matrix evaluated at either posterior samples or injection samples.
            Use flag pe_samples to specify which type of samples are being evaluated (pe or injection).

        Args:
            coefs (array_like): basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: The linear combination of the basis components evaluated at the posterior or injection samples given the coefficients.
        """
        return self.funcs[1](coefs) if pe_samples else self.funcs[0](coefs)


class BSplineSpinMagnitude(Base1DBSplineModel):
    """Class to construct a spin magnitude B-Spline model for a single binary component

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        a (array_like): spin magntiude pe samples to evaluate the basis spline at
        a_inj (array_like): spin magnitude injection samples to evalute the basis spline at
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        a,
        a_inj,
        knots=None,
        degree=3,
        **kwargs,
    ):
        super().__init__(
            n_splines,
            a,
            a_inj,
            knots=knots,
            degree=degree,
            **kwargs,
        )


class BSplineSpinTilt(Base1DBSplineModel):
    """Class to construct a cosine tilt (cos(theta)) B-Spline model for a single binary component

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        ct (array_like): cosine tilt pe samples to evaluate the basis spline at
        ct_inj (array_like): cosine tilt injection samples to evalute the basis spline at
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        ct,
        ct_inj,
        knots=None,
        degree=3,
        **kwargs,
    ):
        super().__init__(
            n_splines,
            ct,
            ct_inj,
            knots=knots,
            degree=degree,
            xrange=(-1, 1),
            **kwargs,
        )


class BSplineChiEffective(Base1DBSplineModel):
    """Class to construct a chi effective B-Spline model for a single binary component

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        chieff (array_like): chi effective pe samples to evaluate the basis spline at
        chieff_inj (array_like): chi effective injection samples to evalute the basis spline at
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        chieff,
        chieff_inj,
        knots=None,
        degree=3,
        **kwargs,
    ):
        super().__init__(
            n_splines,
            chieff,
            chieff_inj,
            knots=knots,
            degree=degree,
            xrange=(-1, 1),
            **kwargs,
        )


class BSplineSymmetricChiEffective(Base1DBSplineModel):
    """Class to construct a chi effective B-Spline model symmetric about chi effective = zero  for a single binary component

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        chieff (array_like): chi effective pe samples to evaluate the basis spline at
        chieff_inj (array_like): chi effective injection samples to evalute the basis spline at
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        chieff,
        chieff_inj,
        knots=None,
        degree=3,
        **kwargs,
    ):
        super().__init__(
            n_splines,
            jnp.abs(chieff),
            jnp.abs(chieff_inj),
            knots=knots,
            degree=degree,
            xrange=(0, 1),
            **kwargs,
        )

    def __call__(self, coefs, pe_samples=True):
        """will evalute the projection of the coefficients along the design matrix evaluated at either posterior samples or injection samples.
            Use flag pe_samples to specify which type of samples are being evaluated (pe or injection).

        Args:
            coefs (array_like): basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: The linear combination of the basis components evaluated at the posterior or injection samples given the coefficients.
        """
        return 0.5 * self.funcs[1](coefs) if pe_samples else 0.5 * self.funcs[0](coefs)


class BSplineChiPrecess(Base1DBSplineModel):
    """Class to construct an effective precession (chi_p) B-Spline model for a single binary component

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        chip (array_like): chi_p pe samples to evaluate the basis spline at
        chip_inj (array_like): chi_p injection samples to evalute the basis spline at
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        chip,
        chip_inj,
        knots=None,
        degree=3,
        **kwargs,
    ):
        super().__init__(
            n_splines,
            chip,
            chip_inj,
            knots=knots,
            degree=degree,
            **kwargs,
        )


class BSplineRatio(Base1DBSplineModel):
    """Class to construct a mass ratio (q) B-Spline model for a single binary component

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        q (array_like): mass ratio pe samples to evaluate the basis spline at
        q_inj (array_like): mass ratio injection samples to evalute the basis spline at
        qmin (float, optional): minimum mass ratio value. Spline is truncated below this minimum mass ratio.
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        q,
        q_inj,
        qmin=0,
        knots=None,
        degree=3,
        **kwargs,
    ):
        super().__init__(
            n_splines,
            q,
            q_inj,
            knots=knots,
            degree=degree,
            xrange=(qmin, 1),
            **kwargs,
        )


class BSplineMass(Base1DBSplineModel):
    """Class to construct a mass B-Spline model for a single binary component

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        m (array_like): mass pe samples to evaluate the basis spline at
        m_inj (array_like): mass injection samples to evalute the basis spline at
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred. Defaults to None.
        mmin (float, optional): minimum mass value. Spline is truncated below this minimum mass. Defaults to 2.
        mmax (float, optional): maximum mass value. Spline is truncated above this maximum mass. Defaults to 100.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        n_splines,
        m,
        m_inj,
        knots=None,
        mmin=2,
        mmax=100,
        degree=3,
        **kwargs,
    ):
        super().__init__(
            n_splines,
            m,
            m_inj,
            knots=knots,
            xrange=(mmin, mmax),
            degree=degree,
            **kwargs,
        )


class BSplineRedshift(Base1DBSplineModel):
    """Class to construct a redshift B-Spline model

    Args:
        n_splines (int): number of degrees of freedom of basis, i.e. number of basis components
        z (array_like): redshift pe samples to evaluate the basis spline at
        z_inj (array_like): redshift injection samples to evalute the basis spline at
        dVdc (array_like): differential co-moving volume pe samples to evaluate the basis spline at.
        dVdc (array_like): differential co-moving volume injection samples to evaluate the basis spline at.
        knots (array_like, optional): array of knots, if non-uniform knot placing is preferred. Defaults to None.
        zmax (float, optional): maximum redshift value. basis spline will be truncated above this value. Defaults to 2.3.
        degree (int, optional): degree of the spline, i.e. cubcic splines = 3. Defaults to 3 (cubic spline).
        basis (class, optional): type of basis to use (ex. LogYBSpline). Defaults to Bspline.
    """

    def __init__(
        self,
        n_splines,
        z,
        z_inj,
        dVdc,
        dVdc_inj,
        knots=None,
        zmax=2.3,
        degree=3,
        basis=BSpline,
        **kwargs,
    ):
        super().__init__(
            n_splines,
            z,
            z_inj,
            knots=knots,
            xrange=(1e-4, zmax),
            degree=degree,
            basis=basis,
            **kwargs,
        )
        self.zmax = zmax
        self.dVcdzgrid = jnp.array(Planck15.differential_comoving_volume(np.linspace(1e-4, zmax, 2500)).value * 4 * np.pi)
        self.differential_comov_vols = [dVdc_inj, dVdc]
        self.zs = [z_inj, z]

    def eval_spline(self, bases, coefs):
        """given design matrix and coefficients, project coefficients onto the basis.

        Args:
            bases (array_like): design matrix of of spline
            coefs (array_like): basis spline coefficients

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        return self.interpolator.project(bases, coefs)

    def norm(self, coefs):
        """compute the normalization coefficient for the redshift basis-spline.

        Args:
            coefs (array_like): basis spline coefficients

        Returns:
            float: the redshift normalization coefficient.
        """
        return trapezoid(
            self.dVcdzgrid / (1 + self.grid) * jnp.einsum("i...,i->...", self.grid_bases, coefs),
            self.grid,
        )

    def __call__(self, coefs, pe_samples):
        """will evalute the projection of the coefficients along the design matrix evaluated at either posterior samples or injection samples.
            Use flag pe_samples to specify which type of samples are being evaluated (pe or injection).

        Args:
            coefs (array_like): basis spline coefficients
            pe_samples (bool, optional): If True, design matrix is evaluated along posterior samples. If False, design matrix is evaluated
                                        along injection samples. Defaults to True.

        Returns:
            array_like: The linear combination of the basis components evaluated at the posterior or injection samples given the coefficients.
        """

        return (
            self.funcs[1](coefs) * self.differential_comov_vols[1] / (1 + self.zs[1])
            if pe_samples
            else self.funcs[0](coefs) * self.differential_comov_vols[0] / (1 + self.zs[0])
        )
