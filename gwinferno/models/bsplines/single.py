import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck15

from ...interpolation import BSpline


class Base1DBSplineModel(object):
    def __init__(
        self,
        nknots,
        xx,
        xx_inj,
        knots=None,
        xrange=(0, 1),
        order=3,
        prefix="c",
        domain="x",
        basis=BSpline,
        **kwargs,
    ):
        self.nknots = nknots
        self.domain = domain
        self.xmin, self.xmax = xrange
        self.order = order
        self.prefix = prefix
        self.interpolator = basis(
            nknots,
            knots=knots,
            xrange=xrange,
            k=order + 1,
            **kwargs,
        )
        self.variable_names = [f"{self.prefix}{i}" for i in range(self.nknots)]
        self.pe_design_matrix = jnp.array(self.truncate_dmat(xx, self.interpolator.bases(xx)))
        self.inj_design_matrix = jnp.array(self.truncate_dmat(xx_inj, self.interpolator.bases(xx_inj)))
        self.funcs = [self.inj_pdf, self.pe_pdf]

    def truncate_dmat(self, x, dmat):
        return jnp.where(jnp.less(x, self.xmin) | jnp.greater(x, self.xmax), 0, dmat)

    def eval_spline(self, bases, coefs):
        return self.interpolator.project(bases, coefs)

    def pe_pdf(self, coefs):
        return self.eval_spline(self.pe_design_matrix, coefs)

    def inj_pdf(self, coefs):
        return self.eval_spline(self.inj_design_matrix, coefs)

    def __call__(self, ndim, coefs):
        return self.funcs[ndim - 1](coefs)


class BSplineSpinMagnitude(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        a,
        a_inj,
        knots=None,
        order=3,
        prefix="c",
        domain="a",
        **kwargs,
    ):
        super().__init__(
            nknots,
            a,
            a_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            **kwargs,
        )


class BSplineSpinTilt(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        ct,
        ct_inj,
        knots=None,
        order=3,
        prefix="x",
        domain="cos_tilt",
        **kwargs,
    ):
        super().__init__(
            nknots,
            ct,
            ct_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            xrange=(-1, 1),
            **kwargs,
        )


class BSplineChiEffective(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        chieff,
        chieff_inj,
        knots=None,
        order=3,
        prefix="x",
        domain="chi_eff",
        **kwargs,
    ):
        super().__init__(
            nknots,
            chieff,
            chieff_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            xrange=(-1, 1),
            **kwargs,
        )


class BSplineSymmetricChiEffective(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        chieff,
        chieff_inj,
        knots=None,
        order=3,
        prefix="x",
        domain="chi_eff",
        **kwargs,
    ):
        super().__init__(
            nknots,
            jnp.abs(chieff),
            jnp.abs(chieff_inj),
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            xrange=(0, 1),
            **kwargs,
        )

    def __call__(self, ndim, coefs):
        return 0.5 * self.funcs[ndim - 1](coefs)


class BSplineChiPrecess(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        chip,
        chip_inj,
        knots=None,
        order=3,
        prefix="w",
        domain="chi_p",
        **kwargs,
    ):
        super().__init__(
            nknots,
            chip,
            chip_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            **kwargs,
        )


class BSplineRatio(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        q,
        q_inj,
        qmin=0,
        knots=None,
        order=3,
        prefix="u",
        **kwargs,
    ):
        super().__init__(
            nknots,
            q,
            q_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            xrange=(qmin, 1),
            domain="mass_ratio",
            **kwargs,
        )


class BSplineMass(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        m,
        m_inj,
        knots=None,
        mmin=2,
        mmax=100,
        order=3,
        prefix="f",
        domain="mass",
        **kwargs,
    ):
        super().__init__(
            nknots,
            m,
            m_inj,
            knots=knots,
            xrange=(mmin, mmax),
            order=order,
            prefix=prefix,
            domain=domain,
            **kwargs,
        )


class BSplineRedshift(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        z,
        z_inj,
        dVdc,
        dVdc_inj,
        knots=None,
        zmax=2.3,
        order=3,
        prefix="u",
        domain="redshift",
        basis=BSpline,
        **kwargs,
    ):
        super().__init__(
            nknots,
            z,
            z_inj,
            knots=knots,
            xrange=(1e-4, zmax),
            order=order,
            prefix=prefix,
            domain=domain,
            basis=basis,
            **kwargs,
        )
        self.zmax = zmax
        self.dVcdzgrid = jnp.array(Planck15.differential_comoving_volume(np.linspace(1e-4, zmax, 2500)).value * 4 * np.pi)
        self.differential_comov_vols = [dVdc_inj, dVdc]
        self.zs = [z_inj, z]

    def eval_spline(self, bases, coefs):
        return self.interpolator.project(bases, coefs)

    def norm(self, coefs):
        return jnp.trapz(
            self.dVcdzgrid / (1 + self.grid) * jnp.einsum("i...,i->...", self.grid_bases, coefs),
            self.grid,
        )

    def __call__(self, ndim, coefs):
        return self.funcs[ndim - 1](coefs) * self.differential_comov_vols[ndim - 1] / (1 + self.zs[ndim - 1])
