"""
a module that stores 2D joint population models constructed from bsplines
"""

import jax.numpy as jnp

from ...interpolation import RectBivariateBasisSpline, BSpline


class Base2DBSplineModel(object):
    def __init__(
        self,
        xnknots,
        ynknots,
        xx,
        yy,
        xx_inj,
        yy_inj,
        xorder = 3,
        yorder = 3,
        xrange=(0, 1),
        yrange=(0, 1),
        xbasis = BSpline,
        ybasis = BSpline,
        basis=RectBivariateBasisSpline,
        **kwargs,
    ):
        self.xknots = xnknots
        self.yknots = ynknots
        self.xmin, self.xmax = xrange
        self.ymin, self.ymax = yrange
        self.interpolator = basis(xnknots, ynknots, xrange=xrange, yrange=yrange, xbasis=xbasis, ybasis=ybasis, kx=xorder, ky=yorder, **kwargs)
        self.pe_design_matrix = jnp.array(self.interpolator.bases(xx, yy))
        self.inj_design_matrix = jnp.array(self.interpolator.bases(xx_inj, yy_inj))
        self.funcs = [self.inj_pdf, self.pe_pdf]

    def eval_spline(self, bases, coefs):
        return self.interpolator.project(bases, coefs)

    def pe_pdf(self, coefs):
        return self.eval_spline(self.pe_design_matrix, coefs)

    def inj_pdf(self, coefs):
        return self.eval_spline(self.inj_design_matrix, coefs)

    def __call__(self, coefs, pe_samples = True):
        return self.funcs[1](coefs) if pe_samples else self.funcs[0](coefs)


class BSplineJointMassRatioChiEffective(Base2DBSplineModel):
    def __init__(
        self,
        chiknots,
        qknots,
        chieff,
        q,
        chieff_inj,
        q_inj,
        **kwargs,
    ):
        super().__init__(
            xnknots=chiknots,
            ynknots=qknots,
            xx=chieff,
            yy=q,
            xx_inj=chieff_inj,
            yy_inj=q_inj,
            xrange=(-1, 1),
            yrange=(0, 1),
            **kwargs,
        )
class BSplineJointMassRedshift(Base2DBSplineModel):
    def __init__(
            nknots_m,
            nknots_z,
            m1,
            z,
            m1_inj,
            z_inj,
            mmin=3.,
            mmax=100.,
            order_m=3,
            order_z=3,
            basis_m=BSpline,
            basis_z=BSpline,
            **kwargs,
    ):
        super().__init__(
            nknots_m,
            nknots_z,
            m1,
            z,
            m1_inj,
            z_inj,
            xorder = order_m,
            yorder = order_z,
            xrange = (mmin, mmax),
            yrange = (0, 2),
            xbasis = basis_m,
            ybasis = basis_z,
        )