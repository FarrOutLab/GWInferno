"""
a module that stores 2D joint population models constructed from bsplines
"""

import jax.numpy as jnp

from ...interpolation import RectBivariateBasisSpline


class Base2DBSplineModel(object):
    def __init__(
        self,
        xnknots,
        ynknots,
        xx,
        yy,
        xx_inj,
        yy_inj,
        xrange=(0, 1),
        yrange=(0, 1),
        basis=RectBivariateBasisSpline,
        **kwargs,
    ):
        self.xknots = xnknots
        self.yknots = ynknots
        self.xmin, self.xmax = xrange
        self.ymin, self.ymax = yrange
        self.interpolator = basis(xnknots, ynknots, xrange=xrange, yrange=yrange, **kwargs)
        self.pe_design_matrix = jnp.array(self.interpolator.bases(xx, yy))
        self.inj_design_matrix = jnp.array(self.interpolator.bases(xx_inj, yy_inj))
        self.funcs = [self.inj_pdf, self.pe_pdf]

    def eval_spline(self, bases, coefs):
        return self.interpolator.project(bases, coefs)

    def pe_pdf(self, coefs):
        return self.eval_spline(self.pe_design_matrix, coefs)

    def inj_pdf(self, coefs):
        return self.eval_spline(self.inj_design_matrix, coefs)

    def __call__(self, ndim, coefs):
        return self.funcs[ndim - 1](coefs)


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
