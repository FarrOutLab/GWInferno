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
    def __init__(
        self,
        nknots,
        a1,
        a2,
        a1_inj,
        a2_inj,
        knots=None,
        order=3,
        prefix="c",
        **kwargs,
    ):
        self.primary_model = BSplineSpinMagnitude(
            nknots=nknots,
            a=a1,
            a_inj=a1_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="a_1",
            **kwargs,
        )
        self.secondary_model = BSplineSpinMagnitude(
            nknots=nknots,
            a=a2,
            a_inj=a2_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="a_2",
            **kwargs,
        )

    def __call__(self, ndim, coefs):
        p_a1 = self.primary_model(ndim, coefs)
        p_a2 = self.secondary_model(ndim, coefs)
        return p_a1 * p_a2


class BSplineIndependentSpinMagnitudes(object):
    def __init__(
        self,
        nknots1,
        nknots2,
        a1,
        a2,
        a1_inj,
        a2_inj,
        knots1=None,
        order1=3,
        prefix1="c",
        knots2=None,
        order2=3,
        prefix2="w",
        **kwargs,
    ):
        self.primary_model = BSplineSpinMagnitude(
            nknots=nknots1,
            a=a1,
            a_inj=a1_inj,
            knots=knots1,
            order=order1,
            prefix=prefix1,
            domain="a_1",
            **kwargs,
        )
        self.secondary_model = BSplineSpinMagnitude(
            nknots=nknots2,
            a=a2,
            a_inj=a2_inj,
            knots=knots2,
            order=order2,
            prefix=prefix2,
            domain="a_2",
            **kwargs,
        )

    def __call__(self, ndim, pcoefs, scoefs):
        p_a1 = self.primary_model(ndim, pcoefs)
        p_a2 = self.secondary_model(ndim, scoefs)
        return p_a1 * p_a2


class BSplineIIDSpinTilts(object):
    def __init__(
        self,
        nknots,
        ct1,
        ct2,
        ct1_inj,
        ct2_inj,
        knots=None,
        order=3,
        prefix="x",
        **kwargs,
    ):
        self.primary_model = BSplineSpinTilt(
            nknots=nknots,
            ct=ct1,
            ct_inj=ct1_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="cos_tilt_1",
            **kwargs,
        )
        self.secondary_model = BSplineSpinTilt(
            nknots=nknots,
            ct=ct2,
            ct_inj=ct2_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="cos_tilt_2",
            **kwargs,
        )

    def __call__(self, ndim, coefs):
        p_ct1 = self.primary_model(ndim, coefs)
        p_ct2 = self.secondary_model(ndim, coefs)
        return p_ct1 * p_ct2


class BSplineIndependentSpinTilts(object):
    def __init__(
        self,
        nknots1,
        nknots2,
        ct1,
        ct2,
        ct1_inj,
        ct2_inj,
        knots1=None,
        order1=3,
        prefix1="x",
        knots2=None,
        order2=3,
        prefix2="z",
        **kwargs,
    ):
        self.primary_model = BSplineSpinTilt(
            nknots=nknots1,
            ct=ct1,
            ct_inj=ct1_inj,
            knots=knots1,
            order=order1,
            prefix=prefix1,
            domain="cos_tilt_1",
            **kwargs,
        )
        self.secondary_model = BSplineSpinTilt(
            nknots=nknots2,
            ct=ct2,
            ct_inj=ct2_inj,
            knots=knots2,
            order=order2,
            prefix=prefix2,
            domain="cos_tilt_2",
            **kwargs,
        )

    def __call__(self, ndim, pcoefs, scoefs):
        p_ct1 = self.primary_model(ndim, pcoefs)
        p_ct2 = self.secondary_model(ndim, scoefs)
        return p_ct1 * p_ct2


class BSplinePrimaryPowerlawRatio(object):
    def __init__(
        self,
        nknots,
        m1,
        m1_inj,
        mmin=2,
        mmax=100,
        knots=None,
        order=3,
        prefix="c",
        basis=BSpline,
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            nknots,
            m1,
            m1_inj,
            knots=knots,
            mmin=mmin,
            mmax=mmax,
            order=order,
            prefix=prefix,
            domain="mass_1",
            basis=basis,
            **kwargs,
        )

    def __call__(self, m1, q, beta, mmin, coefs):
        # p_m1 = jnp.where(jnp.greater_equal(m1, mmin), self.primary_model(len(m1.shape), coefs), 0)
        # norm_factor = self.primary_model.norm_mmin_cut(mmin, coefs)
        # p_m1 = p_m1 / norm_factor
        p_m1 = self.primary_model(len(m1.shape), coefs)
        p_q = powerlaw_pdf(q, beta, mmin / m1, 1)
        return p_m1 * p_q


class PLPeakPrimaryBSplineRatio(object):
    def __init__(
        self,
        nknots,
        q,
        q_inj,
        m1,
        m1_inj,
        knots=None,
        order=3,
        prefix="q",
    ):
        self.ratio_model = BSplineRatio(
            nknots,
            q,
            q_inj,
            m1,
            m1_inj,
            knots=knots,
            order=order,
            prefix=prefix,
        )

    def __call__(self, m1, ndim, coefs, **kwargs):
        p_q = self.ratio_model(ndim, coefs)
        p_m1 = plpeak_primary_pdf(m1, **kwargs)
        return p_m1 * p_q


class BSplinePrimaryBSplineRatio(object):
    def __init__(
        self,
        nknots_m,
        nknots_q,
        m1,
        m1_inj,
        q,
        q_inj,
        knots_m=None,
        knots_q=None,
        order_m=3,
        order_q=3,
        prefix_m="c",
        prefix_q="q",
        m1min=3.0,
        m2min=3.0,
        mmax=100.0,
        basis_m=BSpline,
        basis_q=BSpline,
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            nknots_m,
            m1,
            m1_inj,
            knots=knots_m,
            mmin=m1min,
            mmax=mmax,
            order=order_m,
            prefix=prefix_m,
            domain="mass_1",
            basis=basis_m,
            **kwargs,
        )
        self.ratio_model = BSplineRatio(
            nknots_q,
            q,
            q_inj,
            qmin=m2min / mmax,
            knots=knots_q,
            order=order_q,
            prefix=prefix_q,
            basis=basis_q,
            **kwargs,
        )

    def __call__(self, ndim, mcoefs, qcoefs):
        return self.ratio_model(ndim, qcoefs) * self.primary_model(ndim, mcoefs)


class BSplineIIDComponentMasses(object):
    def __init__(
        self,
        nknots,
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
            nknots=nknots,
            m=m1,
            m_inj=m1_inj,
            knots=knots,
            mmin=mmin,
            mmax=mmax,
            **kwargs,
        )
        self.secondary_model = BSplineMass(
            nknots=nknots,
            m=m2,
            m_inj=m2_inj,
            knots=knots,
            mmin=mmin,
            mmax=mmax,
            **kwargs,
        )
        self.qs = [m2_inj / m1_inj, m2 / m1]

    def __call__(self, ndim, coefs, beta=0):
        p_m1 = self.primary_model(ndim, coefs)
        p_m2 = self.secondary_model(ndim, coefs)
        return jnp.where(
            jnp.less(self.qs[ndim - 1], 0) | jnp.greater(self.qs[ndim - 1], 1),
            0,
            p_m1 * p_m2,
        ) * jnp.power(self.qs[ndim - 1], beta)


class BSplineIndependentComponentMasses(object):
    def __init__(
        self,
        nknots1,
        nknots2,
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
        prefix1="c",
        order1=3,
        prefix2="w",
        order2=3,
    ):
        self.primary_model = BSplineMass(
            nknots=nknots1,
            m=m1,
            m_inj=m1_inj,
            knots=knots1,
            mmin=mmin1,
            mmax=mmax1,
            prefix=prefix1,
            order=order1,
        )
        self.secondary_model = BSplineMass(
            nknots=nknots2,
            m=m2,
            m_inj=m2_inj,
            knots=knots2,
            mmin=mmin2,
            mmax=mmax2,
            prefix=prefix2,
            order=order2,
        )

    def __call__(self, m1, m2, ndim, pcoefs, scoefs, beta):
        p_m1 = self.primary_model(ndim, pcoefs)
        p_m2 = self.secondary_model(ndim, scoefs)
        return p_m1 * p_m2 * (m1 / m2) ** beta


class BSplineEffectiveSpinDims(object):
    def __init__(
        self,
        nknotse,
        nknotsp,
        chieff,
        chip,
        chieff_inj,
        chip_inj,
        knotse=None,
        knotsp=None,
        prefixe="c",
        ordere=3,
        prefixp="w",
        orderp=3,
    ):
        self.chi_eff_model = BSplineChiEffective(
            nknotse,
            chieff,
            chieff_inj,
            knots=knotse,
            order=ordere,
            prefix=prefixe,
        )

        self.chi_p_model = BSplineChiPrecess(
            nknotsp,
            chip,
            chip_inj,
            knots=knotsp,
            order=orderp,
            prefix=prefixp,
        )

    def __call__(self, ndim, ecoefs, pcoefs):
        p_chieff = self.chi_eff_model(ndim, ecoefs)
        p_chip = self.chi_p_model(ndim, pcoefs)
        return p_chieff * p_chip
