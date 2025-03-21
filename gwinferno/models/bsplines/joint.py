import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

from ...interpolation import BivariateBSpline
from .single import Base1DBSplineModel_IJR        

class Base2DBSplineModel(BivariateBSpline):

    def __init__(self, u_domain=(0.0, 1.0), v_domain=(0.0, 1.0),
        u_pe_vals, u_inj_vals,
        v_pe_vals, v_inj_vals,
        u_order=4, v_order=4,
        ul_BSplines, vl_BSplines,
        # TODO: change variable name ```basis``` to something else -- this arg is already being used for the Base1D
        basis=Base1DBSplineModel_IJR,
        **kwargs,):
        """
        Args:
            u_pe_vals, v_pe_vals (array_like): parameter estimation samples for basis evaluation
            u_inj_vals, v_inj_vals (array-like): injection samples for basis evaluation
            ul_Bsplines, vl_BSplines (int): total number of basis functions - 1
            u_domain, v_domain (tuple, default=(0.0, 1.0)): domain of the B-splines
            u_order, v_order (int, default=4): order of the B-splines, i.e., `4` for cubic splines
            basis (class, default=BivariateBSpline): type of basis to use
        """
        self.domain = jnp.array([u_domain, v_domain])
        self.ls = jnp.array([ul_BSplines, vl_BSplines])
        self.orders = jnp.array([u_order, v_order])
        self.degrees = jnp.array([u_order - 1, v_order - 1])
        self.l_Es = jnp.array([self.ls[0] + 2*self.degrees[0], self.ls[1] + 2*self.degrees[1]])
        self.u_model = basis(pe_vals = u_pe_vals, inj_vals = u_inj_vals, 
            l_BSplines = ul_BSplines, domain=u_domain, order=u_order, **kwargs,)
        self.v_model = basis(pe_vals = v_pe_vals, inj_vals = v_inj_vals, 
            l_BSplines = vl_BSplines, domain=v_domain, order=v_order, **kwargs,)

    # Because you are not using the BivariateBSpline as an interpolator, you must construct the design tensor in this class
    # However, you can use spline from BivariateBSpline to construct the spline from the design tensor!
    # TODO: Eventually, you want to use the interpolator and modify as needed.
    def design_tensor(self, independent=False, **kwargs,):
        u_pe_dm = self.u_model.pe_design_matrix
        u_inj_dm = self.u_model.inj_design_matrix
        v_pe_dm = self.v_model.pe_design_matrix
        v_inj_dm = self.v_model.inj_design_matrix
        if independent:
            pe_dt =  jnp.tensordot(u_pe_dm, v_pe_dm, axes = 0)
            inj_dt = jnp.tensordot(u_inj_dm, v_inj_dm, axes = 0)
            return pe_dt, inj_dt
        else:
            pe_dt = jnp.empty((self.l_Es[0] + 1, self.l_Es[1] + 1, u_pe_dm.shape[1]))
            inj_dt = jnp.empty((self.l_Es[0] + 1, self.l_Es[1] + 1, u_inj_dm.shape[1]))
            # TODO: Figure out how to not use a nested for-loop to perform this operation 
            for s in range(self.l_Es[1] + 1):
                for k in range(self.l_Es[0] + 1):
                    pe_dt = pe_dt.at[k,s,:].set(u_pe_dm[k,:] * v_pe_dm[s,:])
                    inj_dt = inj_dt.at[k,s,:].set(u_inj_dm[k,:] * v_inj_dm[s,:])
            return pe_dt, inj_dt

    def eval_spline(self, coefs, design_tensor, **kwargs,):
        return self.spline(coefs, design_tensor, **kwargs)

    def pe_pdf(self, coefs):
        return self.eval_spline(coefs, self.design_tensor[0])

    def inj_pdf(self, coefs):
        return self.eval_spline(coefs, self.design_tensor[1])

    def __call__(self, coefs, pe_samples = True):
        return self.funcs[1](coefs) if pe_samples else self.funcs[0](coefs)