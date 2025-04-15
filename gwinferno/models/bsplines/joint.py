import jax.numpy as jnp

from ...interpolation import BivariateBSpline
from ...interpolation import SplineDOF_wrapper
from .single import Base1DBSplineModel_IJR     
   

class Base2DBSplineModel(BivariateBSpline):

    def __init__(self, u_pe_vals=None, u_inj_vals=None,
        v_pe_vals=None, v_inj_vals=None,
        u_domain=(0.0, 1.0), v_domain=(0.0, 1.0),
        u_order=4, v_order=4,
        ul_BSplines=None, vl_BSplines=None,
        # TODO: change variable name ```basis``` to something else -- this arg is already being used for the Base1D
        basis=Base1DBSplineModel_IJR,
        book = True,
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
        self.ls = (SplineDOF_wrapper(n_splines = ul_BSplines, degree = u_order - 1), SplineDOF_wrapper(n_splines = vl_BSplines, degree = v_order - 1)) if book else (ul_BSplines, vl_BSplines)
        # self.ls = jnp.array([ul_BSplines, vl_BSplines])
        self.orders = (u_order, v_order)
        self.degrees = (u_order - 1, v_order - 1)
        self.l_Es = (self.ls[0] + 2*self.degrees[0], self.ls[1] + 2*self.degrees[1])
        self.u_model = basis(pe_vals = u_pe_vals, inj_vals = u_inj_vals, 
            l_BSplines = self.ls[0], domain=u_domain, order=u_order, **kwargs,)
        self.v_model = basis(pe_vals = v_pe_vals, inj_vals = v_inj_vals, 
            l_BSplines = self.ls[1], domain=v_domain, order=v_order, **kwargs,)
        self.u_pe_dm = self.u_model.pe_design_matrix
        self.v_pe_dm = self.v_model.pe_design_matrix
        self.u_inj_dm = self.u_model.inj_design_matrix
        self.v_inj_dm = self.v_model.inj_design_matrix
        self.funcs = [self.inj_pdf, self.pe_pdf]
        print(self.u_model.interpolator)

    # Because you are not using the BivariateBSpline as an interpolator, you must construct the design tensor in this class
    # However, you can use spline from BivariateBSpline to construct the spline from the design tensor!
    # TODO: Eventually, you want to use the interpolator and modify as needed.
    def _pe_design_tensor(self,  independent=False, **kwargs,):
        # u_pe_dm = self.u_model.pe_design_matrix
        # v_pe_dm = self.v_model.pe_design_matrix
        # print('u_pe_dm shape ', self.u_pe_dm.shape)
        if independent:
            pe_dt =  jnp.tensordot(self.u_pe_dm, self.v_pe_dm, axes = 0)
            return pe_dt
        else:
            # print('pre-pe dt shap:', (self.l_Es[0] + 1, self.l_Es[1] + 1, self.u_pe_dm.shape[1], self.u_pe_dm.shape[2]))
            pe_dt = jnp.zeros((self.l_Es[0] + 1, self.l_Es[1] + 1, self.u_pe_dm.shape[1], self.u_pe_dm.shape[2]))
            # print('pe dt shape ', pe_dt.shape)
            # TODO: Figure out how to not use a nested for-loop to perform this operation 
            for s in range(self.l_Es[1] + 1):
                for k in range(self.l_Es[0] + 1):
                    pe_dt = pe_dt.at[k,s].set(self.u_pe_dm[k] * self.v_pe_dm[s])
            return pe_dt
        
    def _inj_design_tensor(self, independent=False, **kwargs,):
        # u_inj_dm = self.u_model.inj_design_matrix
        # v_inj_dm = self.v_model.inj_design_matrix
        # print('u_inj_dm shape ', self.u_inj_dm.shape)
        if independent:
            inj_dt = jnp.tensordot(self.u_inj_dm, self.v_inj_dm, axes = 0)
            return inj_dt
        else:
            inj_dt = jnp.zeros((self.l_Es[0] + 1, self.l_Es[1] + 1, self.u_inj_dm.shape[1]))
            # print('inj dt shape: ', inj_dt.shape)
            for s in range(self.l_Es[1] + 1):
                for k in range(self.l_Es[0] + 1):
                    inj_dt = inj_dt.at[k,s].set(self.u_inj_dm[k] * self.v_inj_dm[s])
            return inj_dt

    # TODO: too many independent args, reduce this!
    def eval_spline(self, coefs, design_tensor, independent = False):
        return self.spline(coefs, design_tensor, independent = independent)

    def pe_pdf(self, coefs, independent = False):
        return self.eval_spline(coefs, self._pe_design_tensor(independent=independent), independent=independent)

    def inj_pdf(self, coefs, independent = False):
        return self.eval_spline(coefs, self._inj_design_tensor(independent=independent), independent=independent)

    def __call__(self, coefs, pe_samples = True, independent = False):
        return self.funcs[1](coefs, independent=independent) if pe_samples else self.funcs[0](coefs, independent=independent)