import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

from .single import Base1DBSplineModel_IJR

class BivariateBSpline():

    def __init__(self,
                 u_domain, v_domain,
                 u_order = int, v_order = int,
                 u_l = int, v_l = int,
                 u_knots=None, v_knots=None,
                 normalization=False):
        """Class to construct the 2D B-spline design tensor.

        Args:
            u_domain (array-like): lower and upper values for the u domain of interest
            v_domain (array-like): lower and upper values for the v domain of interest
            u_order (int): order of B-spline in the u-direction
            v_order (int): order of B-spline in the v-direction
            u_l (int): u_l+1 number of B-splines
            v_l (int): v_l+1 number of B-splines
            u_knots (array-like, optional): knot vector in the u-direction (both exterior and interior knots)
            v_knots (array-like, optional): knot vector in the v-direction (both exterior and interior knots)
            normalization (bool, optional): flag to numerically normalize the spline.
            independent (bool, optional): flag to construct 2D B-splines with all points (True), or pairs of points (False)
        """
        self.domain = np.array([u_domain, v_domain])
        self.ls = np.array([u_l, v_l])
        self.orders = np.array([u_order, v_order])
        self.degrees = np.array([u_order - 1, v_order - 1])
        self.normalization = normalization
        self.u_BSpline = Base1DBSplineModel_IJR(u_domain = self.domain[0], P = self.orders[0], l = self.ls[0], knots = u_knots, normalization=self.normalization)
        self.v_BSpline = Base1DBSplineModel_IJR(u_domain = self.domain[1], P = self.orders[1], l = self.ls[1], knots = v_knots, normalization=self.normalization)

    def design_tensor(self, u_domain, v_domain, independent=True):
        u_Bspline_dm = self.u_BSpline.design_matrix(u_domain)
        v_Bspline_dm = self.v_BSpline.design_matrix(v_domain)
        if independent:
            return np.tensordot(u_Bspline_dm, v_Bspline_dm, axes = 0)
        else:
            uv_Bspline = np.empty((self.u_BSpline.l_E + 1, self.v_BSpline.l_E + 1, u_Bspline_dm.shape[1]))
            # TODO: Figure out how to not use a nested for-loop to perform this operation 
            for s in range(self.v_BSpline.l_E + 1):
                for k in range(self.u_BSpline.l_E + 1):
                    uv_Bspline[k,s,:] = u_Bspline_dm[k,:] * v_Bspline_dm[s,:]
            return uv_Bspline
    
    def spline(self, coeffs, design_tensor, independent=True):
        if independent:
            assert coeffs.shape[0] == design_tensor.shape[0] and coeffs.shape[1] == design_tensor.shape[2], "The number of coefficients must match the number of B-splines."
            return jnp.einsum('kisj,ks->ij', design_tensor, coeffs)
        else:
            assert coeffs.shape[0] == design_tensor.shape[0] and coeffs.shape[1] == design_tensor.shape[1], "The number of coefficients must match the number of B-splines."
            return jnp.einsum('ksj,ks->j', design_tensor, coeffs)
        