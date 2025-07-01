"""
a module for interpolation calculations using jax
"""

import jax.numpy as jnp
import numpy as np
from jax.scipy.integrate import trapezoid
from jax.tree_util import register_pytree_node_class

OOB_VAL = -9  # Value to return for out-of-bounds values.  nan or -inf breaks autograd, so we use this instead.


@register_pytree_node_class
class NaturalCubicUnivariateSpline(object):
    """
    Minimal port of Scipy's UnivariateSpline to JAX -- restricted to only cubic splines with the "natural"
    boundary conditions (based from implementation in jax_cosmo.scipy.interpolate)
    """

    def __init__(self, x, y, coefficients=None):
        k, x, y = 3, jnp.atleast_1d(x), jnp.atleast_1d(y)
        assert len(x) == len(y), "Input arrays must be the same length."
        assert x.ndim == 1 and y.ndim == 1, "Input arrays must be 1D."
        n_data = len(x)
        h = jnp.diff(x)
        p = jnp.diff(y)
        if coefficients is None:
            assert n_data > 3, "Not enough input points for cubic spline."
            zero = jnp.array([0.0])
            one = jnp.array([1.0])
            A00, A01, A02, ANN, AN1, AN2 = one, zero, zero, one, (-one), zero
            A = jnp.diag(jnp.concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
            upper_diag1 = jnp.diag(jnp.concatenate((A01, h[1:])), k=1)
            upper_diag2 = jnp.diag(jnp.concatenate((A02, jnp.zeros(n_data - 3))), k=2)
            lower_diag1 = jnp.diag(jnp.concatenate((h[:-1], AN1)), k=-1)
            lower_diag2 = jnp.diag(jnp.concatenate((jnp.zeros(n_data - 3), AN2)), k=-2)
            A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2
            center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
            s = jnp.concatenate((zero, center, zero))
            coefficients = jnp.linalg.solve(A, s)
        self.k, self._x, self._y = k, x, y
        self._coefficients = coefficients

    def tree_flatten(self):
        children = (self._x, self._y, self._coefficients)
        return children

    @classmethod
    def tree_unflatten(cls, children):
        x, y, coefs = children
        return cls(x, y, coefs)

    def __call__(self, x):
        t, a, b, c, d = self._compute_coeffs(x)
        return a + b * t + c * t**2 + d * t**3

    def _compute_coeffs(self, xs):
        knots, y, coefficients = self._x, self._y, self._coefficients
        ind = jnp.digitize(xs, knots) - 1
        ind = jnp.clip(ind, 0, len(knots) - 2)
        t = xs - knots[ind]
        h = jnp.diff(knots)[ind]
        c = coefficients[ind]
        c1 = coefficients[ind + 1]
        a = y[ind]
        a1 = y[ind + 1]
        b = (a1 - a) / h - (2 * c + c1) * h / 3.0
        d = (c1 - c) / (3 * h)
        return (t, a, b, c, d)


class BasisSpline(object):
    def __init__(
        self,
        n_df,
        knots=None,
        interior_knots=None,
        xrange=(0, 1),
        k=4,
        normalize=True,
    ):
        """
        Class to construct a basis spline (with the M-Spline basis)

        Args:
            n_df (int): number of degrees of freedom for the spline
            knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
            interior_knots (array_like, optional): array of interior knots,
                if non-uniform knot placing is preferred. Defaults to None.
            xrange (tuple, optional): domain of spline. Defaults to (0, 1).
            k (int, optional): order of the spline +1, i.e. cubic splines->k=4. Defaults to 4 (cubic spline).
            normalize (bool, optional): flag whether or not to numerically normalize the spline. Defaults to True.
        """
        self.order = k
        self.N = n_df
        self.xrange = xrange
        if knots is None:
            if interior_knots is None:
                interior_knots = np.linspace(xrange[0], xrange[1], n_df - k + 2)
            dx = interior_knots[1] - interior_knots[0]
            knots = jnp.linspace(xrange[0] - dx * (k - 1), xrange[1] + dx * (k - 1), len(interior_knots) + (k - 1) * 2)

        self.knots = knots
        self.interior_knots = interior_knots
        assert len(self.knots) == self.N + self.order

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            self.grid = jnp.linspace(*xrange, 1000)
            self.grid_bases = jnp.array(self.bases(self.grid))
            self.basis_vols = jnp.array([trapezoid(self.grid_bases[i, :], self.grid) for i in range(self.N)])

    def norm(self, coefs):
        """
        norm numerically normalizes the spline

        Args:
            coefs (array_like): coefficients for the basis components

        Returns:
            float: the normalization factor given the coefficients
        """
        n = 1.0 / jnp.sum(self.basis_vols * coefs.flatten()) if self.normalize else 1.0
        return n

    def _basis(self, xs, i, k):
        """
        _basis protected method that computes the ith basis component of order k recursively using
            the Cox-de Boor recursion relation

        Args:
            xs (array_like): input values to evaluate the basis spline at
            i (int): the ith basis component
            k (int): order of the spline

        Returns:
            array_like: the ith basis component of order k evaluated at xs
        """
        if self.knots[i + k] - self.knots[i] < 1e-6:
            return np.zeros_like(xs)
        elif k == 1:
            v = np.zeros_like(xs)
            v[(xs >= self.knots[i]) & (xs < self.knots[i + 1])] = 1 / (self.knots[i + 1] - self.knots[i])
            return v
        else:
            v = (xs - self.knots[i]) * self._basis(xs, i, k - 1) + (self.knots[i + k] - xs) * self._basis(xs, i + 1, k - 1)
            return (v * k) / ((k - 1) * (self.knots[i + k] - self.knots[i]))

    def _bases(self, xs):
        """
        _bases construct the set of basis components

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            list: list of all N basis components evaluated at xs
        """
        return [self._basis(xs, i, k=self.order) for i in range(self.N)]

    def bases(self, xs):
        """
        bases form the basis spline design matrix evaluated at xs.  All basis values outside the range
        of the spline are set to zero.

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            array_like: the design matrix evaluated at xs. shape (N, *xs.shape)
        """
        design_matrix = jnp.concatenate(self._bases(xs)).reshape(self.N, *xs.shape)
        return jnp.where(jnp.less(xs, self.xrange[0]) | jnp.greater(xs, self.xrange[1]), 0.0, design_matrix)

    def get_coefficients(self, xs, ys):
        """
        computes the coefficients of the basis components given data 1-D data (xs, ys)

        Args:
            xs (array_like): The x values of data
            ys (array_like): The y values of data

        Returns:
            the coefficients (array_like), interpolated y-values evaluated at xs (array_like),
            the design matrix evaluated at xs with shape (N, *xs.shape)
        """

        design_matrix = jnp.transpose(self.bases(xs))
        BtBi = jnp.linalg.inv(jnp.matmul(jnp.transpose(design_matrix), design_matrix))
        alpha = jnp.matmul(jnp.matmul(BtBi, jnp.transpose(design_matrix)), ys)
        return alpha, np.einsum("ji,i->j", design_matrix, alpha), design_matrix

    def project(self, bases, coefs):
        """
        project given a design matrix (or bases) and coefficients, project the coefficients onto the spline

        Args:
            bases (array_like): The set of basis components or design matrix to project onto
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        coefs /= jnp.sum(coefs)
        return jnp.einsum("i...,i->...", bases, coefs) * self.norm(coefs)

    def eval(self, xs, coefs):
        """
        eval Evaluate basis spline at xs given coefficients

        Args:
            xs (array_like): input values to evaluate the basis spline at
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components evaluated at xs given the coefficients
        """
        return self.project(self.bases(xs), coefs)

    def __call__(self, xs, coefs):
        """
        __call__ Evaluate basis spline at xs given coefficients

        Args:
            xs (array_like): input values to evaluate the basis spline at
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components evaluated at xs given the coefficients
        """
        return self.eval(xs, coefs)


class BSpline(BasisSpline):
    def __init__(
        self,
        n_df,
        knots=None,
        interior_knots=None,
        xrange=(0, 1),
        k=4,
        normalize=False,
    ):
        """
        Class to construct a basis spline (B-Spline)

        Args:
            n_df (int): number of degrees of freedom for the spline
            knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
            interior_knots (array_like, optional): array of interior knots,
                if non-uniform knot placing is preferred. Defaults to None.
            xrange (tuple, optional): domain of spline. Defaults to (0, 1).
            k (int, optional): order of the spline +1, i.e. cubic splines->k=4. Defaults to 4 (cubic spline).
            normalize (bool, optional): flag whether or not to numerically normalize the spline. Defaults to False.
        """
        super().__init__(
            n_df=n_df,
            knots=knots,
            interior_knots=interior_knots,
            xrange=xrange,
            k=k,
            normalize=normalize,
        )

    def _bases(self, xs):
        """
        _bases construct the set of basis components for the canonical B-Spline with the Cox-de Boor recursion relation

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            list: list of all N basis components evaluated at xs
        """
        return [(self.knots[i + self.order] - self.knots[i]) / self.order * self._basis(xs, i, k=self.order) for i in range(self.N)]

    def norm(self, coefs):
        """
        norm numerically normalizes the spline

        Args:
            coefs (array_like): coefficients for the basis components

        Returns:
            float: the normalization factor given the coefficients
        """
        n = 1.0 / trapezoid(self._project(self.grid_bases, coefs), self.grid) if self.normalize else 1.0
        return n

    def _project(self, bases, coefs):
        """
        _project given a design matrix (or bases) and coefficients, project the coefficients onto the spline

        Args:
            bases (array_like): The set of basis components or design matrix to project onto
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        # print('bases shape: ', bases.shape)
        return jnp.einsum("i...,i->...", bases, coefs)

    def project(self, bases, coefs):
        """
        project given a design matrix (or bases) and coefficients, project the coefficients onto the spline with normalization

        Args:
            bases (array_like): The set of basis components or design matrix to project onto
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        return self._project(bases, coefs) * self.norm(coefs)


class LogXBSpline(BSpline):
    def __init__(self, n_df, knots=None, interior_knots=None, xrange=(0.01, 1), normalize=True, **kwargs):
        """
        Class to construct a basis spline (B-Spline) in the log space of the domain (X)

        Args:
            n_df (int): number of degrees of freedom for the spline
            knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
            interior_knots (array_like, optional): array of interior knots,
                if non-uniform knot placing is preferred. Defaults to None.
            xrange (tuple, optional): domain of spline. Defaults to (0.01, 1).
            k (int, optional): order of the spline +1, i.e. cubic splines->k=4. Defaults to 4 (cubic spline).
            normalize (bool, optional): flag whether or not to numerically normalize the spline. Defaults to True.
        """
        knots = None if knots is None else np.log(knots)
        interior_knots = None if interior_knots is None else np.log(interior_knots)
        xrange = np.log(xrange)
        super().__init__(n_df, knots=knots, interior_knots=interior_knots, xrange=xrange, **kwargs)

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            self.grid = jnp.linspace(*np.exp(xrange), 1000)
            self.grid_bases = jnp.array(self.bases(self.grid))

    def bases(self, xs):
        """
        bases form the basis spline design matrix evaluated at xs (in log space). All basis values outside the range
        of the spline are set to zero.

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            array_like: the design matrix evaluated at xs. shape (N, *xs.shape)
        """
        return super().bases(jnp.log(xs))


class LogYBSpline(BSpline):
    def __init__(self, n_df, knots=None, interior_knots=None, xrange=(0, 1), normalize=True, **kwargs):
        """
        Class to construct a basis spline (B-Spline) in the log space of the range (Y)

        Args:
            n_df (int): number of degrees of freedom for the spline
            knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
            interior_knots (array_like, optional): array of interior knots,
                if non-uniform knot placing is preferred. Defaults to None.
            xrange (tuple, optional): domain of spline. Defaults to (0, 1).
            k (int, optional): order of the spline +1, i.e. cubic splines->k=4. Defaults to 4 (cubic spline).
            normalize (bool, optional): flag whether or not to numerically normalize the spline. Defaults to True.
        """
        super().__init__(n_df, knots=knots, interior_knots=interior_knots, xrange=xrange, **kwargs)
        self.normalize = normalize
        if normalize:
            self.grid = jnp.linspace(*xrange, 1000)
            self.grid_bases = jnp.array(self.bases(self.grid))

    def _project(self, bases, coefs):
        """
        _project given a design matrix (or bases) and coefficients, project the coefficients onto the spline.
        Any inf's in the bases (positive or negative) will be treated as -inf, and 0. will be returned.

        Args:
            bases (array_like): The set of basis components or design matrix to project onto
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        logvals = jnp.nan_to_num(jnp.einsum("i...,i->...", bases, coefs), nan=-jnp.inf, posinf=-jnp.inf)
        return jnp.exp(logvals)

    def bases(self, xs):
        """
        Evaluate the basis spline design matrix at xs. -inf will be used for out-of-range values.

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            array_like: the design matrix evaluated at xs. shape (N, *xs.shape)
        """
        design_matrix = super().bases(xs)
        return jnp.where(jnp.less(xs, self.xrange[0]) | jnp.greater(xs, self.xrange[1]), -jnp.inf, design_matrix)


class LogXLogYBSpline(LogYBSpline):
    def __init__(self, n_df, knots=None, interior_knots=None, xrange=(0.1, 1), normalize=True, **kwargs):
        """
        Class to construct a basis spline (B-Spline) in the log-log space of the domain and range (X and Y)

        Args:
            n_df (int): number of degrees of freedom for the spline
            knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
            interior_knots (array_like, optional): array of interior knots,
                if non-uniform knot placing is preferred. Defaults to None.
            xrange (tuple, optional): domain of spline. Defaults to (0.01, 1).
            k (int, optional): order of the spline +1, i.e. cubic splines->k=4. Defaults to 4 (cubic spline).
            normalize (bool, optional): flag whether or not to numerically normalize the spline. Defaults to True.
        """
        knots = None if knots is None else np.log(knots)
        interior_knots = None if interior_knots is None else np.log(interior_knots)
        xrange = np.log(xrange)
        super().__init__(n_df, knots=knots, interior_knots=interior_knots, xrange=xrange, **kwargs)

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            self.grid = jnp.linspace(*jnp.exp(xrange), 1500)
            self.grid_bases = jnp.array(self.bases(self.grid))

    def bases(self, xs):
        """
        bases form the basis spline design matrix evaluated at xs (in log space).
        -inf will be used for out-of-range values.

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            array_like: the design matrix evaluated at xs. shape (N, *xs.shape)
        """
        logxs = jnp.log(xs)
        design_matrix = super().bases(logxs)
        return jnp.where(jnp.less(logxs, self.xrange[0]) | jnp.greater(logxs, self.xrange[1]), -jnp.inf, design_matrix)

class RectBivariateBasisSpline(object):
    def __init__(
        self,
        xdf,
        ydf,
        xrange=(0, 1),
        yrange=(0, 1),
        kx=4,
        ky=4,
        xbasis=BSpline,
        ybasis=BSpline,
        normalize=True,
    ):
        """
        Class to construct a 2D (bivariate) rectangular basis spline

        Args:
            xdf (int): number of degrees of freedom for the spline in the X direction
            ydf (int): number of degrees of freedom for the spline in the Y direction
            xrange (tuple, optional): domain of X spline. Defaults to (0, 1).
            yrange (tuple, optional): domain of Y spline. Defaults to (0, 1).
            kx (int, optional): order of the X spline +1, i.e. cubic splines->k=4. Defaults to 4 (cubic spline).
            ky (int, optional): order of the Y spline +1, i.e. cubic splines->k=4. Defaults to 4 (cubic spline).
            xbasis (object, optional): Choice of basis to use for the X spline. Defaults to BSpline.
            ybasis (object, optional): Choice of basis to use for the Y spline. Defaults to BSpline.
            normalize (bool, optional): flag whether or not to numerically normalize the spline. Defaults to True.
        """
        self.xdf = xdf
        self.ydf = ydf
        self.x_interpolator = xbasis(xdf, xrange=xrange, k=kx, normalize=False)
        self.y_interpolator = ybasis(ydf, xrange=yrange, k=ky, normalize=False)
        self.normalize = normalize
        self.x_bases = None
        self.y_bases = None
        if self.normalize:
            self.gridx = jnp.linspace(*xrange, 750)
            self.gridy = jnp.linspace(*yrange, 750)
            self.gxx, self.gyy = jnp.meshgrid(self.gridx, self.gridy)
            self.grid_bases = self.bases(self.gxx, self.gyy)

    def norm_2d(self, coefs):
        """
        norm_2d numerically normalizes the 2D spline

        Args:
            coefs (array_like): coefficients for the basis components

        Returns:
            float: the normalization factor given the coefficients
        """
        n = 1.0 / trapezoid(trapezoid(self._project(self.grid_bases, coefs), self.gridy), self.gridx) if self.normalize else 1.0
        return n

    def _reset_bases(self):
        self.x_bases = None
        self.y_bases = None

    def bases(self, xs, ys):
        """
        bases form the basis spline design matrix evaluated at xs and ys

        Args:
            xs (array_like): input values to evaluate the X basis spline at
            xs (array_like): input values to evaluate the Y basis spline at

        Returns:
            array_like: the design matrix evaluated at xs. shape (xdf, ydf, *xs.shape)
        """
        self.x_bases = self.x_interpolator.bases(xs)
        self.y_bases = self.y_interpolator.bases(ys)
        out = jnp.array([[self.x_bases[i] * self.y_bases[j] for i in range(self.xdf)] for j in range(self.ydf)]).reshape(
            self.xdf, self.ydf, *xs.shape
        )
        self.reset_bases()
        return out

    def _project(self, bases, coefs):
        """
        _project given a design matrix (or bases) and coefficients, project the coefficients onto the spline

        Args:
            bases (array_like): The set of basis components or design matrix to project onto
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        return jnp.exp(jnp.einsum("ij...,ij->...", bases, coefs))

    def project(self, bases, coefs):
        """
        project given a design matrix (or bases) and coefficients, project the coefficients onto the spline with normalization

        Args:
            bases (array_like): The set of basis components or design matrix to project onto
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        return self._project(bases, coefs) * self.norm_2d(coefs)

class BSpline_IJR():

    def __init__(self, ndof, domain, order, knots=None, normalize=True, **kwargs):
        """Class to construct the B-spline design matrix
        
        Args: 
            ndof (int): total number of basis functions/degrees of freedom
            domain (tuple): minimum and maximum values of the domain
            order (int): order of B-spline
            knots (array-like): knot vector
            normalize (bool): sets normalization of B-spline
        """
        self.ndof = ndof
        self.domain = domain
        self.order = order
        self.degree = order - 1
        self.knots = knots
        self.n_iknots = ndof - order + 2 # number of interior knots
        self.n_eknots = ndof + order # number of exterior + interior knots
        self.normalize = normalize
        if self.knots is not None:
            assert len(self.knots) == self.n_eknots, "The number of knots must be equal to the degrees of freedom plus the order"
        elif self.knots is None:
            interior_knots = jnp.linspace(*domain, self.n_iknots)
            dx = interior_knots[1] - interior_knots[0]
            self.knots = jnp.linspace(domain[0] - self.degree*dx, domain[1] + self.degree*dx, self.n_eknots)
        if normalize:
            self.integration_domain = jnp.linspace(*domain, 1000)
            self.norm_bases = jnp.array([self.design_matrix(self.integration_domain)[i] for i in range(ndof)])

    def _basis(self, domain_vals, order, index, knots):
        """Calculates the index-th B-spline basis function of order `order`
        
        Args:
            domain_vals (array-like): value(s) to evaluate the basis functions at
            order (int): order of B-spline
            index (int): index of B-spline basis function (index = 0 to `ndof`)
            knots (array-like): knot vector
        """
        def omega(domain_vals, order, index, knots):
            return (domain_vals - knots[index]) / (knots[index+order-1] - knots[index])
        def norm_factor(order, index, knots):
            return order / (knots[index+order] - knots[index])
        if order == 1:
            b1 = jnp.zeros_like(domain_vals)
            q = (domain_vals >= knots[index]) & (domain_vals < knots[index+1])
            b1 += q*1
            b1 *= norm_factor(order, index, knots)
            return b1
        else:
            if knots[index+order-1] - knots[index] < 1e-6:
                term_1 = jnp.zeros_like(domain_vals)
            else:
                term_1 = omega(domain_vals, order, index, knots) * self._basis(domain_vals, order-1, index, knots)
            if knots[index+1+order-1] - knots[index+1] < 1e-6:
                term_2 = jnp.zeros_like(domain_vals)
            else:
                term_2 = (1 - omega(domain_vals, order, index+1, knots)) * self._basis(domain_vals, order-1, index+1, knots)
            basis = term_1 + term_2
            # basis *= norm_factor(order, index, knots)
            return basis
    
    def _design_matrix(self, domain_vals):
        """Calculates the design matrix for the set of points in `domain_vals`
        
        Args:
            domain_vals (array-like): value(s) to evaluate the design matrix at
        """
        return jnp.array([self._basis(domain_vals, order=self.order, index=i, knots=self.knots) for i in range(self.ndof)])
    
    def design_matrix(self, domain_vals):
        """Sets the values of the design matrix outside the domain to zero
        
        Args:
            domain_vals (array-like): value(s) to evaluate the design matrix at
        """
        design_matrix = self._design_matrix(domain_vals)
        return jnp.where(jnp.less(domain_vals, self.domain[0]) | jnp.greater(domain_vals, self.domain[1]), 0.0, design_matrix)
    
    def _spline(self, coeffs, design_matrix):
        """Calculates the spline given a set of coefficients and a design matrix

        Args:
            coeffs (array-like): coefficients of the B-spline
            design_matrix (array-like): design matrix of the basis functions
        """
        assert coeffs.shape[0] == design_matrix.shape[0], "The number of coefficients must match the degrees of freedom"
        return jnp.einsum('i...,i->...', design_matrix, coeffs)
    
    def normalization(self, coeffs):
        """Evaluates the normalization factor of a spline
        
        Args:
            coeffs (array-like): coefficients of the B-spline
        """
        if self.normalize: 
            spline = self._spline(coeffs, self.norm_bases)
            n = trapezoid(spline, self.integration_domain)
            return 1.0 / n
        else: return 1.0
    
    def spline(self, coeffs, design_matrix):
        """Calculates the (normalized) spline given a set of coefficients and a design matrix

        Args:
            coeffs (array-like): coefficients of the B-spline
            design_matrix (array-like): design matrix of the basis functions
        """
        return self._spline(coeffs, design_matrix) * self.normalization(coeffs)
    
class LogYBSpline_IJR(BSpline_IJR):

    def __init__(self, ndof, domain, order, knots=None, normalize=True, **kwargs):
        """Class to construct the B-spline design matrix in logspace
        Args:
            ndof (int): total number of basis functions/degrees of freedom
            domain (tuple): minimum and maximum values of the domain
            order (int): order of B-spline
            knots (array-like): knot vector
            normalize (bool): sets normalization of B-spline
        """
        super().__init__(ndof=ndof, domain=domain, order=order, knots=knots, normalize=normalize, **kwargs)
        self.normalize = normalize
        if normalize:
            self.integration_domain = jnp.linspace(*domain, 1000)
            self.norm_bases = jnp.array([self.design_matrix(self.integration_domain)[i] for i in range(ndof)])

    def design_matrix(self, domain_vals):
        """Sets the values of the design matrix outside the domain to -inf
        
        Args:
            domain_vals (array-like): value(s) to evaluate the design matrix at
        """
        design_matrix = super()._design_matrix(domain_vals)
        return jnp.where(jnp.less(domain_vals, self.domain[0]) | jnp.greater(domain_vals, self.domain[1]), -jnp.inf, design_matrix)

    def _spline(self, coeffs, design_matrix):
        """Calculates the log-spline given a set of coefficients and a design matrix
        
        Args:
            coeffs (array-like): coefficients of the B-spline
            design_matrix (array-like): design matrix of the basis functions
        """
        assert coeffs.shape[0] == design_matrix.shape[0], "The number of coefficients must match the degrees of freedom"
        log_spline = jnp.nan_to_num(jnp.einsum('i...,i->...', design_matrix, coeffs), nan = -jnp.inf, posinf = -jnp.inf)
        return jnp.exp(log_spline)

class BivariateBSpline():

    def __init__(self, ndofs, domains, orders, knots=(None, None), normalize=True, basis = BSpline_IJR, **kwargs):
        """Class to construct the B-spline design tensor in 2D. 
        
        Args:
            ndofs (tuple): pair of the total number of basis functions/degrees of freedom 
            domains (array-like): pair of tuples of the minimum and maximum values of the domains
            orders (tuple): pair of the orders of the B-splines
            knots (array-like, default: tuple of None): pair of knot vectors
            normalize (bool): sets normalization of B-splines
            basis (class): interpolator basis class used to construct the design matrices
        """
        self.ndofs = ndofs
        self.domains = domains
        self.orders = orders
        self.normalize = normalize
        self.u_interpolator = basis(ndof=ndofs[0], domain=domains[0], order=orders[0], knots=knots[0], normalize=normalize, **kwargs)
        self.v_interpolator = basis(ndof=ndofs[1], domain=domains[1], order=orders[1], knots=knots[1], normalize=normalize, **kwargs)
        if normalize:
            self.norm_bases = jnp.tensordot(self.u_interpolator.norm_bases, self.v_interpolator.norm_bases, axes = 0)
        
    def design_tensor(self, u_domain, v_domain, full_product=False):
        """"Calculates the design tensor for the set of points in `u_domain` and `v_domain`
        
        Args:
            u_domain, v_domain (array-like): value(s) in `domain[0]` and `domain[1]` to evaluate the design tensor at, respectively
            full_product (bool): flag to compute the design tensor between all points (`True`), or pairs of points (`False`)
        """
        u_dm = self.u_interpolator.design_matrix(domain_vals=u_domain)
        v_dm = self.v_interpolator.design_matrix(domain_vals=v_domain)
        if full_product:
            return jnp.tensordot(u_dm, v_dm, axes=0)
        else:
            dt = jnp.array([u_dm[i] * v_dm[j] for i in range(self.u_interpolator.ndof) for j in range(self.v_interpolator.ndof)]).reshape(self.u_interpolator.ndof, self.v_interpolator.ndof, *u_domain.shape)
            return dt
        
    def _spline(self, coeffs, design_tensor, full_product=False):
        """Calculates the spline given a set of coefficients and a design tensor

        Args:
            coeffs (array-like): coefficients of the B-spline
            design_tensor (array-like): design tensor of the basis functions
            full_product (bool): flag to compute spline between all points (`True`), or pairs of points (`False`)
        """
        if full_product:
            assert coeffs.shape[0] == design_tensor.shape[0] and coeffs.shape[1] == design_tensor.shape[2], "The number of coefficients must match the degrees of freedom"
            return jnp.einsum('kisj,ks->ij', design_tensor, coeffs)
        else:
            assert coeffs.shape[0] == design_tensor.shape[0] and coeffs.shape[1] == design_tensor.shape[1], "The number of coefficients must match the degrees of freedom"
            return jnp.einsum('ks...,ks->...', design_tensor, coeffs)
        
    def normalization(self, coeffs):
        """Evaluates the normalization factor of a spline
        
        Args:
            coeffs (array-like): coefficients of the B-spline
        """
        # if self.normalize:
        #     spline = self._spline(coeffs, self.norm_bases, full_product=True)
        #     n = trapezoid(trapezoid(spline, self.v_interpolator.integration_domain, axis=1), self.u_interpolator.integration_domain, axis=0)
        #     return 1.0 / n
        # else: return 1.0
        # TODO: Running MCMC and numerically normalizing spline makes MCMC run really long. For now, do not numerically normalize!
        # TODO: Normalize the basis functions instead!
        return 1.0

    def spline(self, coeffs, design_tensor, full_product=False):
        """Calculates the (normalized) spline given a set of coefficients and a design tensor

        Args:
            coeffs (array-like): coefficients of the B-spline
            design_tensor (array-like): design tensor of the basis functions
            full_product (bool): flag to compute spline between all points (`True`), or pairs of points (`False`)
        """
        return self._spline(coeffs, design_tensor, full_product) * self.normalization(coeffs)
    