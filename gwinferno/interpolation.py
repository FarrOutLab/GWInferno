"""
a module for interpolation calculations using jax

Out-of-bounds convention
------------------------
``.bases()`` always returns a **finite** design matrix. Every basis function is zero outside the
spline's domain, so an all-zero column marks an out-of-bounds point. ``_project()`` recovers that
mask with ``jnp.any(bases != 0, axis=0)`` and forces the projected value to zero there. The mask is
derived from the design matrix rather than stored on the instance, so a design matrix stays valid
however many times ``.bases()`` is called.

This approach works with autograd while, e.g., encoding out-of-bounds as ``-inf`` in the design
matrix breaks it.

Note that ``bases()`` is a host-side operation, as it branches on knot values and uses in-place NumPy
assignment, so it cannot be traced. Build design matrices outside ``jit``, as only ``project()`` and
``norm()`` are jax operations.

Knot layout
-----------
``n_df`` basis functions of order ``k = degree + 1`` need ``n_df + k`` knots. Those come from
``n_df - degree + 1`` **breakpoints**, the points where the polynomial pieces join, which span the
domain with both boundaries included, plus ``degree`` exterior knots off each end. For n_df=10 and
degree=3 on [0, 1] that is 6 strictly interior + 2 boundary + 6 exterior = 14 knots.

The exterior knots continue the uniform spacing rather than repeating the boundary, so this is an
open-ended rather than a clamped knot vector. That choice is what makes the B-spline basis a
partition of unity on exactly ``xrange``: the sum falls away outside it instead of being cut off at
a repeated knot. It also means the spline does not interpolate its end coefficients at the
boundaries, and that the two edge M-spline functions integrate to less than 1 over the domain,
since ``bases()`` clips the part of their support lying outside it.

Because the padding is derived from the breakpoints' first gap, a non-uniform breakpoint array
controls only the interior; the exterior knots stay uniformly spaced.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.integrate import trapezoid
from jax.tree_util import register_pytree_node_class


# Names removed by the rename and restructure, with what to use instead. This exists because
# pipeline/parser.py resolves models by dotted string out of YAML config, so a stale config would
# otherwise fail with a bare AttributeError naming neither the cause nor the fix.
_REMOVED_NAMES = {
    "BasisSpline": "use MSpline instead. It was always the M-spline basis, each component integrating to 1; "
    "BSpline is the partition-of-unity basis.",
    "RectBivariateBasisSpline": "it raised AttributeError on construction and has been removed, with no replacement for now.",
    "OOB_VAL": "it was never used. Out-of-bounds points are now marked by an all-zero column in the design matrix; see in_bounds().",
}


def __getattr__(name):
    if name in _REMOVED_NAMES:
        raise AttributeError(f"gwinferno.interpolation.{name} was removed: {_REMOVED_NAMES[name]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def in_bounds(bases):
    """
    in_bounds recover the in-domain mask from a design matrix.

    Every basis function vanishes outside the spline's domain, so an all-zero column marks an
    out-of-bounds point. Inside the domain the B-spline basis is a partition of unity (the
    M-spline basis is strictly positive), so at least one entry per column is non-zero.

    Design matrices are built outside ``jit`` and so reach traced code as compile-time constants.
    Reducing over one inside the graph makes XLA constant-fold it, which on a large matrix costs
    seconds of compile time per call site. When the matrix is concrete the mask is therefore
    computed eagerly with NumPy, entering the graph as a constant instead of a reduction. The jax
    path remains for the rare case of a traced design matrix.

    Args:
        bases (array_like): design matrix of shape (N, *xs.shape)

    Returns:
        array_like: boolean mask of shape xs.shape, True where the point is inside the domain
    """
    if isinstance(bases, jax.core.Tracer):
        return jnp.any(bases != 0.0, axis=0)
    return jnp.asarray(np.any(np.asarray(bases) != 0.0, axis=0))


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


class SplineBasis(object):
    """
    Shared machinery for the spline bases: the knot vector, the Cox-de Boor recursion, the design
    matrix, and the projection contract

        project(bases, coefs) == _project(bases, coefs) * norm(coefs)

    Subclasses supply ``_bases`` (which basis functions the knots generate) and ``norm`` (how those
    basis functions are normalized). :class:`MSpline` and :class:`BSpline` are siblings here: they
    share a knot vector and a recursion, and differ only in the normalization convention of the
    basis functions themselves.

    Not instantiated directly.
    """

    def __init__(
        self,
        n_df,
        knots=None,
        breakpoints=None,
        xrange=(0, 1),
        degree=3,
        *,
        normalize,
        n_grid=1000,
        log_x=False,
        log_y=False,
    ):
        """
        Construct the knot vector and, when normalizing, the integration grid

        Args:
            n_df (int): number of degrees of freedom for the spline
            knots (array_like, optional): array of knots, if non-uniform knot placing is preferred.
                Defaults to None.
            breakpoints (array_like, optional): the points where the polynomial pieces join,
                spanning the domain with both boundaries included. There are n_df - degree + 1 of
                them. Defaults to None, i.e. uniform spacing. Note these are not "interior knots"
                in scipy's sense, which excludes the two boundary points; see the module docstring
                for the knot layout. A non-uniform array controls only the interior, since the
                exterior knots are always spaced uniformly using this array's first gap.
            xrange (tuple, optional): domain of spline. Defaults to (0, 1).
            degree (int, optional): polynomial degree of the spline, i.e. 3 for cubic. Defaults to 3.
                Note this is scipy's convention; the order used internally is degree + 1.
            normalize (bool): whether to numerically normalize the spline. Required, and keyword-only:
                whether a basis is a density is a modelling decision.
            n_grid (int, optional): number of points in the numerical normalization grid. Defaults to 1000.
            log_x (bool, optional): place the knots in log(x) and take the log of inputs to
                ``bases``, so the spline is a function of log(x). Requires a positive lower bound.
                Defaults to False.
            log_y (bool, optional): treat the projection as log(y) and exponentiate it, so the
                spline models the log of the quantity of interest. Defaults to False.
        """
        # Set first: bases() reads these, and __init__ calls bases() below to build the grid.
        self.log_x, self.log_y = log_x, log_y
        if log_x and xrange[0] <= 0:
            raise ValueError(f"a log-x spline needs a strictly positive lower bound, got xrange={tuple(xrange)}")
        if log_x:
            # Knots, breakpoints and the domain all live in the spline's internal coordinates.
            xrange = np.log(xrange)
            knots = None if knots is None else np.log(knots)
            breakpoints = None if breakpoints is None else np.log(breakpoints)

        self.degree = degree
        k = self.order
        self.N = n_df
        self.xrange = xrange
        if knots is None:
            if breakpoints is None:
                breakpoints = np.linspace(xrange[0], xrange[1], n_df - k + 2)
            dx = breakpoints[1] - breakpoints[0]
            knots = jnp.linspace(xrange[0] - dx * (k - 1), xrange[1] + dx * (k - 1), len(breakpoints) + (k - 1) * 2)

        self.knots = knots
        self.breakpoints = breakpoints
        if len(self.knots) != self.N + self.order:
            raise ValueError(
                f"expected {self.N + self.order} knots for n_df={self.N} and degree={self.degree}, got {len(self.knots)}"
            )

        self.normalize = normalize
        self.n_grid = n_grid
        self.basis_vols = np.ones(self.N)
        if normalize:
            # Always in user coordinates: norm() integrates dx there, and bases() re-applies log_x.
            lo, hi = np.exp(self.xrange) if self.log_x else self.xrange
            self.grid = jnp.linspace(lo, hi, n_grid)
            self.grid_bases = jnp.array(self.bases(self.grid))
            self.basis_vols = jnp.array([trapezoid(self.grid_bases[i, :], self.grid) for i in range(self.N)])

    @property
    def order(self):
        """int: the spline order, ``degree + 1``. Kept derived so ``degree`` is the only state."""
        return self.degree + 1

    def norm(self, coefs):
        """
        norm the normalization factor for the given coefficients, or 1.0 when not normalizing

        Subclasses implement this, since it depends on how the basis functions are normalized.

        Args:
            coefs (array_like): coefficients for the basis components

        Returns:
            float: the normalization factor given the coefficients
        """
        raise NotImplementedError

    def _m_basis(self, xs, i, k):
        """
        _m_basis the ith M-spline basis component of order k, computed recursively with the
            Cox-de Boor recursion relation

        The recursion is intrinsically M-normalized: the order-1 pieces are 1/(t[i+1] - t[i]), so
        each component integrates to 1. :class:`BSpline` rescales these by (t[i+k] - t[i])/k to get
        the canonical partition-of-unity B-spline basis.

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
            v = (xs - self.knots[i]) * self._m_basis(xs, i, k - 1) + (self.knots[i + k] - xs) * self._m_basis(xs, i + 1, k - 1)
            return (v * k) / ((k - 1) * (self.knots[i + k] - self.knots[i]))

    def _bases(self, xs):
        """
        _bases construct the set of basis components

        Subclasses implement this, since it is what distinguishes the bases from one another.

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            list: list of all N basis components evaluated at xs
        """
        raise NotImplementedError

    def bases(self, xs):
        """
        bases form the basis spline design matrix evaluated at xs.  All basis values outside the range
        of the spline are set to zero, as are any non-finite inputs (which is how a log-x spline sees
        a non-positive x). The result is always finite; see the module docstring for why.

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            array_like: the design matrix evaluated at xs. shape (N, *xs.shape)
        """
        xs = jnp.log(xs) if self.log_x else jnp.asarray(xs)
        design_matrix = jnp.concatenate(self._bases(xs)).reshape(self.N, *xs.shape)
        return jnp.where(self._out_of_bounds(xs), 0.0, design_matrix)

    def _out_of_bounds(self, xs):
        """
        _out_of_bounds mask of inputs lying outside the spline's domain.

        ``xs`` is in the spline's internal coordinates, so for a log-x basis this is already
        ``log(x)``: a non-positive x arrives here as ``-inf`` or ``nan`` and is masked either way.

        Args:
            xs (array_like): input values, in the spline's internal coordinates

        Returns:
            array_like: boolean mask, True where xs is out of bounds
        """
        return jnp.less(xs, self.xrange[0]) | jnp.greater(xs, self.xrange[1]) | ~jnp.isfinite(xs)

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
        # Solved as a least-squares problem rather than by inverting the normal equations: forming
        # B^T B squares the condition number, and design matrices of overlapping splines are already
        # poorly conditioned.
        alpha = jnp.linalg.lstsq(design_matrix, ys)[0]
        return alpha, jnp.einsum("ji,i->j", design_matrix, alpha), design_matrix

    def _project(self, bases, coefs):
        """
        _project the unnormalized linear combination of the basis components

        Args:
            bases (array_like): The set of basis components or design matrix to project onto
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        values = jnp.einsum("i...,i->...", bases, coefs)
        if not self.log_y:
            return values
        # The spline models log(y), so exponentiate. Out-of-bounds points are marked by an all-zero
        # column, and must give 0 rather than exp(0) == 1.
        return jnp.where(in_bounds(bases), jnp.exp(values), 0.0)

    def project(self, bases, coefs):
        """
        project given a design matrix (or bases) and coefficients, project the coefficients onto the spline

        This is the contract the subclasses are written against: normalization is always
        ``_project`` scaled by ``norm``, so a subclass overrides one or the other but never this.

        Args:
            bases (array_like): The set of basis components or design matrix to project onto
            coefs (array_like): coefficients for the basis components

        Returns:
            array_like: The linear combination of the basis components given the coefficients
        """
        return self._project(bases, coefs) * self.norm(coefs)

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


class MSpline(SplineBasis):
    """
    The M-spline basis: each basis function integrates to 1, so the coefficients act as weights on
    a mixture of densities.

    This is what the Cox-de Boor recursion produces directly. See :class:`BSpline` for the
    partition-of-unity sibling.
    """

    def __init__(self, *args, log_y=False, **kwargs):
        if log_y:
            raise ValueError("MSpline does not support log_y: its norm() is built on basis_vols, which assumes a linear y")
        super().__init__(*args, log_y=False, **kwargs)

    def _bases(self, xs):
        """
        _bases construct the set of M-spline basis components

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            list: list of all N basis components evaluated at xs
        """
        return [self._m_basis(xs, i, k=self.order) for i in range(self.N)]

    def norm(self, coefs):
        """
        norm numerically normalizes the spline

        Each basis function integrates to its entry in ``basis_vols``, so the integral of the
        projection is just the coefficient-weighted sum of those.

        Args:
            coefs (array_like): coefficients for the basis components

        Returns:
            float: the normalization factor given the coefficients
        """
        n = 1.0 / jnp.sum(self.basis_vols * coefs.flatten()) if self.normalize else 1.0
        return n


class BSpline(SplineBasis):
    """
    The canonical B-spline basis: the basis functions sum to 1 across the domain, so the
    coefficients are control points rather than mixture weights.

    These are the M-spline components rescaled by (t[i+k] - t[i])/k. See :class:`MSpline` for the
    unit-integral sibling, and :class:`SplineBasis` for the constructor arguments.

    ``normalize`` is required rather than defaulted: for a B-spline used as a building block inside
    an exponent, 1 / int sum_i c_i B_i(x) dx is not a meaningful scaling and can approach zero or
    change sign for unconstrained real coefficients, so the caller has to say what it wants.
    """

    def _bases(self, xs):
        """
        _bases construct the set of basis components for the canonical B-Spline with the Cox-de Boor recursion relation

        Args:
            xs (array_like): input values to evaluate the basis spline at

        Returns:
            list: list of all N basis components evaluated at xs
        """
        return [(self.knots[i + self.order] - self.knots[i]) / self.order * self._m_basis(xs, i, k=self.order) for i in range(self.N)]

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


class LogXBSpline(BSpline):
    """
    A B-spline in log(x): the knots are placed in log space and ``bases`` takes the log of its
    input, so the spline is a function of log(x) while still being normalized over dx.

    The default ``xrange`` lower bound is positive because log(0) is undefined.
    """

    def __init__(self, n_df, knots=None, breakpoints=None, xrange=(0.01, 1), *, normalize, **kwargs):
        super().__init__(n_df, knots=knots, breakpoints=breakpoints, xrange=xrange, normalize=normalize, log_x=True, **kwargs)


class LogYBSpline(BSpline):
    """
    A B-spline in log(y): the projection is exponentiated, so the coefficients describe the log of
    the quantity of interest and it stays positive for any real coefficients.
    """

    def __init__(self, n_df, knots=None, breakpoints=None, xrange=(0, 1), *, normalize, **kwargs):
        super().__init__(n_df, knots=knots, breakpoints=breakpoints, xrange=xrange, normalize=normalize, log_y=True, **kwargs)


class LogXLogYBSpline(BSpline):
    """
    A B-spline in both log(x) and log(y). See :class:`LogXBSpline` and :class:`LogYBSpline`.
    """

    def __init__(self, n_df, knots=None, breakpoints=None, xrange=(0.1, 1), *, normalize, **kwargs):
        super().__init__(
            n_df, knots=knots, breakpoints=breakpoints, xrange=xrange, normalize=normalize, log_x=True, log_y=True, **kwargs
        )
