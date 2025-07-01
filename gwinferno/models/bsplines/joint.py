import jax.numpy as jnp

from ...interpolation import BSpline_IJR
from ...interpolation import BivariateBSpline
from .single import Base1DBSplineModel_IJR
from .single import LogYBSpline_IJR     
from .single import BSplineSpinTilt_IJR
from .single import BSplineSpinMagnitude_IJR
   
class Base2DBSplineModel():

    def __init__(self, ndofs, domains, pe_vals, inj_vals, orders, basis=BSpline_IJR, full_product=False, **kwargs):
        """Base class for 2D B-spline population inference, with `BSpline_IJR` as the default basis
        
        Args:
            ndofs (tuple): pair of the total number of basis functions/degrees of freedom
            domains (array-like): pair of tuples of the minimum and maximum values of the domains
            pe_vals, inj_vals (array-like): pair of parameter estimation and injection samples for basis evaluation, respectively
            orders (tuple): pair of the orders of the B-splines
            basis (class): interpolator basis class used to construct the design matrices
            full_product (bool): flag to compute the design tensor between all points (`True`), or pairs of points (`False`)
        """
        self.ndofs = ndofs
        self.domains = domains
        self.orders = orders
        self.full_product = full_product
        self.interpolator = BivariateBSpline(ndofs=ndofs, domains=domains, orders=orders, basis=basis)
        self.pe_dt = self.interpolator.design_tensor(pe_vals[0], pe_vals[1], full_product)
        self.inj_dt = self.interpolator.design_tensor(inj_vals[0], inj_vals[1], full_product)
        self.funcs = [self.inj_pdf, self.pe_pdf]

    def eval_spline(self, coeffs, design_tensor):
        """Calculates the (normalized) spline given a set of coefficients and a design tensor

        Args:
            coeffs (array-like): coefficients of the B-spline
            design_tensor (array-like): design tensor of the basis functions
        """
        return self.interpolator.spline(coeffs, design_tensor, self.full_product)
    
    def pe_pdf(self, coeffs):
        """Project the coefficients `coeffs` onto the design tensor evaluated at the parameter estimation samples

        Args:
            coeffs (array_like): coefficients of the B-spline
        """
        return self.eval_spline(coeffs, self.pe_dt)
    
    def inj_pdf(self, coeffs):
        """Project the coefficients `coeffs` onto the design tensor evaluated at the injection samples

        Args:
            coeffs (array_like): coefficients of the B-spline
        """
        return self.eval_spline(coeffs, self.inj_dt)
    
    def __call__(self, coeffs, pe_samples=True):
        """Evaluate the projection of the coefficients along the design tensor over the parameter estimation or injection samples.
        Use flag `pe_samples` to specify which samples are being evaluated (parameter estimation or injection).

        Args:
            coeffs (array_like): coefficients of the B-spline
            pe_samples (bool):
                If `True`, design tensor is evaluated across parameter estimation samples
                If `False`, design tensor is evaluated across injection samples
        """
        return self.funcs[1](coeffs) if pe_samples else self.funcs[0](coeffs)
    
class BivariateBSplineSpinMagTilt(Base2DBSplineModel):

    def __init__(self, ndofs, pe_vals, inj_vals, orders=(4,4), basis=BSpline_IJR, full_product=False, **kwargs):
        """A 2D B-spline model for the spin magnitude and cosine of spin tilt of a component of a binary pair

        Args:
            ndofs (tuple): pair (primary, primary) of the total number of basis functions/degrees of freedom
            pe_vals, inj_vals (array-like): pair (spin mag, spin tilt) of parameter estimation and injection samples for basis evaluation, respectively
            orders (tuple): pair of the orders of the B-splines
            basis (class): interpolator basis class used to construct the design matrices
            full_product (bool): flag to compute the design tensor between all points (`True`), or pairs of points (`False`)
        """
        spin_mag_domain = (0.0, 1.0)
        spin_tilt_domain = (-1.0, 1.0)
        spin_mag_tilt_domain = jnp.array([spin_mag_domain, spin_tilt_domain])
        domains = kwargs.pop("domains", spin_mag_tilt_domain)
        super().__init__(ndofs, domains, pe_vals, inj_vals, orders, basis, full_product, **kwargs)

class BivariateBSplineSpinTilt(Base2DBSplineModel):

    def __init__(self, ndofs, pe_vals, inj_vals, orders=(4,4), basis=LogYBSpline_IJR, full_product=False, **kwargs):
        """"A 2D B-spline model for the cosine of spin tilts of the components of a binary pair

        Args:
            ndofs (tuple): pair (primary, secondary) of the total number of basis functions/degrees of freedom
            pe_vals, inj_vals (array-like): pair (spin tilt, spin tilt) of parameter estimation and injection samples for basis evaluation, respectively
            orders (tuple): pair of the orders of the B-splines
            basis (class): interpolator basis class used to construct the design matrices
            full_product (bool): flag to compute the design tensor between all points (`True`), or pairs of points (`False`)
        """
        spin_tilt_domain = (-1.0, 1.0)
        spin_tilt_tilt_domain = jnp.array([spin_tilt_domain, spin_tilt_domain])
        domains = kwargs.pop("domains", spin_tilt_tilt_domain)
        super().__init__(ndofs, domains, pe_vals, inj_vals, orders, basis, full_product, **kwargs)
    
class BivariateBSplineSpinMag(Base2DBSplineModel):

    def __init__(self, ndofs, pe_vals, inj_vals, orders=(4,4), basis=LogYBSpline_IJR, full_product=False, **kwargs):
        """"A 2D B-spline model for the spin magnitudes of the components of a binary pair

        Args:
            ndofs (tuple): pair (primary, secondary) of the total number of basis functions/degrees of freedom
            pe_vals, inj_vals (array-like): pair (spin mag, spin mag) of parameter estimation and injection samples for basis evaluation, respectively
            orders (tuple): pair of the orders of the B-splines
            basis (class): interpolator basis class used to construct the design matrices
            full_product (bool): flag to compute the design tensor between all points (`True`), or pairs of points (`False`)
        """
        spin_mag_domain = (0.0, 1.0)
        spin_mag_mag_domain = jnp.array([spin_mag_domain, spin_mag_domain])
        domains = kwargs.pop("domains", spin_mag_mag_domain)
        super().__init__(ndofs, domains, pe_vals, inj_vals, orders, basis, full_product, **kwargs)
