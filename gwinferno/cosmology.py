"""
a module for basic cosmology calculations using jax. See https://arxiv.org/pdf/astro-ph/9905116 for description of parameters and functions.
"""

# adapted from code written by Reed Essick included in the gw-distributions package at:
# https://git.ligo.org/reed.essick/gw-distributions/-/blob/master/gwdistributions/utils/cosmology.py


import jax.numpy as jnp
from jax.lax import fori_loop

# Planck 2015 Cosmology (Table4 in arXiv:1502.01589, OmegaMatter from astropy Planck 2015)
C_SI = 299792458.0  # m/s
PLANCK_2015_Ho = 67.74 / (1e-3)  # (km/s/Mpc) / (km/m) = m/s/Mpc
PLANCK_2015_OmegaMatter = 0.3089
PLANCK_2015_OmegaLambda = 1.0 - PLANCK_2015_OmegaMatter
PLANCK_2015_OmegaRadiation = 0.0

PLANCK_2015_LVK_Ho = 67.90 / 1e-3
PLANCK_2015_LVK_OmegaMatter = 0.3065
PLANCK_2015_LVK_OmegaLambda = 1.0 - PLANCK_2015_LVK_OmegaMatter
PLANCK_2015_LVK_OmegaRadiation = PLANCK_2015_OmegaRadiation

DEFAULT_DZ = 1e-3  # should be good enough for most numeric integrations we want to do


class Cosmology(object):
    """
    a class that implements specific cosmological computations.
    **NOTE**, we work in SI units throughout, though distances are specified in Mpc.
    """

    def __init__(self, Ho, omega_matter, omega_radiation, omega_lambda, max_z=10.0, dz=DEFAULT_DZ):
        self.Ho = Ho
        self.c_over_Ho = C_SI / self.Ho
        self.OmegaMatter = omega_matter
        self.OmegaRadiation = omega_radiation
        self.OmegaLambda = omega_lambda
        self.OmegaKappa = 1.0 - (self.OmegaMatter + self.OmegaRadiation + self.OmegaLambda)
        assert self.OmegaKappa == 0, "we only implement flat cosmologies! OmegaKappa must be 0"

        self.extend(max_z, dz=dz)

    @property
    def DL(self):
        return self.Dc * (1 + self.z)

    def update(self, i, x):

        z = x[0]
        dz = z[1] - z[0]
        Dc = x[1]
        Vc = x[2]

        dDcdz = self.dDcdz(z[i])
        dVcdz = self.dVcdz(z[i], Dc[i])
        new_dDcdz = self.dDcdz(z[i] + dz)
        Dc = Dc.at[i + 1].set(Dc[i] + 0.5 * (dDcdz + new_dDcdz) * dz)

        new_dVcdz = self.dVcdz(z[i] + dz, Dc[i + 1])
        Vc = Vc.at[i + 1].set(Vc[i] + 0.5 * (dVcdz + new_dVcdz) * dz)

        return jnp.array([z, Dc, Vc])

    def extend(self, max_z, dz=DEFAULT_DZ):
        """
        integrate to solve for distance measures.
        """

        self.z = jnp.arange(0, max_z, dz)
        Dc = jnp.zeros_like(self.z)
        Vc = jnp.zeros_like(self.z)

        X = jnp.array([self.z, Dc, Vc])
        extended_X = fori_loop(0, self.z.shape[0] - 1, self.update, X)
        self.Dc = extended_X[1]
        self.Vc = extended_X[2]

    def z2E(self, z):
        """
        returns E(z) = sqrt(OmegaLambda + OmegaKappa*(1+z)**2 + OmegaMatter*(1+z)**3 + OmegaRadiation*(1+z)**4)
        """
        one_plus_z = 1.0 + z
        return (
            self.OmegaLambda + self.OmegaKappa * one_plus_z**2 + self.OmegaMatter * one_plus_z**3 + self.OmegaRadiation * one_plus_z**4
        ) ** 0.5

    def dDcdz(self, z):
        """
        returns (c/Ho)/E(z)
        """
        dDc = self.c_over_Ho / self.z2E(z)
        return dDc

    def dVcdz(self, z, Dc=None, dz=DEFAULT_DZ):
        """
        return dVc/dz
        """
        if Dc is None:
            Dc = self.z2Dc(z, dz=dz)
        return 4 * jnp.pi * Dc**2 * self.dDcdz(z)

    def logdVcdz(self, z, Dc=None, dz=DEFAULT_DZ):
        """
        return ln(dVc/dz), useful when constructing probability distributions without overflow errors
        """
        if Dc is None:
            Dc = self.z2Dc(z, dz=dz)
        return jnp.log(4 * jnp.pi) + 2 * jnp.log(Dc) + jnp.log(self.dDcdz(z))

    def z2Dc(self, z, dz=DEFAULT_DZ):
        """
        return Dc for each z specified
        """
        max_z = jnp.max(z)
        if jnp.greater(max_z, jnp.max(self.z)):
            self.extend(max_z=max_z, dz=dz)
            return jnp.interp(z, self.z, self.Dc)
        else:
            return jnp.interp(z, self.z, self.Dc)

    def DL2z(self, DL, dz=DEFAULT_DZ):
        """
        returns redshifts for each DL specified.
        """
        max_DL = jnp.max(DL)
        if max_DL > jnp.max(self.DL):  # need to extend the integration
            self.extend(max_DL=max_DL, dz=dz)
        return jnp.interp(DL, self.DL, self.z)

    def z2DL(self, z, dz=DEFAULT_DZ):
        """
        returns luminosity distance at the specified redshifts
        """
        max_z = jnp.max(z)
        if max_z > jnp.max(self.z):
            self.extend(max_z=max_z, dz=dz)
        return jnp.interp(z, self.z, self.DL)


# define default cosmology

PLANCK_2015_Cosmology = Cosmology(
    PLANCK_2015_Ho,
    PLANCK_2015_OmegaMatter,
    PLANCK_2015_OmegaRadiation,
    PLANCK_2015_OmegaLambda,
)

PLANCK_2015_LVK_Cosmology = Cosmology(
    PLANCK_2015_LVK_Ho,
    PLANCK_2015_LVK_OmegaMatter,
    PLANCK_2015_LVK_OmegaRadiation,
    PLANCK_2015_LVK_OmegaLambda,
)
