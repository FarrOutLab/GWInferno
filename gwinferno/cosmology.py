"""
a module for basic cosmology calculations using jax
"""

# adapted from code written by Reed Essick included in the gw-distributions package at:
# https://git.ligo.org/reed.essick/gw-distributions/-/blob/master/gwdistributions/utils/cosmology.py

import jax.numpy as jnp
import numpy as np

# define units in SI
C_SI = 299792458.0
PC_SI = 3.085677581491367e16
MPC_SI = PC_SI * 1e6
G_SI = 6.6743e-11
MSUN_SI = 1.9884099021470415e30

# define units in CGS
G_CGS = G_SI * 1e3
C_CGS = C_SI * 1e2
PC_CGS = PC_SI * 1e2
MPC_CGS = MPC_SI * 1e2
MSUN_CGS = MSUN_SI * 1e3

# Planck 2018 Cosmology (Table1 in arXiv:1807.06209)
PLANCK_2018_Ho = 67.32 / (MPC_SI * 1e-3)  # (km/s/Mpc) / (m/Mpc * km/m) = s**-1
PLANCK_2018_OmegaMatter = 0.3158
PLANCK_2018_OmegaLambda = 1.0 - PLANCK_2018_OmegaMatter
PLANCK_2018_OmegaRadiation = 0.0

DEFAULT_DZ = 5e-3  # should be good enough for most numeric integrations we want to do


class Cosmology(object):
    """
    a class that implements specific cosmological computations.
    **NOTE**, we work in CGS units throughout, so Ho must be specified in s**-1 and distances are specified in cm
    """

    def __init__(self, Ho, omega_matter, omega_radiation, omega_lambda, distance_unit="mpc", initial_z_integ=2.3):
        self.Ho = Ho
        self.c_over_Ho = C_CGS / self.Ho
        self.unit_mod = MPC_CGS if distance_unit == "mpc" else 1.0
        self.OmegaMatter = omega_matter
        self.OmegaRadiation = omega_radiation
        self.OmegaLambda = omega_lambda
        self.OmegaKappa = 1.0 - (self.OmegaMatter + self.OmegaRadiation + self.OmegaLambda)
        assert self.OmegaKappa == 0, "we only implement flat cosmologies! OmegaKappa must be 0"
        self.z = jnp.array([0.0])
        self.Dc = jnp.array([0.0])
        self.Vc = jnp.array([0.0])
        self.extend(max_z=initial_z_integ, dz=DEFAULT_DZ)

    @property
    def DL(self):
        return self.Dc * (1 + self.z)

    def extend(self, max_DL=-jnp.inf, max_Dc=-jnp.inf, max_z=-jnp.inf, max_Vc=-jnp.inf, dz=DEFAULT_DZ):
        """
        integrate to solve for distance measures.
        """
        # note, this could be slow due to trapazoidal approximation with small step size
        # extract current state
        zs = list(self.z)
        Dcs = list(self.Dc)
        Vcs = list(self.Vc)
        z = zs[-1]
        Dc = Dcs[-1]
        Vc = Vcs[-1]
        DL = Dc * (1 + z)

        while jnp.less(Dc, max_Dc) | jnp.less(DL, max_DL) | jnp.less(z, max_z) | jnp.less(Vc, max_Vc):
            dDcdz = self.dDcdz(z)
            dVcdz = self.dVcdz(z, Dc)
            new_z = z + dz
            new_dDcdz = self.dDcdz(new_z)
            new_Dc = Dc + 0.5 * (dDcdz + new_dDcdz) * dz
            new_dVcdz = self.dVcdz(new_z, new_Dc)
            new_Vc = Vc + 0.5 * (dVcdz + new_dVcdz) * dz
            new_DL = (1 + new_z) * new_Dc
            # update state
            z, DL, Dc, Vc = new_z, new_DL, new_Dc, new_Vc
            # append to arrays
            zs.append(z)
            Dcs.append(Dc)
            Vcs.append(Vc)
        self.z = jnp.array(zs)
        self.Dc = jnp.array(Dcs)
        self.Vc = jnp.array(Vcs)


    def z2E(self, z):
        """
        returns E(z) = sqrt(OmegaLambda + OmegaKappa*(1+z)**2 + OmegaMatter*(1+z)**3 + OmegaRadiation*(1+z)**4)
        """
        one_plus_z = 1.0 + z
        return (
            self.OmegaLambda + self.OmegaKappa * one_plus_z**2 + self.OmegaMatter * one_plus_z**3 + self.OmegaRadiation * one_plus_z**4
        ) ** 0.5

    def dDcdz(self, z, mpc=False):
        """
        returns (c/Ho)/E(z)
        """
        dDc = self.c_over_Ho / self.z2E(z)
        if mpc:
            return dDc / MPC_CGS
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
        return jnp.log(4 * jnp.pi) + 2 * jnp.log(Dc) + jnp.log(self.dDcdz(z)) - 3.0 * jnp.log(self.unit_mod)

    def z2Dc(self, z, dz=DEFAULT_DZ):
        """
        return Dc for each z specified
        """
        #max_z = jnp.max(z)
        #if jnp.greater(max_z, jnp.max(self.z)):
        #    self.extend(max_z=max_z,dz=dz)
        return jnp.interp(z, self.z, self.Dc)

    def DL2z(self, DL, dz=DEFAULT_DZ):
        """
        returns redshifts for each DL specified.
        """
        DL_cgs = DL * self.unit_mod
        max_DL = jnp.max(DL_cgs)
        if max_DL > jnp.max(self.DL):  # need to extend the integration
            self.extend(max_DL=max_DL, dz=dz)
        return jnp.interp(DL_cgs, self.DL, self.z)

    def z2DL(self, z, dz=DEFAULT_DZ):
        """
        returns luminosity distance at the specified redshifts
        """
        max_z = jnp.max(z)
        if max_z > jnp.max(self.z):
            self.extend(max_z=max_z, dz=dz)
        return jnp.interp(z, self.z, self.DL) / self.unit_mod


# define default cosmology

PLANCK_2018_Cosmology = Cosmology(
    PLANCK_2018_Ho,
    PLANCK_2018_OmegaMatter,
    PLANCK_2018_OmegaRadiation,
    PLANCK_2018_OmegaLambda,
)
