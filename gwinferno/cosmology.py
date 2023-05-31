"""
a module that houses basic cosmology logic -- adapted from code written by Reed Essick included in the gw-distributions package at:
https://git.ligo.org/reed.essick/gw-distributions/-/blob/master/gwdistributions/utils/cosmology.py
"""

import jax.numpy as jnp

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

DEFAULT_DZ = 1e-3  # should be good enough for most numeric integrations we want to do


class Cosmology(object):
    """
    a class that implements specific cosmological computations.
    **NOTE**, we work in CGS units throughout, so Ho must be specified in s**-1 and distances are specified in cm
    """

    def __init__(self, Ho, omega_matter, omega_radiation, omega_lambda):
        self._Ho = Ho
        self._OmegaMatter = omega_matter
        self._OmegaRadiation = omega_radiation
        self._OmegaLambda = omega_lambda
        assert self.OmegaKappa == 0, "we only implement flat cosmologies! OmegaKappa must be 0"
        self._distances = {
            "z": jnp.array([0]),
            "DL": jnp.array([0]),
            "Dc": jnp.array([0]),
            "Vc": jnp.array([0]),
        }

    @property
    def Ho(self):
        return self._Ho

    @property
    def c_over_Ho(self):
        return C_CGS / self.Ho

    @property
    def OmegaMatter(self):
        return self._OmegaMatter

    @property
    def OmegaRadiation(self):
        return self._OmegaRadiation

    @property
    def OmegaLambda(self):
        return self._OmegaLambda

    @property
    def OmegaKappa(self):
        return 1.0 - (self.OmegaMatter + self.OmegaRadiation + self.OmegaLambda)

    @property
    def distances(self):
        return self._distances

    @property
    def z(self):
        return self.distances["z"]

    @property
    def DL(self):
        return self.distances["DL"]

    @property
    def Dc(self):
        return self.distances["Dc"]

    @property
    def Vc(self):
        return self.distances["Vc"]

    def extend(self, max_DL=-jnp.inf, max_Dc=-jnp.inf, max_z=-jnp.inf, max_Vc=-jnp.inf, dz=DEFAULT_DZ):
        """
        integrate to solve for distance measures.
        """
        # note, this could be slow due to trapazoidal approximation with small step size

        # extract current state
        _ = self.distances

        z_list = list(self.z)
        Dc_list = list(self.Dc)
        Vc_list = list(self.Vc)

        current_z = z_list[-1]
        current_Dc = Dc_list[-1]
        current_DL = current_Dc * (1 + current_z)
        current_Vc = Vc_list[-1]

        # initialize integration
        current_dDcdz = self.dDcdz(current_z)
        current_dVcdz = self.dVcdz(current_z, current_Dc)

        # iterate until we are far enough
        while (current_Dc < max_Dc) or (current_DL < max_DL) or (current_z < max_z) or (current_Vc < max_Vc):
            current_z += dz  # increment

            dDcdz = self.dDcdz(current_z)  # evaluated at the next step
            current_Dc += 0.5 * (current_dDcdz + dDcdz) * dz  # trapazoidal approximation
            current_dDcdz = dDcdz  # update

            dVcdz = self.dVcdz(current_z, current_Dc)  # evaluated at the next step
            current_Vc += 0.5 * (current_dVcdz + dVcdz) * dz  # trapazoidal approximation
            current_dVcdz = dVcdz  # update

            current_DL = (1 + current_z) * current_Dc  # update

            Dc_list.append(current_Dc)  # append
            Vc_list.append(current_Vc)
            z_list.append(current_z)

        # record
        self._distances["z"] = jnp.array(z_list, dtype=float)
        self._distances["Dc"] = jnp.array(Dc_list, dtype=float)
        self._distances["Vc"] = jnp.array(Vc_list, dtype=float)
        self._distances["DL"] = (1.0 + self.z) * self.Dc  # only holds in a flat universe

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
        return self.c_over_Ho / self.z2E(z)

    def dDLdz(self, z, dz=DEFAULT_DZ):
        """
        returns Dc + (1+z)*dDcdz
        """
        return self.z2Dc(z, dz=dz) + (1 + z) * self.dDcdz(z)

    def dVcdz(self, z, Dc=None, dz=DEFAULT_DZ):
        """
        returns dVc/dz
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

    def Dc2z(self, Dc, dz=DEFAULT_DZ):
        """
        return redshifts for each Dc specified.
        """
        max_Dc = jnp.max(Dc)
        if max_Dc > jnp.max(self.Dc):
            self.extend(max_Dc=max_Dc, dz=dz)
        return jnp.interp(Dc, self.Dc, self.z)

    def z2Dc(self, z, dz=DEFAULT_DZ):
        """
        return Dc for each z specified
        """
        max_z = jnp.max(z)
        if max_z > jnp.max(self.z):
            self.extend(max_z=max_z, dz=dz)
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

    def Vc2z(self, Vc, dz=DEFAULT_DZ):
        max_Vc = jnp.max(Vc)
        if max_Vc > jnp.max(self.Vc):
            self.extend(max_Vc=max_Vc, dz=dz)

        return jnp.interp(Vc, self.Vc, self.z)

    def z2Vc(self, z, dz=DEFAULT_DZ):
        max_z = jnp.max(z)
        if max_z > jnp.max(self.z):
            self.extend(max_z=max_z, dz=dz)
        return jnp.interp(z, self.z, self.Vc)

    def z2dVcdz(self, z, dz=DEFAULT_DZ):
        return self.dVcdz(z, dz=dz)


# define default cosmology

PLANCK_2018_Cosmology = Cosmology(
    PLANCK_2018_Ho,
    PLANCK_2018_OmegaMatter,
    PLANCK_2018_OmegaRadiation,
    PLANCK_2018_OmegaLambda,
)
