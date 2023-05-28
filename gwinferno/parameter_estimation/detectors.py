import jax.numpy as jnp

SPEED_OF_LIGHT = 299792458.0
DEG_TO_RAD = jnp.pi / 180.0


def make_get_polarization_tensor(mode):
    if mode.lower() == "plus":

        def kernel(m, n):
            return jnp.einsum("i,j->ij", m, m) - jnp.einsum("i,j->ij", n, n)

    elif mode.lower() == "cross":

        def kernel(m, n):
            return jnp.einsum("i,j->ij", m, n) + jnp.einsum("i,j->ij", n, m)

    else:
        raise ValueError(f"{mode} not a polarization mode!")

    def get_polarization_tensor(ra, dec, gmst, psi):
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec
        u = jnp.array([jnp.cos(phi) * jnp.cos(theta), jnp.cos(theta) * jnp.sin(phi), -jnp.sin(theta)])
        v = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0])
        m = -u * jnp.sin(psi) - v * jnp.cos(psi)
        n = -u * jnp.cos(psi) + v * jnp.sin(psi)
        return kernel(m, n)

    return get_polarization_tensor


def get_geocentric_vertex(lat, lon, elev):
    semi_maj = 6378137.0  # for ellipsoid model of Earth, in m
    semi_min = 6356752.314  # in m
    radius = semi_maj**2 * (semi_maj**2 * jnp.cos(lat) ** 2 + semi_min**2 * jnp.sin(lat) ** 2) ** (-0.5)
    x_comp = (radius + elev) * jnp.cos(lat) * jnp.cos(lon)
    y_comp = (radius + elev) * jnp.cos(lat) * jnp.sin(lon)
    z_comp = ((semi_min / semi_maj) ** 2 * radius + elev) * jnp.sin(lat)
    return jnp.array([x_comp, y_comp, z_comp])


def construct_arm(latitude, longitude, arm_tilt, arm_azimuth):
    e_long = jnp.array([-jnp.sin(longitude), jnp.cos(longitude), 0])
    e_lat = jnp.array([-jnp.sin(latitude) * jnp.cos(longitude), -jnp.sin(latitude) * jnp.sin(longitude), jnp.cos(latitude)])
    e_h = jnp.array([jnp.cos(latitude) * jnp.cos(longitude), jnp.cos(latitude) * jnp.sin(longitude), jnp.sin(latitude)])
    return jnp.cos(arm_tilt) * jnp.cos(arm_azimuth) * e_long + jnp.cos(arm_tilt) * jnp.sin(arm_azimuth) * e_lat + jnp.sin(arm_tilt) * e_h


def get_detector_tensor(arm1, arm2):
    return 0.5 * (jnp.einsum("i,j->ij", arm1, arm1) - jnp.einsum("i,j->ij", arm2, arm2))


class Detector(object):
    name = "BaseDetector"

    def __init__(self, freq_domain_strain, freq_array, psd_array, start_time, duration) -> None:
        self.strain = jnp.array(freq_domain_strain)
        self.frequencies = jnp.array(freq_array)
        self.psd = jnp.array(psd_array)
        self.start_time = start_time
        self.duration = duration
        print(
            f"Initialized {self.name} Detector with frequency domain strain data:",
            "\n duration={duration}seconds starting at GPS",
            "time={start_time}\n {len(freq_array)} Frequency Bins in range:",
            "[{freq_array[0]}Hz,{freq_array[-1]}Hz]",
        )
        self.arm1 = None
        self.arm2 = None
        self.vertex = None
        self.tensor = None
        self.get_pol_tensor = {"plus": None, "cross": None}
        self.noise_logL = None

    def noise_log_likelihood(self):
        if self.noise_logL is None:
            self.noise_logL = -2.0 / self.duration * jnp.sum(jnp.abs(self.strain) ** 2 / self.psd)
        return self.noise_logL

    def setup_geometry(self):
        self.arm1 = construct_arm(self.lat, self.long, self.xarm_tilt, self.xarm_azimuth)
        self.arm2 = construct_arm(self.lat, self.long, self.yarm_tilt, self.yarm_azimuth)
        self.vertex = get_geocentric_vertex(self.lat, self.long, self.elevation)
        self.geocent = jnp.zeros((3,), dtype=float)
        self.tensor = get_detector_tensor(self.arm1, self.arm2)
        self.get_pol_tensor = {"plus": make_get_polarization_tensor("plus"), "cross": make_get_polarization_tensor("cross")}

    def delay_time_from_geocenter(self, ra, dec, time):
        gmst = jnp.mod(time, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec
        omega = jnp.array([jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)])
        delta_d = jnp.array([0.0, 0.0, 0.0]) - self.vertex
        output = jnp.dot(omega, delta_d) / SPEED_OF_LIGHT
        return output

    def antenna_response_cross(self, ra, dec, time, psi):
        polarization_tensor = self.get_pol_tensor["cross"](ra, dec, time, psi)
        return jnp.einsum("ij,ij->", self.tensor, polarization_tensor)

    def antenna_response_plus(self, ra, dec, time, psi):
        polarization_tensor = self.get_pol_tensor["plus"](ra, dec, time, psi)
        return jnp.einsum("ij,ij->", self.tensor, polarization_tensor)

    def detector_response(self, hp, hc, ra, dec, geocent_time, psi):
        resp_pl = self.antenna_response_plus(ra, dec, geocent_time, psi)
        resp_cr = self.antenna_response_cross(ra, dec, geocent_time, psi)
        signal = hp * resp_pl + hc * resp_cr
        time_shift = self.delay_time_from_geocenter(ra, dec, geocent_time)
        dt_geocent = geocent_time - self.start_time
        dt = dt_geocent + time_shift
        output = signal * jnp.exp(-1j * 2 * jnp.pi * self.frequencies * dt)
        return output


class H1(Detector):
    name = "H1"

    def __init__(self, freq_domain_strain, freq_array, psd_array, start_time, duration) -> None:
        super().__init__(freq_domain_strain, freq_array, psd_array, start_time, duration)
        self.lat = (46 + 27.0 / 60 + 18.528 / 3600) * DEG_TO_RAD
        self.long = -(119 + 24.0 / 60 + 27.5657 / 3600) * DEG_TO_RAD
        self.xarm_azimuth = 125.9994 * DEG_TO_RAD
        self.yarm_azimuth = 215.9994 * DEG_TO_RAD
        self.xarm_tilt = -6.195e-4
        self.yarm_tilt = 1.25e-5
        self.elevation = 142.554
        self.setup_geometry()


class L1(Detector):
    name = "L1"

    def __init__(self, freq_domain_strain, freq_array, psd_array, start_time, duration) -> None:
        super().__init__(freq_domain_strain, freq_array, psd_array, start_time, duration)
        self.lat = (30 + 33.0 / 60 + 46.4196 / 3600) * DEG_TO_RAD
        self.long = -(90 + 46.0 / 60 + 27.2654 / 3600) * DEG_TO_RAD
        self.xarm_azimuth = 197.7165 * DEG_TO_RAD
        self.yarm_azimuth = 287.7165 * DEG_TO_RAD
        self.xarm_tilt = 0
        self.yarm_tilt = 0
        self.elevation = -6.574
        self.setup_geometry()


class V1(Detector):
    name = "V1"

    def __init__(self, freq_domain_strain, freq_array, psd_array, start_time, duration) -> None:
        super().__init__(freq_domain_strain, freq_array, psd_array, start_time, duration)
        self.lat = (43 + 37.0 / 60 + 53.0921 / 3600) * DEG_TO_RAD
        self.long = (10 + 30.0 / 60 + 16.1878 / 3600) * DEG_TO_RAD
        self.xarm_azimuth = 70.5674 * DEG_TO_RAD
        self.yarm_azimuth = 160.5674 * DEG_TO_RAD
        self.xarm_tilt = 0
        self.yarm_tilt = 0
        self.elevation = 51.884
        self.setup_geometry()


DETECTORS_IMPLETEMENTED = {"H1": H1, "L1": L1, "V1": V1}
