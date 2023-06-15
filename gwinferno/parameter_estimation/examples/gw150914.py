import arviz as az
import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
from bilby.gw.detector import PowerSpectralDensity
from bilby.gw.detector import get_empty_interferometer
from gwpy.timeseries import TimeSeries
from jax import jit
from jax import random
from lal import GreenwichMeanSiderealTime
from numpyro import distributions as dist
from numpyro.infer import MCMC
from numpyro.infer import NUTS
from ripple.waveforms import IMRPhenomD

from gwinferno.parameter_estimation.detectors import H1
from gwinferno.parameter_estimation.detectors import L1
from gwinferno.parameter_estimation.likelihood import log_likelihood_det
from gwinferno.parameter_estimation.likelihood import noise_log_likelihood_det
from gwinferno.parameter_estimation.numpyro_distributions import Cosine
from gwinferno.parameter_estimation.numpyro_distributions import Powerlaw
from gwinferno.parameter_estimation.numpyro_distributions import Sine

az.style.use("arviz-darkgrid")


def setup_data():
    trigger_time = 1126259462.4
    detectors = ["H1", "L1"]
    maximum_frequency = 512
    minimum_frequency = 20
    reference_frequency = 50
    roll_off = 0.4
    duration = 4
    post_trigger_duration = 2
    end_time = trigger_time + post_trigger_duration
    start_time = end_time - duration

    psd_duration = 32 * duration
    psd_start_time = start_time - psd_duration
    psd_end_time = start_time

    ifo_list = {}
    for det in detectors:
        ifo = get_empty_interferometer(det)
        data = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        ifo.strain_data.set_from_gwpy_timeseries(data)
        psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, cache=True)
        psd_alpha = 2 * roll_off / duration
        psd = psd_data.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")
        ifo.power_spectral_density = PowerSpectralDensity(frequency_array=psd.frequencies.value, psd_array=psd.value)
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        ifo_list[det] = ifo

    IFOS = {
        "H1": H1(
            ifo_list["H1"].frequency_domain_strain[ifo_list[det].frequency_mask],
            ifo_list["H1"].frequency_array[ifo_list[det].frequency_mask],
            ifo_list["H1"].power_spectral_density.psd_array[ifo_list[det].frequency_mask],
            ifo_list["H1"].strain_data.start_time,
            duration,
        ),
        "L1": L1(
            ifo_list["H1"].frequency_domain_strain[ifo_list[det].frequency_mask],
            ifo_list["H1"].frequency_array[ifo_list[det].frequency_mask],
            ifo_list["H1"].power_spectral_density.psd_array[ifo_list[det].frequency_mask],
            ifo_list["H1"].strain_data.start_time,
            duration,
        ),
    }

    def gen_waveform(theta):
        hp, hc = IMRPhenomD.gen_IMRPhenomD_polar(IFOS["H1"].frequencies, theta, reference_frequency)
        return hp, hc

    return IFOS, jit(gen_waveform), GreenwichMeanSiderealTime(trigger_time)


def model(ifos, wf_model, trigger_time):
    geocent_time = numpyro.sample("geocent_time", dist.Uniform(trigger_time - 0.1, trigger_time + 0.1))
    chirp_mass = numpyro.sample("chirp_mass", dist.Uniform(20.0, 40.0))
    mass_ratio = numpyro.sample("mass_ratio", dist.Uniform(0.2, 1.0))
    chi1 = numpyro.sample("chi1", dist.Uniform(-1.0, 1.0))
    chi2 = numpyro.sample("chi2", dist.Uniform(-1.0, 1.0))
    phase = numpyro.sample("phase", dist.Uniform(0.0, 2.0 * jnp.pi))  # phase of coalescence
    lum_dist = numpyro.sample("luminosity_distance", Powerlaw(alpha=2.0, minimum=50.0, maximum=2000.0))
    polarization = numpyro.sample("polarization", dist.Uniform(0.0, jnp.pi))  # Polarization angle
    inclination = numpyro.sample("inclination", Sine(minimum=0.0, maximum=jnp.pi))
    ra = numpyro.sample("ra", dist.Uniform(0.0, 2.0 * jnp.pi))  # Right Ascension
    dec = numpyro.sample("dec", Cosine(minimum=-jnp.pi / 2.0, maximum=jnp.pi / 2.0))
    M_T = chirp_mass * (1.0 + mass_ratio) ** 1.2 / mass_ratio**0.6
    eta = (chirp_mass / M_T) ** (5.0 / 3.0)
    theta = jnp.array([chirp_mass, eta, chi1, chi2, lum_dist, geocent_time, phase, inclination, polarization])
    hp, hc = wf_model(theta)
    for det in ["H1", "L1"]:
        logL_det = log_likelihood_det(ifos[det], hp, hc, ra, dec, trigger_time, polarization)
        noise_logL_det = noise_log_likelihood_det(ifos[det])
        numpyro.deterministic(f"delta_logL_{det}", logL_det - noise_logL_det)
        numpyro.factor(f"log_likehood_{det}", logL_det)


def main():
    ifos, wf_model, trigger_time = setup_data()
    RNG = random.PRNGKey(0)
    MCMC_RNG, RNG = random.split(RNG)
    kernel = NUTS(model, dense_mass=True, max_tree_depth=15)
    mcmc = MCMC(kernel, num_warmup=250, num_samples=2250, num_chains=1)
    mcmc.run(MCMC_RNG, ifos, wf_model, trigger_time)
    mcmc.print_summary()
    idata = az.from_numpyro(mcmc)
    az.plot_trace(idata)
    plt.savefig("GW150914_traceplot.png")
    corner.corner(idata)
    plt.savefig("GW150914_cornerplot.png")
    return


if __name__ == "__main__":
    main()
