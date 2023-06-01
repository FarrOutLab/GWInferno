"""
a module to calculate GW likleihoods (i.e Whittle) in jax
"""

import jax.numpy as jnp
import numpyro


def inner_product(aa, bb, frequency, psd):
    integrand = jnp.conj(aa) * bb / psd
    df = frequency[1] - frequency[0]
    integral = jnp.sum(integrand) * df
    return 4.0 * jnp.real(integral)


def noise_weighted_inner_product(aa, bb, psd, duration):
    integrand = jnp.conj(aa) * bb / psd
    return 4 / duration * jnp.sum(integrand)


def matched_filter_snr(signal, fd_strain, psd, duration):
    rho_mf = noise_weighted_inner_product(aa=signal, bb=fd_strain, psd=psd, duration=duration)
    rho_mf /= optimal_snr_squared(signal=signal, psd=psd, duration=duration) ** 0.5
    return rho_mf


def optimal_snr_squared(signal, psd, duration):
    return noise_weighted_inner_product(signal, signal, psd, duration)


def noise_log_likelihood(ifos):
    noise_logL = 0.0
    for det in ["H1", "L1"]:
        ifo = ifos[det]
        logL = numpyro.deterministic(
            f"noise_log_likelihood_{det}", -0.5 * noise_weighted_inner_product(ifo.strain, ifo.strain, ifo.psd, ifo.duration)
        )
        noise_logL += logL
    return jnp.real(noise_logL)


def log_likelihood(ifos, hp, hc, ra, dec, time, psi):
    d_inner_h = 0.0
    snr_opt_squared = 0.0
    for det in ["H1", "L1"]:
        ifo = ifos[det]
        signal = ifo.detector_response(hp, hc, ra, dec, time, psi)
        dih = inner_product(ifo.strain, signal, ifo.frequencies, ifo.psd)
        snr_optsq = optimal_snr_squared(signal, ifo.psd, ifo.duration)
        numpyro.deterministic(f"log_likelihood_{det}", jnp.real(dih) - snr_optsq / 2.0)
        d_inner_h += dih
        snr_opt_squared += snr_optsq
    return jnp.real(jnp.real(d_inner_h) - snr_opt_squared / 2.0)


def noise_log_likelihood_det(ifo):
    return jnp.real(-0.5 * noise_weighted_inner_product(ifo.strain, ifo.strain, ifo.psd, ifo.duration))


def log_likelihood_det(ifo, hp, hc, ra, dec, time, psi):
    signal = ifo.detector_response(hp, hc, ra, dec, time, psi)
    dih = inner_product(ifo.strain, signal, ifo.frequencies, ifo.psd)
    snr_optsq = optimal_snr_squared(signal, ifo.psd, ifo.duration)
    return jnp.real(jnp.real(dih) - snr_optsq / 2.0)
