#!/usr/bin/env python

import bilby
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
outdir = "outdir"
label = "GW150914"

trigger_time = 1126259462.4
detectors = ["H1", "L1"]
maximum_frequency = 512
minimum_frequency = 20
roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info(f"Downloading analysis data for ifo {det}")
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info(f"Downloading psd data for ifo {det}")
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=psd.frequencies.value, psd_array=psd.value)
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)

logger.info(f"Saving data plots to {outdir}")
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

priors = bilby.gw.prior.BBHPriorDict(filename="GW150914.prior")
priors["geocent_time"] = bilby.core.prior.Uniform(trigger_time - 0.1, trigger_time + 0.1, name="geocent_time")

waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomPv2",
        "reference_frequency": 50,
    },
)

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    time_marginalization=False,
    phase_marginalization=False,
    distance_marginalization=False,
)

result = bilby.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    outdir=outdir,
    label=label,
    nlive=512,
    check_point_delta_t=600,
    check_point_plot=True,
    npool=12,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    resume=False,
)
result.plot_corner()
