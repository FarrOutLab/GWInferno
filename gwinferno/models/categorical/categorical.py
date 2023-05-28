import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro import distributions as dist

from ...distributions import powerlaw_logit_pdf
from ...distributions import powerlaw_pdf
from ...distributions import truncnorm_pdf


def powerlaw_ratio_truncnorm_primary_pdf(ms, qs, mu, sig, beta, mmin, mmax, log=False):
    if log:
        return truncnorm_pdf(ms, mu=mu, sig=sig, low=mmin, high=mmax, log=log) * powerlaw_pdf(
            qs, alpha=beta, low=jnp.exp(mmin) / jnp.exp(ms), high=1.0
        )
    else:
        return truncnorm_pdf(ms, mu=mu, sig=sig, low=mmin, high=mmax, log=log) * powerlaw_pdf(qs, alpha=beta, low=mmin / ms, high=1.0)


def spin_truncnorm_pdf(x1, x2, mu, sig, low=0.0, high=1.0):
    return truncnorm_pdf(x1, mu=mu, sig=sig, low=low, high=high) * truncnorm_pdf(x2, mu=mu, sig=sig, low=low, high=high)


def ordered_peak_mean_priors(npeaks, mmin, mmax, log=True, name=""):
    Ms = jnp.cumsum(numpyro.sample("Ms" + name, dist.Dirichlet(jnp.ones(npeaks + 1))))
    Ms = (Ms * (mmax - mmin)) + mmin
    mps = []
    for i in range(npeaks):
        num = i + 1
        if log:
            mp = numpyro.deterministic("logmp_" + f"{num}" + name, Ms[i])
        else:
            mp = numpyro.deterministic("mp_" + f"{num}" + name, Ms[i])
        mps.append(mp)
    return mps


def radar_plot(categories, events, ax, cm=None):
    categories = [*categories, categories[0]]
    events = np.column_stack((events, events[:, 0]))
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    for i in range(len(events)):
        ax.plot(label_loc, events[i], color=cm[:][i], alpha=0.5)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)


def make_radar_plot(pedata, idata, categories):

    qs = np.array(idata.posterior["Qs"][0]).transpose()

    n_categories = len(categories)
    n_events = qs.shape[0]
    n_samp = qs.shape[1]
    groups = np.zeros([n_categories, n_events])
    for i in range(n_categories):
        x = np.array([np.sum(qs == i, axis=1) / n_samp])
        groups[i] = x
    groups = groups.transpose()
    probs = idata.posterior["Qs"].mean(axis=1).values[0] / (n_categories - 1)

    ticks = np.linspace(0, 1, n_categories)
    cm = plt.cm.tab20(probs)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    radar_plot(categories, groups, ax, cm=cm)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="tab20"), label="category", ax=ax, ticks=ticks, orientation="horizontal", shrink=0.8)
    cbar.ax.set_xticklabels(categories)
    plt.show()


def histogram(data, xlabel, ylabel):
    plt.hist(data.posterior["Qs"].mean(axis=1).values[0], alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_colored_KDES_spin_magnitude(pedata, idata, n_categories, categories, param_map=None):
    param = ["mass_1", "mass_ratio", "a_1", "cos_tilt_1"]
    probs = idata.posterior["Qs"].mean(axis=1).values[0] / (n_categories - 1)
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    ticks = np.linspace(0, 1, n_categories)
    cm = plt.cm.tab20(probs)
    if type(pedata) is dict:
        for i in range(pedata["mass_1"].shape[0]):
            az.plot_dist(pedata[param[0]][i], color=cm[i], plot_kwargs={"alpha": 0.5}, ax=ax[0][0])
            az.plot_dist(pedata[param[1]][i], color=cm[i], plot_kwargs={"alpha": 0.5}, ax=ax[0][1])
            az.plot_dist(pedata[param[2]][i], color=cm[i], plot_kwargs={"alpha": 0.5}, ax=ax[1][0])
            az.plot_dist(pedata[param[3]][i], color=cm[i], plot_kwargs={"alpha": 0.5}, ax=ax[1][1])
    else:
        for i in range(pedata.shape[1]):
            az.plot_dist(pedata[param_map[param[0]]][i], color=cm[i], plot_kwargs={"alpha": 0.5}, ax=ax[0][0])
            az.plot_dist(pedata[param_map[param[1]]][i], color=cm[i], plot_kwargs={"alpha": 0.5}, ax=ax[0][1])
            az.plot_dist(pedata[param_map[param[2]]][i], color=cm[i], plot_kwargs={"alpha": 0.5}, ax=ax[1][0])
            az.plot_dist(pedata[param_map[param[3]]][i], color=cm[i], plot_kwargs={"alpha": 0.5}, ax=ax[1][1])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="tab20"), label="category", ax=ax.ravel().tolist(), ticks=ticks)
    cbar.ax.set_yticklabels(categories)
    ax[0][0].set_xlim(0, 100)
    ax[0][0].set_xlabel(param[0])
    ax[0][1].set_xlabel(param[1])
    ax[1][0].set_xlabel(param[2])
    ax[1][1].set_xlabel(param[3])
    fig.suptitle("Single event KDEs", size=20)


def get_pdf(xx, idata, model, n_model, params, pop_frac=False, chain=0, log_peak=False, **kwargs):
    dat = idata.to_dict()["posterior"]
    pdf = []
    npost = idata.to_dict()["posterior"]["Ps"].shape[1]
    for i in range(npost):
        Params = []
        for k in range(len(params)):
            param = dat[params[k]][chain][i]
            Params.append(param)
        if log_peak:
            Params[0] = jnp.exp(Params[0])
            Params[1] = jnp.exp(Params[1])
        if pop_frac:
            P = dat["Ps"][chain][i][n_model]
            if model == "truncnorm":
                pdf.append(P * truncnorm_pdf(xx, Params[0], Params[1], kwargs["low"], kwargs["high"]))
            if model == "powerlaw_logit":
                pdf.append(P * powerlaw_logit_pdf(xx, Params[0], kwargs["low"], kwargs["high"], Params[1]))
        else:
            if model == "truncnorm":
                pdf.append(truncnorm_pdf(xx, Params[0], Params[1], kwargs["low"], kwargs["high"]))
            if model == "powerlaw_logit":
                pdf.append(powerlaw_logit_pdf(xx, Params[0], kwargs["low"], kwargs["high"], Params[1]))
    return jnp.array(pdf).reshape((npost, len(xx)))


def plot_pdf(x, pdf, label, color="blue", loglog=True, alpha=1.0):
    med = jnp.median(pdf, axis=0)
    low = jnp.percentile(pdf, 5, axis=0)
    high = jnp.percentile(pdf, 95, axis=0)

    if loglog:
        plt.loglog(x, med, lw=2, color=color, label=label, alpha=alpha)
    else:
        plt.plot(x, med, lw=2, color=color, label=label, alpha=alpha)

    plt.fill_between(x, low, high, color=color, alpha=0.1)
