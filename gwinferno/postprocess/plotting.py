"""
a module that stores useful plotting functions
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_mass_dist(pm1s, pqs, ms, qs, mmin=5.0, mmax=100.0, priors=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    for ax, xs, ps, lab in zip(axs, [ms, qs], [pm1s, pqs], ["m1", "q"]):
        me = np.mean(ps, axis=0)
        low = np.percentile(ps, 5, axis=0)
        high = np.percentile(ps, 95, axis=0)
        ax.fill_between(xs, low, high, color="tab:blue", alpha=0.15)
        ax.plot(xs, me, color="tab:blue", lw=4, alpha=0.5, label="MSpline")
        if priors is not None:
            pr = priors[lab]
            lowpr = np.percentile(pr, 5, axis=0)
            highpr = np.percentile(pr, 95, axis=0)
            ax.plot(xs, lowpr, color="k", lw=2, alpha=0.75, ls="--", label="Prior 90\% CI")
            ax.plot(xs, highpr, color="k", lw=2, alpha=0.75, ls="--")

    axs[0].set_xlabel(r"$m_1 \,\,[M_\odot]$", fontsize=16)
    axs[1].set_xlabel(r"$q$", fontsize=16)
    axs[0].set_ylabel(r"$\frac{d\mathcal{R}}{dm_1}$", fontsize=16)
    axs[1].set_ylabel(r"$\frac{d\mathcal{R}}{dq}$", fontsize=16)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_xlim(mmin, mmax)
    axs[0].set_ylim(1e-7, 1e0)
    axs[1].set_xlim(mmin / mmax, 1)
    axs[1].set_ylim(1e-5, 1e0)
    return fig


def plot_chieff_dist(ps, xs, prior=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(xs, low, high, color="tab:blue", alpha=0.15)
    ax.plot(xs, me, color="tab:blue", lw=4, alpha=0.5, label="MSpline")
    if prior is not None:
        lowpr = np.percentile(prior, 5, axis=0)
        highpr = np.percentile(prior, 95, axis=0)
        ax.plot(xs, lowpr, color="k", lw=2, alpha=0.75, ls="--", label="Prior 90\% CI")
        ax.plot(xs, highpr, color="k", lw=2, alpha=0.75, ls="--")
    ax.set_ylim(0, 1.1 * max(high))
    ax.set_xlabel(r"$\chi_\mathrm{eff}$", fontsize=16)
    ax.set_ylabel(r"$\frac{d\mathcal{R}}{d\chi_\mathrm{eff}}$", fontsize=16)
    ax.set_xlim(-1, 1)
    return fig


def plot_iid_spin_dist(pmags, ptilts, mags, tilts, priors=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    for ax, xs, ps, lab in zip(axs, [mags, tilts], [pmags, ptilts], ["mags", "tilts"]):
        me = np.mean(ps, axis=0)
        low = np.percentile(ps, 5, axis=0)
        high = np.percentile(ps, 95, axis=0)
        ax.fill_between(xs, low, high, color="tab:blue", alpha=0.15)
        ax.plot(xs, me, color="tab:blue", lw=4, alpha=0.5, label="MSpline")
        if priors is not None:
            pr = priors[lab]
            lowpr = np.percentile(pr, 5, axis=0)
            highpr = np.percentile(pr, 95, axis=0)
            ax.plot(xs, lowpr, color="k", lw=2, alpha=0.75, ls="--", label="Prior 90\% CI")
            ax.plot(xs, highpr, color="k", lw=2, alpha=0.75, ls="--")
        ax.set_ylim(0, 1.1 * max(high))
    axs[0].set_xlabel(r"$a$", fontsize=16)
    axs[1].set_xlabel(r"$\cos(\theta)$", fontsize=16)
    axs[0].set_ylabel(r"$\frac{d\mathcal{R}}{da}$", fontsize=16)
    axs[1].set_ylabel(r"$\frac{d\mathcal{R}}{d\cos(\theta)}$", fontsize=16)
    axs[0].set_xlim(0, 1)
    axs[1].set_xlim(-1, 1)
    return fig


def plot_ppc_brontosaurus(po, Nobs, m1min, mmax=100, zmax=1.3, chieff=False, params=None):
    if params is None:
        if chieff:
            params = ["mass_1", "mass_ratio", "redshift", "chi_eff"]
        else:
            params = [
                "mass_1",
                "mass_ratio",
                "redshift",
                "a_1",
                "a_2",
                "cos_tilt_1",
                "cos_tilt_2",
            ]
    nplot = len(params)
    fig, axs = plt.subplots(nplot, 1, figsize=(7, 5 * nplot))

    for ax, param in zip(axs, params):

        observed = np.array([po[f"{param}_obs_event_{i}"] for i in range(Nobs)])
        synthetic = np.array([po[f"{param}_pred_event_{i}"] for i in range(Nobs)])

        ax.fill_betweenx(
            y=np.linspace(0, 1, len(observed[:, 0])),
            x1=np.quantile(np.sort(observed, axis=0), 0.05, axis=1),
            x2=np.quantile(np.sort(observed, axis=0), 0.95, axis=1),
            color="tab:blue",
            alpha=0.8,
            label="Observed",
        )
        ax.plot(
            np.quantile(np.sort(observed, axis=0), 0.05, axis=1),
            np.linspace(0, 1, len(observed[:, 0])),
            color="k",
            alpha=0.25,
            lw=0.15,
        )
        ax.plot(
            np.quantile(np.sort(observed, axis=0), 0.95, axis=1),
            np.linspace(0, 1, len(observed[:, 0])),
            color="k",
            alpha=0.25,
            lw=0.15,
        )

        ax.fill_betweenx(
            y=np.linspace(0, 1, len(synthetic[:, 0])),
            x1=np.quantile(np.sort(synthetic, axis=0), 0.05, axis=1),
            x2=np.quantile(np.sort(synthetic, axis=0), 0.95, axis=1),
            color="tab:blue",
            alpha=0.3,
            label="Predicted",
        )
        ax.plot(
            np.quantile(np.sort(synthetic, axis=0), 0.05, axis=1),
            np.linspace(0, 1, len(synthetic[:, 0])),
            color="k",
            alpha=0.25,
            lw=0.15,
        )
        ax.plot(
            np.quantile(np.sort(synthetic, axis=0), 0.95, axis=1),
            np.linspace(0, 1, len(synthetic[:, 0])),
            color="k",
            alpha=0.25,
            lw=0.15,
        )
        ax.plot(
            np.median(np.sort(synthetic, axis=0), axis=1),
            np.linspace(0, 1, len(synthetic[:, 0])),
            color="tab:blue",
            alpha=0.9,
            lw=4,
        )
        ax.legend(loc="upper left")
        ax.set_xlim(
            min(np.min(synthetic), np.min(observed)),
            max(np.max(synthetic), np.max(observed)),
        )

        ax.set_ylim(0, 1)
        ax.grid(which="both", ls=":", lw=1)
        ax.set_ylabel("Cumulative Probability")
        ax.set_xlabel(param)

        if param == "mass_1" or param == "mass_2":
            ax.set_xlim(m1min, mmax)
            ax.set_xscale("log")
        elif "cos_tilt" in param or "chi_eff" in param:
            ax.set_xlim(-1, 1)
        elif param == "redshift":
            ax.set_xlim(0, zmax)
        else:
            ax.set_xlim(0, 1)
    return fig


def plot_m1_vs_z_ppc(po, Nobs, m1min, mmax=100, zmax=1.4):
    pred_m1s = np.array([po[f"mass_1_pred_event_{i}"] for i in range(Nobs)])
    obs_m1s = np.array([po[f"mass_1_obs_event_{i}"] for i in range(Nobs)])
    pred_zs = np.array([po[f"redshift_pred_event_{i}"] for i in range(Nobs)])
    obs_zs = np.array([po[f"redshift_obs_event_{i}"] for i in range(Nobs)])
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    markers = ["o", "^", "+", "v", "x", "<", ">", "1", "2", "8", "P", "*", "d"]
    obs_idxs = np.random.choice(obs_m1s.shape[1], size=len(markers))
    pred_idxs = np.random.choice(pred_m1s.shape[1], size=len(markers))
    for i, m in enumerate(markers):
        ax.scatter(
            pred_m1s[:, pred_idxs[i]],
            pred_zs[:, pred_idxs[i]],
            color="tab:blue",
            marker=m,
            alpha=0.2,
            label="Predicted" if i == 0 else None,
        )
        ax.scatter(
            obs_m1s[:, obs_idxs[i]],
            obs_zs[:, obs_idxs[i]],
            color="tab:orange",
            marker=m,
            alpha=0.2,
            label="Observed" if i == 0 else None,
        )
    ax.axvline(45, color="k", alpha=0.25, lw=2, ls="--")
    ax.axhline(0.4, color="k", alpha=0.25, lw=2, ls="--")
    plt.xlabel(r"$m_1\,[M_\odot]$", fontsize=18)
    plt.ylabel(r"$z$", fontsize=18)
    plt.legend(frameon=False, fontsize=14)
    ax.grid(True, which="major", ls=":")
    ax.tick_params(labelsize=14)
    plt.xscale("log")
    plt.xlim(m1min, mmax)
    plt.ylim(0, zmax)
    return fig


def plot_rofz(dRdz, zs, logx=False, prior=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    me = np.median(dRdz, axis=0)
    low = np.percentile(dRdz, 5, axis=0)
    high = np.percentile(dRdz, 95, axis=0)
    ax.fill_between(zs, low, high, color="tab:blue", alpha=0.15)
    ax.plot(zs, me, color="tab:blue", lw=4, alpha=0.5, label="PL+BSpline")
    ax.plot(zs, me[0] * (1.0 + zs) ** 2.7, lw=5, alpha=0.075, color="k", label="SFR")

    if prior is not None:
        lowpr = np.percentile(prior, 5, axis=0)
        highpr = np.percentile(prior, 95, axis=0)
        ax.plot(
            zs,
            0.6 * me[0] * lowpr,
            color="k",
            lw=2,
            alpha=0.75,
            ls="--",
            label="Prior 90\% CI",
        )
        ax.plot(zs, 1.3 * me[0] * highpr, color="k", lw=2, alpha=0.75, ls="--")

    ax.set_xlabel(r"$z$", fontsize=16)
    ax.set_ylabel(r"$\frac{d\mathcal{R}}{dz}$", fontsize=16)
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(0.01 * min(low), 1.1 * max(high))
    ax.set_xlim(zs[0], zs[-1])
    ax.legend()
    return fig


def plot_ind_spin_dist(pmags, psmags, ptilts, pstilts, mags, tilts, priors=None):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 9), sharey="row")
    j = 0
    i = 0
    ax = axs[i, j]
    ps = pmags
    xs = mags
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(xs, low, high, color="tab:blue", alpha=0.15)
    ax.plot(xs, me, color="tab:blue", lw=4, alpha=0.5, label="BSpline")
    lab = "mag1s"
    if priors is not None:
        pr = priors[lab]
        lowpr = np.percentile(pr, 5, axis=0)
        highpr = np.percentile(pr, 95, axis=0)
        ax.plot(
            xs,
            lowpr,
            color="k",
            lw=2,
            alpha=0.75,
            ls="--",
            label="Prior 90\% CI",
        )
        ax.plot(xs, highpr, color="k", lw=2, alpha=0.75, ls="--")
    ax.set_xlabel(lab, fontsize=16)
    ax.set_ylabel(f"p({lab})", fontsize=16)
    ax.set_ylim(0)
    ax.set_xlim(0, 1)

    j = 1
    i = 0
    ax = axs[i, j]
    ps = psmags
    xs = mags
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(xs, low, high, color="tab:blue", alpha=0.15)
    ax.plot(xs, me, color="tab:blue", lw=4, alpha=0.5, label="BSpline")
    lab = "mag2s"
    if priors is not None:
        pr = priors[lab]
        lowpr = np.percentile(pr, 5, axis=0)
        highpr = np.percentile(pr, 95, axis=0)
        ax.plot(
            xs,
            lowpr,
            color="k",
            lw=2,
            alpha=0.75,
            ls="--",
            label="Prior 90\% CI",
        )
        ax.plot(xs, highpr, color="k", lw=2, alpha=0.75, ls="--")
    ax.set_xlabel(lab, fontsize=16)
    ax.set_ylabel(f"p({lab})", fontsize=16)
    ax.set_ylim(0)
    ax.set_xlim(0, 1)

    j = 0
    i = 1
    ax = axs[i, j]
    ps = ptilts
    xs = tilts
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(xs, low, high, color="tab:blue", alpha=0.15)
    ax.plot(xs, me, color="tab:blue", lw=4, alpha=0.5, label="BSpline")
    lab = "tilt1s"
    if priors is not None:
        pr = priors[lab]
        lowpr = np.percentile(pr, 5, axis=0)
        highpr = np.percentile(pr, 95, axis=0)
        ax.plot(
            xs,
            lowpr,
            color="k",
            lw=2,
            alpha=0.75,
            ls="--",
            label="Prior 90\% CI",
        )
        ax.plot(xs, highpr, color="k", lw=2, alpha=0.75, ls="--")
    ax.set_xlabel(lab, fontsize=16)
    ax.set_ylabel(f"p({lab})", fontsize=16)
    ax.set_ylim(0)
    ax.set_xlim(-1, 1)

    j = 1
    i = 1
    ax = axs[i, j]
    ps = pstilts
    xs = tilts
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(xs, low, high, color="tab:blue", alpha=0.15)
    ax.plot(xs, me, color="tab:blue", lw=4, alpha=0.5, label="BSpline")
    lab = "tilt2s"
    if priors is not None:
        pr = priors[lab]
        lowpr = np.percentile(pr, 5, axis=0)
        highpr = np.percentile(pr, 95, axis=0)
        ax.plot(
            xs,
            lowpr,
            color="k",
            lw=2,
            alpha=0.75,
            ls="--",
            label="Prior 90\% CI",
        )
        ax.plot(xs, highpr, color="k", lw=2, alpha=0.75, ls="--")
    ax.set_xlabel(lab, fontsize=16)
    ax.set_ylabel(f"p({lab})", fontsize=16)
    ax.set_ylim(0)
    ax.set_xlim(-1, 1)
    return fig
