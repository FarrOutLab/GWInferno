import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import numpy as np

plt.rcParams.update({"text.usetex": True, "font.family": "cmr10", "font.size": 20,
                     "axes.labelpad": 8, "axes.titlepad": 8})

def plot_pdf(x, pdf, label, color="blue", loglog=True, alpha=1.0):
    med = np.median(pdf, axis=0)
    low = np.percentile(pdf, 5, axis=0)
    high = np.percentile(pdf, 95, axis=0)

    if loglog:
        plt.loglog(x, med, lw=2, color=color, label=label, alpha=alpha)
    else:
        plt.plot(x, med, lw=2, color=color, label=label, alpha=alpha)

    plt.fill_between(x, low, high, color=color, alpha=0.1)
    plt.grid(ls='--', alpha=alpha/2)

def plot_pdf_axes(ax, u, pdf, alpha=1.0, color="blue", label=None, loglog=True):
    med = np.median(pdf, axis=0)
    low = np.percentile(pdf, 5, axis=0)
    high = np.percentile(pdf, 95, axis=0)

    if loglog:
        ax.loglog(u, med, lw=2, alpha=alpha, color=color, label=label)
    else:
        ax.plot(u, med, lw=2, alpha=alpha, color=color, label=label)
    
    ax.fill_between(u, low, high, alpha=alpha/10, color=color)
    ax.grid(ls='--', alpha=alpha/2)

def plot_2dpdf(ax, u, v, pdf, alpha = 1.0, cmap:str='viridis', norm='log', levels=1000):
    cf = ax.contourf(u, v, pdf, alpha=alpha, cmap=cmap, norm=norm, levels=levels)
    ax.grid(ls='--', alpha=alpha/2)
    return cf

def plot_2dpercentiles(ax10, ax90, u, v, pdf, alpha = 1.0, cmap:str='viridis', norm='linear', levels=1000, shared_cbar:bool=True):
    low = np.percentile(pdf, 10, axis = 0)
    high = np.percentile(pdf, 90, axis = 0)
    if shared_cbar:
        vmin = min((low.min(), high.min()))
        vmax = max((low.max(), high.max()))
        norm = pltcolors.Normalize(vmin, vmax)
    else:
        norm = norm

    tenth_percentile = plot_2dpdf(ax10, u, v, low, alpha, cmap, norm, levels)
    ninetieth_percentile = plot_2dpdf(ax90, u, v, high, alpha, cmap, norm, levels)
    return tenth_percentile, ninetieth_percentile

def plot_primary_pdfs(mpdfs, m1, names, label, result_dir, save=True):
    plt.figure(figsize=(16,9))
    for i in range(len(mpdfs)):
        plot_pdf(m1, mpdfs[i], names[i])
    plt.ylim(1e-5, 1e0)
    plt.xlabel(r"$m_1 [M_\odot]$")
    plt.legend()
    plt.xlim(m1[0], m1[-1])
    plt.show()
    if save:
        plt.savefig(result_dir + f"/mass_pdf_{label}.png", dpi=100)
    plt.close()

def plot_mass_pdfs(mpdfs, qpdfs, m1, q, names, label, result_dir, save=True, colors=["red", "blue", "green"]):

    plt.figure(figsize=(15, 5))
    for i in range(len(mpdfs)):
        plot_pdf(m1, mpdfs[i], names[i], color=colors[i])
    plt.ylim(1e-5, 1e0)
    plt.xlabel("m1")
    plt.legend()
    plt.xlim(m1[0], m1[-1])
    plt.show()
    if save:
        plt.savefig(result_dir + f"/mass_pdf_{label}.png", dpi=100)
    plt.close()

    plt.figure(figsize=(10, 7))
    for i in range(len(mpdfs)):
        plot_pdf(q, qpdfs[i], names[i], color=colors[i], loglog=False)
    plt.ylim(1e-2, 1e1)
    plt.yscale("log")
    plt.xlabel("q")
    plt.legend()
    plt.xlim(0, 1)
    plt.show()
    if save:
        plt.savefig(result_dir + f"/mass_ratio_pdf_{label}.png", dpi=100)
    plt.close()

def plot_spin_pdfs(a_pdfs, tilt_pdfs, aa, cc, names, label, result_dir, save=True, colors=["red", "blue", "green"], secondary=False):

    if secondary:
        comp = "2"
    else:
        comp = "1"

    plt.figure(figsize=(10, 7))
    for i in range(len(a_pdfs)):
        plot_pdf(aa, a_pdfs[i], names[i], loglog=False, color=colors[i])
    plt.ylim(0, 4)
    plt.xlabel(f"a{comp}")
    plt.legend()
    plt.xlim(0, 1)
    plt.show()
    if save:
        plt.savefig(result_dir + f"/spin_mag{comp}_pdf_{label}.png", dpi=100)
    plt.close()

    plt.figure(figsize=(10, 7))
    for i in range(len(tilt_pdfs)):
        plot_pdf(cc, tilt_pdfs[i], names[i], loglog=False, color=colors[i])
    plt.ylim(0, 1.2)
    plt.xlabel(rf"cos$\theta${comp}")
    plt.legend()
    plt.xlim(-1, 1)
    plt.show()
    if save:
        plt.savefig(result_dir + f"/cos_tilt{comp}_pdf_{label}.png", dpi=100)
    plt.close()

def plot_2dspin_pdfs(a_tilt_1_pdfs, a_tilt_2_pdfs, aa, cc, names, label, result_dir, save=True):
    fig = plt.figure(figsize=(16,9), layout='constrained')
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    for i in range(len(a_tilt_1_pdfs)):
        wireframe, contour = plot_2dpdf(ax1, ax2, aa, cc, a_tilt_1_pdfs[i])
    ax1.set_xlabel(r'$a_1$')
    ax1.set_ylabel(r'$\cos\theta_1$')
    ax1.set_zlabel(r'$p(a_1,\cos\theta_1)$')
    ax2.set_xlabel(r'$a_1$')
    ax2.set_ylabel(r'$\cos\theta_1$')
    # ax2.set_zlabel(r'$p(a,\cos\theta)$')
    fig.colorbar(contour, ax = ax2)
    plt.show()
    if save:
        fig.savefig(result_dir + f"/spin_mag_tilt1_2d_pdf_{label}.png", dpi = 100)
    plt.close()

    fig = plt.figure(figsize=(16,9), layout='constrained')
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    for i in range(len(a_tilt_2_pdfs)):
        wireframe, contour = plot_2dpdf(ax1, ax2, aa, cc, a_tilt_2_pdfs[i])
    ax1.set_xlabel(r'$a_2$')
    ax1.set_ylabel(r'$\cos\theta_2$')
    ax1.set_zlabel(r'$p(a_2,\cos\theta_2)$')
    ax2.set_xlabel(r'$a_2$')
    ax2.set_ylabel(r'$\cos\theta_2$')
    # ax2.set_zlabel(r'$p(a,\cos\theta)$')
    fig.colorbar(contour, ax = ax2)
    plt.show()
    if save:
        fig.savefig(result_dir + f"/spin_mag_tilt2_2d_pdf_{label}.png", dpi = 100)
    plt.close()

def plot_primary_chiq_pdfs(mpdfs, chiqpdfs, ms, chi_effs, qs, names, label, result_dir, save=True):

    plt.figure(figsize=(16,9))
    for i in range(len(mpdfs)):
        plot_pdf(ms, mpdfs[i], names[i])
    plt.ylim(1e-5, 1e0)
    plt.xlabel(r"$m_1$")
    plt.legend()
    plt.xlim(ms[0], ms[-1])
    plt.show()
    if save:
        plt.savefig(result_dir + f"/mass_pdf_{label}.png", dpi=100)
    plt.close()

    fig = plt.figure(figsize=(16,9), layout='constrained')
    wf_ax = fig.add_subplot(121, projection='3d')
    cf_ax = fig.add_subplot(122)
    for i in range(len(chiqpdfs)):
        wf, cf = plot_2dpdf(wf_ax, cf_ax, chi_effs, qs, chiqpdfs[i])
    wf_ax.set_xlabel(r'$\chi_\text{eff}$')
    wf_ax.set_ylabel(r'$q$')
    wf_ax.set_zlabel(r'$p(\chi_\text{eff},q)$')
    cf_ax.set_xlabel(r'$\chi_\text{eff}$')
    cf_ax.set_ylabel(r'$q$')
    fig.colorbar(cf, ax=cf_ax)
    plt.show()
    if save:
        fig.savefig(result_dir + f"/chi-eff_mass-ratio_2d_pdf_{label}.png", dpi=100)
    plt.close()

def plot_marginal_chi_q_pdfs(chieffpdfs, qpdfs, chi_effs, qs, chieff_lims, qmin, label, result_dir, save=True):
    plt.figure(figsize=(16,9), layout='constrained')
    for i in range(len(chieffpdfs)):
        plot_pdf(chi_effs, chieffpdfs[i], label, loglog=False)
    plt.xlabel(r'$\chi\textsubscript{eff}$')#_\text{eff}$')
    plt.ylabel(r'$\tilde{p}(\chi\textsubscript{eff})$')
    plt.title(r'$\tilde{p}(\chi\textsubscript{eff})=\int_{%s}^1dq\,p(\chi\textsubscript{eff},q)$' % qmin)
    plt.show()
    if save:
        plt.savefig(result_dir + f"/marginal_effective_spin_pdf_{label}.png", dpi=100)
    plt.close()

    plt.figure(figsize=(16,9), layout='constrained')
    for i in range(len(qpdfs)):
        plot_pdf(qs, qpdfs[i], label, loglog=False)
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\tilde{p}(q)$')
    plt.title(r'$\tilde{p}(q)=\int_{%s}^{%s}d\chi\textsubscript{eff}\,p(\chi\textsubscript{eff},q)$' % (chieff_lims[0],chieff_lims[-1]))
    plt.show()
    if save:
        plt.savefig(result_dir + f"/marginal_mass-ratio_pdf_{label}.png", dpi=100)
    plt.close()

def plot_chiq_slices(chiqpdfs, chi_effs, qs, label, result_dir, save=True):
    fig = plt.figure(figsize=(16,9), layout='constrained')
    ax_chi_q1 = fig.add_subplot(121)
    # ax_chi_q0 = fig.add_subplot(222)
    # ax_q_chin1 = fig.add_subplot(223)
    ax_q_chi0 = fig.add_subplot(122)
    # axes_dict = {r'$p(\chi_\text{eff},q=1)$':ax_chi_q1, r'$p(\chi_\text{eff},q=0)$':ax_chi_q0, r'$p(\chi_\text{eff}=-1,q)$':ax_q_chin1, r'$p(\chi_\text{eff}=1,q)$':ax_q_chi1}
    axes_dict = {r'$p(\chi\textsubscript{eff},q=1)$':ax_chi_q1, r'$p(\chi\textsubscript{eff}=0,q)$':ax_q_chi0}
    for i in range(len(chiqpdfs)):
        plot_pdf_axes(ax_chi_q1, chi_effs, chiqpdfs[i,:,:,-2], label=label, loglog=False)
        plot_pdf_axes(ax_q_chi0, qs, chiqpdfs[i,:,len(chi_effs)//2,:], label=label, loglog=False)
        # plot_pdf_axes(ax_chi_q0, chi_effs, chiqpdfs[i,:,:,0], label=label, loglog=False)
        # plot_pdf_axes(ax_q_chin1, qs, chiqpdfs[i,:,0,:], label=label, loglog=False)
        # plot_pdf_axes(ax_q_chi1, qs, chiqpdfs[i,:,-1,:], label=label, loglog=False)
        # plot_pdf_axes(ax=, qs, chiqpdfs[i,:,], label=label, loglog=False)
    for key, value in axes_dict.items():
        if 'q=' in key:
            value.set_xlabel(r'$\chi\textsubscript{eff}$')
            value.set_ylabel(key)
        else:
            value.set_xlabel(r'$q$')
            value.set_ylabel(key)
    plt.show()
    if save:
        plt.savefig(result_dir + f"/chi_and_q_slices_pdf_{label}.png", dpi=100)
    plt.close()

def plot_chiq_pdfs_percentiles(chiqpdfs, chi_effs, qs, label, result_dir, save=True):
    fig = plt.figure(figsize=(16,9), layout='constrained')
    ax10_shared_cbar = fig.add_subplot(221)
    ax90_shared_cbar = fig.add_subplot(222)
    ax10 = fig.add_subplot(223)
    ax90 = fig.add_subplot(224)
    axes_dict = {r'$10\textsuperscript{th}$ percentile (shared colorbar)': ax10_shared_cbar, r'$90\textsuperscript{th}$ percentile (shared colorbar)':ax90_shared_cbar,
                 r'$10\textsuperscript{th}$ percentile': ax10, r'$90\textsuperscript{th}$ percentile':ax90}
    for i in range(len(chiqpdfs)):
        cf10_shared_cbar, cf90_shared_cbar = plot_2dpercentiles(ax10_shared_cbar, ax90_shared_cbar, chi_effs, qs, chiqpdfs[i])
        cf10, cf90 = plot_2dpercentiles(ax10, ax90, chi_effs, qs, chiqpdfs[i], shared_cbar=False)
    cfs_dict = {r'$10\textsuperscript{th}$ percentile (shared colorbar)': cf10_shared_cbar, r'$90\textsuperscript{th}$ percentile (shared colorbar)':cf90_shared_cbar,
                 r'$10\textsuperscript{th}$ percentile': cf10, r'$90\textsuperscript{th}$ percentile':cf90}
    for key, value in axes_dict.items():
        value.set_xlabel(r'$\chi\textsubscript{eff}$')
        value.set_ylabel(r'$q$')
        value.set_title(f'{key}')
        fig.colorbar(cfs_dict[key], ax=axes_dict[key])
    plt.show()
    if save:
        fig.savefig(result_dir + f"/chi-eff_mass-ratio_10th_90th_percentile_{label}.png", dpi=100)
    plt.close()

def plot_chiq_pdfs_samples(chiqpdfs, chi_effs, qs, label, result_dir, save=True):
    fig = plt.figure(figsize=(16,9), layout='constrained')
    rng = np.random.default_rng()
    pdf = []
    pdf_idx = []
    for i in range(10):
        rng_idx = rng.integers(chiqpdfs.shape[1])
        pdf.append(chiqpdfs[0, rng_idx])
        pdf_idx.append(rng_idx)
    pdf = np.array(pdf).reshape(2,5,*chiqpdfs[0,0].shape)
    pdf_idx = np.array(pdf_idx).reshape(2,5)
    # norm = pltcolors.Normalize(pdf.min(), pdf.max())
    for i in range(2):
        for j in range(5):
            ax = plt.subplot2grid((2,5), (i,j))
            cf = plot_2dpdf(ax, chi_effs, qs, pdf[i,j])#, norm=norm)
            ax.set_xlabel(r'$\chi\textsubscript{eff}$')
            ax.set_ylabel(r'$q$')
            ax.set_title(f'Sample {pdf_idx[i,j]} out of {chiqpdfs.shape[1]}')
            fig.colorbar(cf, ax=ax)
    plt.show()
    if save:
        fig.savefig(result_dir + f"/chi-eff_mass-ratio_random_samples_{label}.png", dpi=100)
    plt.close()

def plot_chiq_pdfs_stats(chiqpdfs, chi_effs, qs, label, result_dir, sample_frac = 1.0, save=True):
    # add axes
    fig = plt.figure(figsize=(16,9), layout='constrained')
    mean_ax = fig.add_subplot(221, adjustable='box')
    med_ax = fig.add_subplot(222, adjustable='box')
    std_ax = fig.add_subplot(223, adjustable='box')
    frac_uncert_ax = fig.add_subplot(224, adjustable='box')
    axes_dict = {'mean':mean_ax, 'median':med_ax, 'standard deviation':std_ax, 'fractional uncertainty':frac_uncert_ax}
    # select random sample of pdfs
    rng = np.random.default_rng()
    print('Total number of random samples chosen', chiqpdfs.shape[1]*sample_frac)
    rng_100_samples = rng.choice(chiqpdfs, size=int(chiqpdfs.shape[1]*sample_frac), axis=1)[0]
    # compute the mean, median, standard deviation, and fractional uncertainty of the random sample
    mean = np.mean(rng_100_samples, axis=0)
    med = np.mean(rng_100_samples, axis=0)
    std = np.std(rng_100_samples, axis=0, mean=mean)
    ninetyfifth = np.percentile(rng_100_samples, 95, axis=0)
    fifth = np.percentile(rng_100_samples, 5, axis=0)
    frac_uncert = (ninetyfifth - fifth) / med
    # plot
    cf_dict = {'mean':mean, 'median':med, 'standard deviation':std, 'fractional uncertainty':frac_uncert}
    for key, value in cf_dict.items():
        cmap = 'Reds' if key == 'fractional uncertainty' else 'viridis'
        norm = 'log' if key == 'fractional uncertainty' else 'linear'
        cf = plot_2dpdf(axes_dict[key], chi_effs, qs, value, cmap=cmap, norm=norm)
        axes_dict[key].set_xlabel(r'$\chi\textsubscript{eff}$')
        axes_dict[key].set_ylabel(r'$q$')
        axes_dict[key].set_title(f'{key}')
        fig.colorbar(cf, ax=axes_dict[key])

    plt.show()
    if save:
        fig.savefig(result_dir + f"/chi-eff_mass-ratio_stats_{label}.png", dpi=100)

    plt.close()

def plot_rate_of_z_pdfs(z_pdfs, z, label, result_dir, save=True):

    plt.figure(figsize=(10, 7))
    plot_pdf(z, z_pdfs, "redshift")
    plt.xlabel("z")
    plt.ylabel("R(z)")
    plt.legend()
    # plt.xlim(z[0], 1.5)
    # plt.ylim(5, 1e3)
    plt.show()
    if save:
        plt.savefig(result_dir + f"/redshift_pdf_{label}.png", dpi=100)
    plt.close()
