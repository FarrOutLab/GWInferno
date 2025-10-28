import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import numpy as np


def plot_pdf(x, pdf, label, color="blue", loglog=True, alpha=1.0):
    med = np.median(pdf, axis=0)
    low = np.percentile(pdf, 5, axis=0)
    high = np.percentile(pdf, 95, axis=0)

    if loglog:
        plt.loglog(x, med, lw=2, color=color, label=label, alpha=alpha)
    else:
        plt.plot(x, med, lw=2, color=color, label=label, alpha=alpha)

    plt.fill_between(x, low, high, color=color, alpha=0.1)

# def plot_2dpdf(ax1, ax2, u, v, pdf, label, color = "blue", loglog = False, alpha = 1.0):
#     rng = np.random.default_rng()
#     rand = pdf[rng.integers(pdf.shape[0])]
#     U, V = np.meshgrid(u, v, indexing = 'ij')
#     wireframe = ax1.plot_wireframe(U, V, rand, color = color, alpha = alpha/2)
#     filled_contour = ax2.contourf(U, V, rand, cmap = 'viridis', alpha = alpha)
#     ax2.grid(ls='--', alpha=alpha/2)
#     return wireframe, filled_contour

def plot_2dpdf(ax1, u, v, pdf, alpha = 1.0, cmap:str='viridis', norm:str='linear'):
    U, V = np.meshgrid(u, v, indexing = 'ij')
    cf = ax1.contourf(U, V, pdf, cmap = cmap, alpha = alpha, norm=norm, levels=20)
    ax1.grid(ls='--', alpha=alpha/2)
    return cf

def plot_2dstats(ax1, ax2, u, v, pdf, label, alpha = 1.0):
    low = np.percentile(pdf, 10, axis = 0)
    high = np.percentile(pdf, 90, axis = 0)
    vmin = min((low.min(), high.min()))
    vmax = max((low.max(), high.max()))
    norm = pltcolors.Normalize(vmin, vmax)

    U, V = np.meshgrid(u, v, indexing = 'ij')
    tenth_percentile = ax1.contourf(U, V, low, cmap = 'viridis', alpha = alpha, norm=norm, levels=20)
    ninetieth_percentile = ax2.contourf(U, V, high, cmap = 'viridis', alpha = alpha, norm=norm, levels=20)
    ax1.grid(ls='--', alpha=alpha/2)
    ax2.grid(ls='--', alpha=alpha/2)
    return tenth_percentile, ninetieth_percentile

def plot_primary_pdfs(mpdfs, m1, names, label, result_dir, save=True):
    plt.figure(figsize=(16,9))
    for i in range(len(mpdfs)):
        plot_pdf(m1, mpdfs[i], names[i])
    plt.ylim(1e-5, 1e0)
    plt.xlabel(r"$m_1$")
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

def plot_chiq_pdfs_stats(chiqpdfs, chi_effs, qs, label, result_dir, save=True):
    fig = plt.figure(figsize=(16,9), layout='constrained')
    cf_ax10 = fig.add_subplot(121)
    cf_ax90 = fig.add_subplot(122)
    axes_dict = {r'$10^\text{th}$ percentile': cf_ax10, r'$90^\text{th}$ percentile':cf_ax90}
    for i in range(len(chiqpdfs)):
        cf10, cf90 = plot_2dstats(cf_ax10, cf_ax90, chi_effs, qs, chiqpdfs[i], label)
    for key, value in axes_dict.items():
        value.set_xlabel(r'$\chi_\text{eff}$')
        value.set_ylabel(r'$q$')
        value.set_title(f'{key}')
        # value.axis('scaled') # hmm
    fig.colorbar(cf10, ax=cf_ax10)
    fig.colorbar(cf90, ax=cf_ax90)
    plt.show()
    if save:
        fig.savefig(result_dir + f"/chi-eff_mass-ratio_10th_90th_percentile_{label}.png", dpi=100)
    plt.close()

def plot_chiq_pdfs_samples(chiqpdfs, chi_effs, qs, label, result_dir, save=True):
    fig = plt.figure(figsize=(16,9), layout='constrained')
    rng = np.random.default_rng()
    rands = []
    rands_idx = []
    for i in range(10):
        rng_idx = rng.integers(chiqpdfs[0].shape[0])
        rands.append(chiqpdfs[0, rng_idx])
        rands_idx.append(rng_idx)
    rands = np.array(rands).reshape(2,5,*chiqpdfs[0,0].shape)
    rands_idx = np.array(rands_idx).reshape(2,5)
    norm = pltcolors.Normalize(rands.min(), rands.max())
    for i in range(2):
        for j in range(5):
            ax = plt.subplot2grid((2,5), (i,j))
            cf = plot_2dpdf(ax, chi_effs, qs, rands[i,j], norm=norm)
            ax.set_xlabel(r'$\chi_\text{eff}$')
            ax.set_ylabel(r'$q$')
            ax.set_title(f'Sample {rands_idx[i,j]}')
            # ax.axis('scaled') # hmm
            fig.colorbar(cf, ax=ax)
    plt.show()
    if save:
        fig.savefig(result_dir + f"/chi-eff_mass-ratio_random_samples_{label}.png", dpi=100)
    plt.close()

def plot_chiq_pdfs_rng_stats(chiqpdfs, chi_effs, qs, label, result_dir, sample_frac = 1.0, save=True):
    # add axes
    fig = plt.figure(figsize=(16,9), layout='constrained')
    cf_mean_ax = fig.add_subplot(221, adjustable='box')
    cf_med_ax = fig.add_subplot(222, adjustable='box')
    cf_std_ax = fig.add_subplot(223, adjustable='box')
    cf_frac_uncert_ax = fig.add_subplot(224, adjustable='box')
    axes_dict = {'mean':cf_mean_ax, 'median':cf_med_ax, 'standard deviation':cf_std_ax, 'fractional uncertainty':cf_frac_uncert_ax}
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
        axes_dict[key].set_xlabel(r'$\chi_\text{eff}$')
        axes_dict[key].set_ylabel(r'$q$')
        axes_dict[key].set_title(f'{key}')
        # axes_dict[key].axis('scaled') # hmm
        fig.colorbar(cf, ax=axes_dict[key])

    plt.show()
    if save:
        fig.savefig(result_dir + f"/chi-eff_mass-ratio_random_samples_stats_{label}.png", dpi=100)

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
