import matplotlib.pyplot as plt
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


def plot_mass_pdfs(mpdfs, qpdfs, m1, q, names, label, result_dir, save=True, colors=["red", "blue", "green"], alt_label=""):

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
        plt.savefig(result_dir + f"/mass_ratio_pdf_{label}_{alt_label}.png", dpi=100)
    plt.close()


def plot_spin_pdfs(a_pdfs, tilt_pdfs, aa, cc, names, label, result_dir, save=True, colors=["red", "blue", "green"], secondary=False, alt_label=""):

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
        plt.savefig(result_dir + f"/spin_mag{comp}_pdf_{label}_{alt_label}.png", dpi=100)
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
        plt.savefig(result_dir + f"/cos_tilt{comp}_pdf_{label}_{alt_label}.png", dpi=100)
    plt.close()


def plot_rate_of_z_pdfs(z_pdfs, z, names, label, result_dir, colors=["red", "blue", "green"], save=True):

    plt.figure(figsize=(10, 7))
    for i in range(len(z_pdfs)):
        plot_pdf(z, z_pdfs[i], names[i], color=colors[i], loglog=False)
    plt.xlabel("z")
    plt.ylabel("R(z)")
    plt.legend()
    plt.yscale("log")
    plt.xlim(0, 1.5)
    plt.ylim(5, 1e3)
    plt.show()
    if save:
        plt.savefig(result_dir + f"/redshift_pdf_{label}.png", dpi=100)
    plt.close()
