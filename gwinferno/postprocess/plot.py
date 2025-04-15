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

def plot_2dpdf(ax1, ax2, u, v, pdf, label, color = "blue", loglog = False, alpha = 1.0):
    med = np.median(pdf, axis = 0)
    # low = np.percentile(pdf, 5, axis = 0)
    # high = np.percentile(pdf, 95, axis = 0)

    if loglog:
        pass
    else:
        U, V = np.meshgrid(u, v, indexing = 'ij')
        wireframe = ax1.plot_wireframe(U, V, med, color = color, alpha = alpha)
        filled_contour = ax2.contourf(U, V, med, cmap = 'viridis', alpha = alpha)
        return wireframe, filled_contour

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

def plot_2dspin_pdfs(a_pdfs, tilt_pdfs, aa, cc, names, label, result_dir, save=True, colors=["red", "blue", "green"]):
    fig = plt.figure(figsize=(10,7), layout = 'tight')
    ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(len(a_pdfs)):
        wireframe, contour = plot_2dpdf(ax1, ax2, aa, aa, a_pdfs[i], label = label, color=colors[i])
    ax1.set_xlabel('a1')
    ax1.set_ylabel('a2')
    ax2.set_xlabel('a1')
    ax2.set_ylabel('a2')
    fig.colorbar(contour, ax = ax2)
    plt.show()
    if save:
        fig.savefig(result_dir + f"/spin_mag2d_pdf_{label}.png", dpi = 100)
    plt.close()

    fig = plt.figure(figsize=(10,7), layout = 'tight')
    ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(len(tilt_pdfs)):
        wireframe, contour = plot_2dpdf(ax1, ax2, cc, cc, tilt_pdfs[i], label = label, color=colors[i])
    ax1.set_xlabel(r'cos$\theta_1$')
    ax1.set_ylabel(r'cos$\theta_2$')
    ax2.set_xlabel(r'cos$\theta_1$')
    ax2.set_ylabel(r'cos$\theta_2$')
    fig.colorbar(contour, ax = ax2)
    plt.show()
    if save:
        fig.savefig(result_dir + f"/cos_tilt2d_pdf_{label}.png", dpi = 100)
    plt.close()

def plot_rate_of_z_pdfs(z_pdfs, z, label, result_dir, save=True):

    plt.figure(figsize=(10, 7))
    plot_pdf(z, z_pdfs, "redshift")
    plt.xlabel("z")
    plt.ylabel("R(z)")
    plt.legend()
    plt.xlim(z[0], 1.5)
    plt.ylim(5, 1e3)
    plt.show()
    if save:
        plt.savefig(result_dir + f"/redshift_pdf_{label}.png", dpi=100)
    plt.close()
