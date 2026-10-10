import os

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import sympy as sy
from matplotlib import colormaps as cm
from scipy.optimize import root_scalar

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


def delta_tilde(eps):
    return np.exp(-2 * np.pow(eps, 2))


def g_bar(eps):
    return 4 * gamma * delta_tilde(eps) * (1 / np.tanh(beta * delta_tilde(eps)))


def analytic_eigs(eps):
    return np.array(
        [
            np.zeros_like(eps),
            -4 * np.pow(eps, 2) * g_bar(eps),
            2
            * (
                -np.pow(eps, 2) * g_bar(eps)
                + np.sqrt(
                    np.pow(eps, 4) * g_bar(eps) ** 2 - (1 + 0j) * delta_tilde(eps) ** 2
                )
            ),
            2
            * (
                -np.pow(eps, 2) * g_bar(eps)
                - np.sqrt(
                    np.pow(eps, 4) * g_bar(eps) ** 2 - (1 + 0j) * delta_tilde(eps) ** 2
                )
            ),
        ]
    )


def bifurcation_point(x: float) -> float:
    return np.pow(x, 2) * g_bar(x) - delta_tilde(x)


save_path = os.path.dirname(__file__) + "/../figs/Analytic_Eigs.pdf"

M = 1
omega = 5
lambs = np.arange(0, 10, 0.2)
zeta = 0.0
beta = 1.0
gamma = 0.05
Lambda = 1000.0

marker_size_scale_factor = np.sqrt(2)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, layout="constrained")
# fig.set_size_inches(6.0462, 2.5)
fig.set_size_inches(7.0833, 2)

colours = cm.get_cmap("viridis")

edge_colours = [colours(i / 4) for i in range(4)]
markers = ["o", "v", "s", "*"]

eff_eigs: list[np.typing.ArrayLike] = []

bifurcation = omega * root_scalar(bifurcation_point, x0=5.0).root

# for lamb in lambs:
#     epsilon = lamb / omega
#     env = qt.OhmicEnvironment(1 / beta, gamma, Lambda, 1.0)
#     eps, Om, z = sy.symbols("e Om z")
#     with open("Quantum_Rabi_Model/H_S_eff.txt") as file:
#         h_s_eff = sy.lambdify([eps, Om, z], sy.sympify(file.read()), "numpy")
#     h_s_eff = h_s_eff(epsilon, omega, zeta)
#     Sx = qt.tensor(qt.qeye(M), qt.sigmax())
#     pos = qt.qzero_like(Sx)

#     h_s_eff_trunc = qt.Qobj(h_s_eff[: 2 * M, : 2 * M], [[M, 2], [M, 2]])

#     couple = pos - 2 * epsilon * Sx

#     redfield_tensor = qt.bloch_redfield_tensor(
#         h_s_eff_trunc, a_ops=[(couple, env)], fock_basis=True, sec_cutoff=-1
#     )

#     eigs = redfield_tensor.eigenenergies()
#     sorted_idx = np.argsort(np.real(eigs))
#     eff_eigs.append(eigs[sorted_idx][::-1])


# eff_eigs_array = np.array(eff_eigs)

dense_lambs = np.linspace(0, 10, 1000)

analytic_spectrum = analytic_eigs(dense_lambs / omega)

for i in range(4):
    # ax1.scatter(
    #     lambs,
    #     np.real(eff_eigs_array[:, i]),
    #     ec=edge_colours[3 - i],
    #     marker=markers[3 - i],
    #     fc="None",
    #     s=plt.rcParams["lines.markersize"] / marker_size_scale_factor,
    #     label=rf"$\mu_{i+1}$ numeric",
    #     zorder=100 + 100 * i,
    # )
    ax1.plot(
        dense_lambs,
        np.real(analytic_spectrum[i, :]),
        c=edge_colours[3 - i],
        label=rf"$\mu_{i+1}$",
        zorder=50 + 100 * i,
    )
    # scatter = ax2.scatter(
    #     lambs,
    #     np.imag(eff_eigs_array[:, i]),
    #     ec=edge_colours[3 - i],
    #     marker=markers[3 - i],
    #     fc="None",
    #     s=plt.rcParams["lines.markersize"] / marker_size_scale_factor,
    #     label=rf"$\mu_{i+1}$ numeric",
    #     zorder=100 + 100 * i,
    # )
    (plot,) = ax2.plot(
        dense_lambs,
        np.imag(analytic_spectrum[i, :]),
        c=edge_colours[3 - i],
        label=rf"$\mu_{i+1}$",
        zorder=50 + 100 * i,
    )

vline = ax1.vlines(
    bifurcation,
    np.real(analytic_spectrum[1, -1]),
    0.0,
    ls="--",
    color="black",
    zorder=600,
)
ax2.vlines(
    bifurcation,
    -np.imag(analytic_spectrum[2, 0]),
    np.imag(analytic_spectrum[2, 0]),
    ls="--",
    color="black",
    zorder=600,
)

ax1.set_ylabel(r"$\text{Re}(\mu)$")
fig.supxlabel(r"$\lambda/\Delta$")
ax2.set_ylabel(r"$\text{Im}(\mu)$")
leg = ax1.legend(loc="lower left")
leg.set_zorder(700)

plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
plt.show()
