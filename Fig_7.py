import os

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import sympy as sy
from matplotlib import colormaps as cm

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

save_path = os.path.dirname(__file__) + "/../figs/Eig_v_M_Bias.pdf"

Ms = np.arange(4, 0, -1)
omega = 5
epsilons = np.arange(0, 1.5, 0.01)
zeta = 0.5
beta = 1.0
gamma = 0.05
Lambda = 1000.0

marker_size_scale_factor = np.sqrt(4)

fig, axs = plt.subplots(len(Ms), 2, layout="constrained")
# fig.set_size_inches(6.0462, 3.5)
fig.set_size_inches(7.0833, 3.5)

colours = cm.get_cmap("viridis")

# edge_colours = [colours(i / 4) for i in range(4)]
markers = ["o", "v", "s", "*"]

no_bias_eigs: list[list[np.typing.ArrayLike]] = [[] for _ in range(len(Ms))]
bias_eigs: list[list[np.typing.ArrayLike]] = [[] for _ in range(len(Ms))]

for epsilon in epsilons:
    lamb = omega * epsilon
    env = qt.OhmicEnvironment(1 / beta, gamma, Lambda, 1.0)
    eps, Om, z = sy.symbols("e Om z")
    with open("Quantum_Rabi_Model/H_S_eff.txt") as file:
        h_s_eff = sy.lambdify([eps, Om, z], sy.sympify(file.read()), "numpy")
    h_s_eff_no_bias = h_s_eff(epsilon, omega, 0.0)
    h_s_eff_bias = h_s_eff(epsilon, omega, zeta)
    for M in Ms:
        Sz = qt.tensor(qt.qeye(M), qt.sigmaz())
        Sx = qt.tensor(qt.qeye(M), qt.sigmax())
        if M > 1:
            pos = qt.tensor(np.sqrt(2) * qt.position(M), qt.qeye(2))
        else:
            pos = qt.qzero_like(Sz)

        h_s_eff_no_bias_trunc = qt.Qobj(
            h_s_eff_no_bias[: 2 * M, : 2 * M], [[M, 2], [M, 2]]
        )

        couple = pos - 2 * epsilon * Sx

        redfield_tensor = qt.bloch_redfield_tensor(
            h_s_eff_no_bias_trunc, a_ops=[(couple, env)], fock_basis=True, sec_cutoff=-1
        )

        eigs = redfield_tensor.eigenenergies()
        mu = np.sort(np.real(eigs))
        no_bias_eigs[M - 1].append(-mu)

        h_s_eff_bias_trunc = qt.Qobj(h_s_eff_bias[: 2 * M, : 2 * M], [[M, 2], [M, 2]])

        redfield_tensor = qt.bloch_redfield_tensor(
            h_s_eff_bias_trunc, a_ops=[(couple, env)], fock_basis=True, sec_cutoff=-1
        )

        eigs = redfield_tensor.eigenenergies()
        mu = np.sort(np.real(eigs))
        bias_eigs[M - 1].append(-mu)


no_bias_eigs_array = [np.array(no_bias_eigs[i]) for i in range(len(Ms))]
bias_eigs_array = [np.array(bias_eigs[i]) for i in range(len(Ms))]

for M in Ms:
    for i in range(4 * M**2):
        axs[M - 1][0].plot(
            epsilons,
            no_bias_eigs_array[M - 1][:, i],
            # c=edge_colours[3 - i],
            mfc="None",
            ms=plt.rcParams["lines.markersize"] / marker_size_scale_factor,
            zorder=1000 - 10 * M,
        )
        axs[M - 1][1].plot(
            epsilons,
            bias_eigs_array[M - 1][:, i],
            # c=edge_colours[3 - i],
            mfc="None",
            ms=plt.rcParams["lines.markersize"] / marker_size_scale_factor,
            zorder=1000 - 10 * M,
        )
    twin = axs[M - 1][1].twinx()
    twin.set_yticks([])
    twin.set_ylabel(
        rf"$M={M}$",
    )

axs[0][0].set_title("No Bias $E=0$")
axs[0][1].set_title(r"Bias $E=\Delta/2$")

# inset = axs[0][1].inset_axes([0.1, 0.5, 0.4, 0.4], zorder=1100)
# for i in range(4):
#     inset.plot(
#         epsilons[20:56],
#         bias_eigs_array[0][20:56, i],
#         mfc="None",
#         ms=plt.rcParams["lines.markersize"] / marker_size_scale_factor,
#     )

axs[0][1].set_ylim([0, 0.25])
axs[1][0].set_ylim([0, 0.4])
axs[1][1].set_ylim([0, 0.4])
axs[2][0].set_ylim([0, 0.4])
axs[2][1].set_ylim([0, 0.4])
axs[3][0].set_ylim([0, 0.4])
axs[3][1].set_ylim([0, 0.4])

axs[2][1].vlines([0.25, 1], 0, 0.4, ls="--", color="black")

fig.supylabel(r"$-\text{Re}(\mu)$")
fig.supxlabel(r"$\lambda/\Omega$")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
plt.show()
