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

save_path = os.path.dirname(__file__) + "/../figs/Mu_2.pdf"

Ms = np.arange(4, 0, -1)
omega = 5
epsilons = np.arange(0, 1.5, 0.0025)
zeta = 0.5
beta = 1.0
gamma = 0.05
Lambda = 1000.0

marker_size_scale_factor = np.sqrt(4)

fig, axs = plt.subplots(1, 2, layout="constrained")
# fig.set_size_inches(6.0462, 2)
fig.set_size_inches(7.0833, 2)

colours = cm.get_cmap("viridis")

edge_colours = [colours(i / 4) for i in range(4)]
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
        mu = np.sort(np.real(eigs[np.abs(np.imag(eigs)) < 1e-12]))
        no_bias_eigs[M - 1].append(-mu[-2])

        h_s_eff_bias_trunc = qt.Qobj(h_s_eff_bias[: 2 * M, : 2 * M], [[M, 2], [M, 2]])

        redfield_tensor = qt.bloch_redfield_tensor(
            h_s_eff_bias_trunc, a_ops=[(couple, env)], fock_basis=True, sec_cutoff=-1
        )

        eigs = redfield_tensor.eigenenergies()
        mu = np.sort(np.real(eigs[np.abs(np.imag(eigs)) < 1e-12]))
        bias_eigs[M - 1].append(-mu[-2])


no_bias_eigs_array = np.array(no_bias_eigs)
bias_eigs_array = np.array(bias_eigs)

for M in Ms[::-1]:
    axs[0].plot(
        epsilons[:361],
        no_bias_eigs_array[M - 1][:361],
        c=edge_colours[M - 1],
        # marker=markers[M - 1],
        # mfc="None",
        # ms=plt.rcParams["lines.markersize"] / marker_size_scale_factor,
        zorder=1000 - 10 * M,
    )
    axs[1].plot(
        epsilons,
        bias_eigs_array[M - 1],
        c=edge_colours[M - 1],
        # marker=markers[M - 1],
        # mfc="None",
        # ms=plt.rcParams["lines.markersize"] / marker_size_scale_factor,
        zorder=1000 - 10 * M,
        label=f"M={M}",
    )

axs[0].set_title("No Bias $E=0$")
axs[1].set_title(r"Bias $E=\Delta/2$")

axs[1].legend(labelspacing=0.3)

fig.supylabel(r"$-\text{Re}(\mu)$")
fig.supxlabel(r"$\lambda/\Omega$")
fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
plt.show()
