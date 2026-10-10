import os

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import sympy as sy

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
    }
)
plt.rcParams["text.latex.preamble"] = r"""
\usepackage{amsmath}
\renewcommand{\Re}{\operatorname{Re}}
\renewcommand{\Im}{\operatorname{Im}}
"""

save_path = os.path.dirname(__file__) + "/../figs/Redfield_Spectrum.pdf"

Ms = [3, 2, 1]
omega = 5.0
lambs = [0.5, 2.5, 7.5]
zeta = 0.0
beta = 1.0
gamma = 0.05
Lambda = 1000.0

marker_size_scale_factor = 2.5

fig, axs = plt.subplots(3, 2, layout="constrained")
# fig.set_size_inches(6.0462, 4)
fig.set_size_inches(7.0833, 4)

edge_colours = ["tab:blue", "tab:orange", "tab:green"]
shapes = ["o", "v", "s"]
labels = ["$M=1$", "$M=2$", "$M=3$"]
markers = []

for idx, lamb in enumerate(lambs):
    epsilon = lamb / omega
    env = qt.OhmicEnvironment(1 / beta, gamma, Lambda, 1.0)
    eps, Om, z = sy.symbols("e Om z")
    with open("Quantum_Rabi_Model/H_S_eff.txt") as file:
        h_s_eff = sy.lambdify([eps, Om, z], sy.sympify(file.read()), "numpy")
    h_s_eff = h_s_eff(epsilon, omega, zeta)
    for M in Ms:
        Sz = qt.tensor(qt.qeye(M), qt.sigmaz())
        Sx = qt.tensor(qt.qeye(M), qt.sigmax())
        num = qt.tensor(qt.num(M), qt.qeye(2))
        if M > 1:
            pos = qt.tensor(np.sqrt(2) * qt.position(M), qt.qeye(2))
            rc_couple = qt.tensor(np.sqrt(2) * qt.position(M), qt.sigmax())
        else:
            pos = qt.qzero_like(Sz)
            rc_couple = qt.qzero_like(Sz)

        H_S = Sz + zeta * Sx + omega * num + lamb * rc_couple

        redfield_tensor = qt.bloch_redfield_tensor(
            H_S, a_ops=[(pos, env)], fock_basis=True, sec_cutoff=-1
        )

        eigs = redfield_tensor.eigenenergies()

        scatter = axs[idx][0].scatter(
            np.real(eigs),
            np.imag(eigs),
            ec=edge_colours[M - 1],
            fc=[edge_colours[0], "None", "None"][M - 1],
            linewidth=np.pow(1.8, M / 3.0),
            marker=shapes[M - 1],
            s=(plt.rcParams["lines.markersize"] ** 2)
            / ((4 - M) * marker_size_scale_factor),
            zorder=100 - 10 * M,
        )
        markers.append(scatter)

        h_s_eff_trunc = qt.Qobj(h_s_eff[: 2 * M, : 2 * M], [[M, 2], [M, 2]])

        couple = pos - 2 * epsilon * Sx

        redfield_tensor = qt.bloch_redfield_tensor(
            h_s_eff_trunc, a_ops=[(couple, env)], fock_basis=True, sec_cutoff=-1
        )

        eigs = redfield_tensor.eigenenergies()

        axs[idx][1].scatter(
            np.real(eigs),
            np.imag(eigs),
            ec=edge_colours[M - 1],
            fc=[edge_colours[0], "None", "None"][M - 1],
            linewidth=np.pow(1.8, M / 3.0),
            marker=shapes[M - 1],
            s=(plt.rcParams["lines.markersize"] ** 2)
            / ((4 - M) * marker_size_scale_factor),
            zorder=100 - 10 * M,
        )

axs[0][0].set_title("(a1) RC")
axs[0][1].set_title("(a2) CDH")
axs[1][0].set_title("(b1) RC")
axs[1][1].set_title("(b2) CDH")
axs[2][0].set_title("(c1) RC")
axs[2][1].set_title("(c2) CDH")

fig.text(0.5, 0.963, r"$\lambda=\Omega/10$", transform=fig.transFigure, ha="center")
fig.text(0.5, 0.6495, r"$\lambda=\Omega/2$", transform=fig.transFigure, ha="center")
fig.text(0.5, 0.335, r"$\lambda=3\Omega/2$", transform=fig.transFigure, ha="center")

fig.supxlabel(r"$\Re(\mu)$")
fig.supylabel(r"$\Im(\mu)$")
plt.figlegend(
    markers[:3:-1],
    labels,
    loc="outside right upper",
    handletextpad=0.4,
    borderpad=0.2,
    borderaxespad=0.25,
    frameon=False,
)

fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
plt.show()
