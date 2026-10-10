import os

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import sympy as sy
from matplotlib import colormaps as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

save_path = os.path.dirname(__file__) + "/../figs/Eig_Vecs.pdf"

M = 3
delta = 1.0
omega = 5.0 / delta
lamb = omega / 4
zeta = 1.0 / delta
beta = 1.0 * delta
gamma = 0.05
Lambda = 1000.0

cbar_width = 0.15

fig1 = plt.figure(layout="constrained")
matrix_fig1, cbar_fig1 = fig1.subfigures(1, 2, width_ratios=[1, cbar_width])
cbar_ax1 = cbar_fig1.subplots()
params1, params2 = matrix_fig1.subfigures(2, 1)
axs1 = params1.subplots(2, 4)
axs2 = params2.subplots(2, 4)

# fig1.set_size_inches(6.0462, 6)
fig1.set_size_inches(7.0833, 6)

colours = cm.get_cmap("viridis")

if M > 1:
    pos = qt.tensor(np.sqrt(2) * qt.position(M), qt.qeye(2))
else:
    pos = qt.qzero([1, 2])
Sx = qt.tensor(qt.qeye(M), qt.sigmax())

epsilon = lamb / omega

eps, Om, z = sy.symbols("e Om z")
with open("Quantum_Rabi_Model/H_S_eff.txt") as file:
    h_s_eff = sy.lambdify([eps, Om, z], sy.sympify(file.read()), "numpy")
h_s_eff = qt.Qobj(h_s_eff(epsilon, omega, zeta)[: 2 * M, : 2 * M], [[M, 2], [M, 2]])
eng_states = h_s_eff.eigenstates()[1]

couple = pos - 2 * epsilon * Sx

if M > 1:
    U = qt.tensor(-1j * epsilon * np.sqrt(2) * qt.momentum(M), qt.sigmax()).expm()
else:
    U = qt.qeye([M, 2])

env = qt.OhmicEnvironment(1 / beta, gamma, Lambda, 1.0)

redfield_tensor = qt.bloch_redfield_tensor(
    h_s_eff, a_ops=[(couple, env)], fock_basis=True, sec_cutoff=-1
)

eigvals, eigvecs = redfield_tensor.eigenstates()
eigvals_copy = np.copy(eigvals)
mins = []
max_elem = eigvals[np.argmax(-np.real(eigvals))]
for i in range(len(eigvals_copy)):
    idx = np.argmin(-np.real(eigvals_copy))
    mins.append(idx)
    eigvals_copy[idx] = max_elem

# Plotting ----------------------------------------------------------------------------------------

for count in range(8):
    idx = mins[count]
    row = count // 4
    col = count % 4
    eigvec = qt.vector_to_operator(eigvecs[idx]).transform(eng_states).tidyup().unit()
    eigval = eigvals[idx]
    eng_density_mat = axs1[row][col].imshow(
        np.abs(eigvec.full()),
        cmap=colours,
        vmin=0,
        vmax=1,
        origin="upper",
        extent=[-0.5, 2 * M - 0.5, 2 * M - 0.5, -0.5],
        aspect="equal",
        interpolation="nearest",
    )
    imag_sign = "+" if np.imag(eigval) >= 0 else "-"
    axs1[row][col].set_xticks([])
    axs1[row][col].set_yticks([])
    if np.abs(np.imag(eigval)) < 1e-12:
        imaginary_part = ""
    else:
        if np.abs(np.imag(eigval)) < 0.1:
            imaginary_part = f"{imag_sign} {np.abs(np.imag(eigval)):.2f}i"
        else:
            imaginary_part = f"{imag_sign} {np.abs(np.imag(eigval)):.1f}i"
    if np.abs(np.real(eigval)) < 1e-12:
        real_part = "0"
    else:
        if np.abs(np.real(eigval)) < 0.1:
            real_part = f"{np.real(eigval):.2f}"
        else:
            real_part = f"{np.real(eigval):.1f}"
    axs1[row][col].set_title(rf"$\mu_{count+1} = {real_part} {imaginary_part}$")

lamb = omega

epsilon = lamb / omega

eps, Om, z = sy.symbols("e Om z")
with open("Quantum_Rabi_Model/H_S_eff.txt") as file:
    h_s_eff = sy.lambdify([eps, Om, z], sy.sympify(file.read()), "numpy")
h_s_eff = qt.Qobj(h_s_eff(epsilon, omega, zeta)[: 2 * M, : 2 * M], [[M, 2], [M, 2]])
eng_states = h_s_eff.eigenstates()[1]

couple = pos - 2 * epsilon * Sx

if M > 1:
    U = qt.tensor(-1j * epsilon * np.sqrt(2) * qt.momentum(M), qt.sigmax()).expm()
else:
    U = qt.qeye([M, 2])

redfield_tensor = qt.bloch_redfield_tensor(
    h_s_eff, a_ops=[(couple, env)], fock_basis=True, sec_cutoff=-1
)

eigvals, eigvecs = redfield_tensor.eigenstates()
eigvals_copy = np.copy(eigvals)
mins = []
max_elem = eigvals[np.argmax(-np.real(eigvals))]
for i in range(len(eigvals_copy)):
    idx = np.argmin(-np.real(eigvals_copy))
    mins.append(idx)
    eigvals_copy[idx] = max_elem

# Plotting ----------------------------------------------------------------------------------------

for count in range(8):
    idx = mins[count]
    row = count // 4
    col = count % 4
    eigvec = qt.vector_to_operator(eigvecs[idx]).transform(eng_states).tidyup().unit()
    eigval = eigvals[idx]
    eng_density_mat = axs2[row][col].imshow(
        np.abs(eigvec.full()),
        cmap=colours,
        vmin=0,
        vmax=1,
        origin="upper",
        extent=[-0.5, 2 * M - 0.5, 2 * M - 0.5, -0.5],
        aspect="equal",
        interpolation="nearest",
    )
    imag_sign = "+" if np.imag(eigval) >= 0 else "-"
    axs2[row][col].set_xticks([])
    axs2[row][col].set_yticks([])
    if np.abs(np.imag(eigval)) < 1e-12:
        imaginary_part = ""
    else:
        if np.abs(np.imag(eigval)) < 0.1:
            imaginary_part = f"{imag_sign} {np.abs(np.imag(eigval)):.2f}i"
        else:
            imaginary_part = f"{imag_sign} {np.abs(np.imag(eigval)):.1f}i"
    if np.abs(np.real(eigval)) < 1e-12:
        real_part = "0"
    else:
        if np.abs(np.real(eigval)) < 0.1:
            real_part = f"{np.real(eigval):.2f}"
        else:
            real_part = f"{np.real(eigval):.1f}"
    axs2[row][col].set_title(rf"$\mu_{count+1} = {real_part} {imaginary_part}$")

fig1.colorbar(eng_density_mat, cax=cbar_ax1, label=r"$|\rho_{ij}|$")
params1.suptitle(r"$\lambda/\Omega = 0.25$")
params2.suptitle(r"$\lambda/\Omega = 1$")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
plt.show()
