#!/usr/bin/env python3

import numpy as np
import pennylane as qml
from scipy.linalg import expm
import matplotlib.pyplot as plt
import math


# === Parameters ===
n = 2
m = 4
n_anc = math.ceil(np.log2(2 * m + 1))
pad_len = 2 ** n_anc

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# ===  A = 1/sqrt(2) * (Z + X) ===
# P_terms = [Z, X]
# H = sum(P_terms) * 1/np.sqrt(2)

# ===  A = 1/2 * (ZZ + ZX + XZ + XX) ===
P_terms = [np.kron(Z, Z), np.kron(Z, X), np.kron(X, Z), np.kron(X, X)]
H = sum(P_terms) / 2

#
# # === Construct TFIM Hamiltonian ===
# H = np.zeros((2**n, 2**n))
# for i in range(n - 1):
#     Z = np.eye(2)
#     for j in range(n):
#         Z_op = np.array([[1, 0], [0, -1]]) if j in [i, i + 1] else np.eye(2)
#         Z = Z_op if j == 0 else np.kron(Z, Z_op)
#     H -= Z
# for i in range(n):
#     X = np.eye(2)
#     for j in range(n):
#         X_op = np.array([[0, 1], [1, 0]]) if j == i else np.eye(2)
#         X = X_op if j == 0 else np.kron(X, X_op)
#     H -= X
λ_min, λ_max = np.min(np.linalg.eigvalsh(H)), np.max(np.linalg.eigvalsh(H))
H_norm = (H - λ_min * np.eye(2**n)) / (λ_max - λ_min) * np.pi/2

# === LCU + SELECT ===
k_vals = np.arange(-m, m + 1)
alpha_k = 2**(-2 * m) * np.array([math.comb(2 * m, m + k) for k in k_vals])
alpha_k /= np.sum(alpha_k)
weights = np.sqrt(np.pad(alpha_k, (0, pad_len - len(alpha_k))))
U_list = [expm(2j * (k - m) * H_norm) for k in range(len(alpha_k))]
U_list += [np.eye(2**n)] * (pad_len - len(U_list))
U_ops = [qml.QubitUnitary(U, wires=range(n_anc, n_anc + n)) for U in U_list]
# === PennyLane Device & Circuit ===
dev = qml.device("default.qubit", wires=n + n_anc)
@qml.qnode(dev)
def W_circuit():
    qml.StatePrep(weights, wires=range(n_anc))
    qml.Select(U_ops, control=range(n_anc))
    qml.adjoint(qml.StatePrep)(weights, wires=range(n_anc))
    return qml.state()

def prepare_initial_state(n, n_anc):
    #psi_sys = np.ones(2**n) / np.sqrt(2**n)
    psi_sys = np.zeros(2 ** n)
    psi_sys[0] = 1
    psi_anc = np.zeros(2**n_anc)
    psi_anc[0] = 1.0
    return np.kron(psi_anc, psi_sys)

def construct_reflection_operator(n, n_anc):
    I_sys = np.eye(2**n)
    P0_anc = np.zeros((2**n_anc, 2**n_anc))
    P0_anc[0, 0] = 1
    #R = np.kron(np.eye(2**n_anc) - 2 * P0_anc, I_sys) ## (I_anc - 2|0><0|)\otimes I_sys
    R = 2*np.kron(P0_anc, I_sys) - np.eye(2**(n_anc+n))
    return R

def simulate_oaa(W, W_dagger, R, psi0, max_iter):
    success_probs, fidelities = [], []
    psi = psi0.copy()
    for _ in range(max_iter):
        psi = R @ psi
        psi = W_dagger @ psi
        psi = R @ psi
        psi = -W @ psi
        reshaped = psi.reshape((2**n_anc, 2**n))
        system_unnorm = reshaped[0, :]
        # reshaped = psi.reshape((2**n, 2**n_anc))
        # system_unnorm = reshaped[:, 0]
        success_prob = np.sum(np.abs(system_unnorm)**2)
        system_state = system_unnorm / np.sqrt(success_prob)
        ground_state = np.linalg.eigh(H_norm)[1][:, 0]
        fidelity = np.abs(np.vdot(ground_state, system_state))**2
        #success_probs.append(success_prob)
        fidelities.append(fidelity)

        amplitude = np.linalg.norm(system_unnorm)
        success_probs.append(amplitude)
    return success_probs, fidelities

if __name__ == "__main__":
    print(f"[INFO] Using {n_anc} ancilla qubits")
    W = qml.matrix(W_circuit, wire_order=range(n + n_anc))()
    W_dagger = W.conj().T
    R = construct_reflection_operator(n, n_anc)
    # psi0 = W_circuit_select()
    psi_init = prepare_initial_state(n, n_anc)
    psi0 = W @ psi_init

    # Baseline success probability (before OAA)
    reshaped = psi0.reshape((2**n_anc, 2**n))  # [ancilla, system]
    system_unnorm = reshaped[0, :]
    #baseline = np.sum(np.abs(system_unnorm) ** 2)
    baseline = np.linalg.norm(system_unnorm)
    ground_state = np.linalg.eigh(H_norm)[1][:, 0]
    system_state = system_unnorm / np.sqrt(baseline)
    baseline1 = np.abs(np.vdot(ground_state, system_state))**2

    max_iter = 10
    success_probs, fidelities = simulate_oaa(W, W_dagger, R, psi0, max_iter)

    # === Plot ===
    iters = list(range(1, max_iter + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列子图

    # --- 左图 ---
    axes[0].plot(iters, success_probs, marker='o')
    axes[0].hlines(baseline, 1, 10, colors='r', linestyles='--', linewidth=1,
              label='Baseline (no OAA)')
    axes[0].set_title("Success Probability vs Iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Pr[ancilla = 0]")
    axes[0].grid(True)

    # --- 右图 ---
    axes[1].plot(iters, fidelities, marker='s', color='orange')
    axes[1].hlines(baseline1, 1, 10, colors='r', linestyles='--', linewidth=1,
                   label='Baseline (no OAA)')
    axes[1].set_title("Fidelity with Ground State vs Iteration")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Fidelity")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


