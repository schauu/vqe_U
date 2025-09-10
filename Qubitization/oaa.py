#!/usr/bin/env python3

import numpy as np
import pennylane as qml
from scipy.linalg import expm
import matplotlib.pyplot as plt
import math

# === Parameters ===
n = 4
m = 6
n_anc = math.ceil(np.log2(2 * m + 1))
pad_len = 2 ** n_anc

# === Construct TFIM Hamiltonian ===
H = np.zeros((2**n, 2**n))
for i in range(n - 1):
    Z = np.eye(2)
    for j in range(n):
        Z_op = np.array([[1, 0], [0, -1]]) if j in [i, i + 1] else np.eye(2)
        Z = Z_op if j == 0 else np.kron(Z, Z_op)
    H -= Z
for i in range(n):
    X = np.eye(2)
    for j in range(n):
        X_op = np.array([[0, 1], [1, 0]]) if j == i else np.eye(2)
        X = X_op if j == 0 else np.kron(X, X_op)
    H -= X
λ_min, λ_max = np.min(np.linalg.eigvalsh(H)), np.max(np.linalg.eigvalsh(H))
H_norm = (H - λ_min * np.eye(2**n)) / (λ_max - λ_min)

# === LCU + SELECT ===
k_vals = np.arange(-m, m + 1)
alpha_k = 2**(-2 * m) * np.array([math.comb(2 * m, m + k) for k in k_vals])
alpha_k /= np.sum(alpha_k)
weights = np.sqrt(np.pad(alpha_k, (0, pad_len - len(alpha_k))))
U_list = [expm(2j * (k - m) * H_norm) for k in range(len(alpha_k))]
U_list += [np.eye(2**n)] * (pad_len - len(U_list))

# === PennyLane Device & Circuit ===
dev = qml.device("default.qubit", wires=n + n_anc)
@qml.qnode(dev)
def W_circuit():
    for i in range(n):
        qml.Hadamard(i) ## construct psi
    qml.MottonenStatePreparation(weights, wires=range(n, n + n_anc)) ## construct anc
    for idx, U in enumerate(U_list):
        bin_str = format(idx, f"0{n_anc}b")
        ctrl_wires = [n + i for i, b in enumerate(bin_str) if b == '1']
        qml.ControlledQubitUnitary(U, control_wires=ctrl_wires, wires=range(n))

    qml.adjoint(qml.MottonenStatePreparation)(weights, wires=range(n, n + n_anc))
    return qml.state()

def W_circuit1():
    # for i in range(n):
    #     qml.Hadamard(i)
    qml.MottonenStatePreparation(weights, wires=range(n, n + n_anc)) ## construct anc
    for idx, U in enumerate(U_list):
        bin_str = format(idx, f"0{n_anc}b")
        ctrl_wires = [n + i for i, b in enumerate(bin_str) if b == '1']
        if ctrl_wires:
            qml.ControlledQubitUnitary(U, control_wires=ctrl_wires, wires=range(n))
        else:
            qml.QubitUnitary(U, wires=range(n))
    qml.adjoint(qml.MottonenStatePreparation)(weights, wires=range(n, n + n_anc))
    return qml.state()

def prepare_initial_state(n, n_anc):
    psi_sys = np.ones(2**n) / np.sqrt(2**n)
    psi_anc = np.zeros(2**n_anc)
    psi_anc[0] = 1.0
    return np.kron(psi_anc, psi_sys)

def construct_reflection_operator(n, n_anc):
    I_sys = np.eye(2**n)
    P0_anc = np.zeros((2**n_anc, 2**n_anc))
    P0_anc[0, 0] = 1
    R = np.kron(np.eye(2**n_anc) - 2 * P0_anc, I_sys) ## (I_anc - 2|0><0|)\otimes I_sys
    return R

def simulate_oaa(W, W_dagger, R, psi0, max_iter):
    success_probs, fidelities = [], []
    psi = psi0.copy()
    for _ in range(max_iter):
        psi = R @ psi
        psi = W_dagger @ psi
        psi = R @ psi
        psi = W @ psi
        #reshaped = psi.reshape((2**n_anc, 2**n))
        #system_unnorm = reshaped[0, :]
        reshaped = psi.reshape((2**n, 2**n_anc))
        system_unnorm = reshaped[:, 0]
        success_prob = np.sum(np.abs(system_unnorm)**2)
        system_state = system_unnorm / np.sqrt(success_prob)
        ground_state = np.linalg.eigh(H_norm)[1][:, 0]
        fidelity = np.abs(np.vdot(ground_state, system_state))**2
        success_probs.append(success_prob)
        fidelities.append(fidelity)
    return success_probs, fidelities

if __name__ == "__main__":
    print(f"[INFO] Using {n_anc} ancilla qubits")
    W = qml.matrix(W_circuit1, wire_order=range(n + n_anc))()
    W_dagger = W.conj().T
    R = construct_reflection_operator(n, n_anc)
    psi0 = W_circuit()
    #psi0 = prepare_initial_state(n, n_anc)

    max_iter = 10
    success_probs, fidelities = simulate_oaa(W, W_dagger, R, psi0, max_iter)

    # === Plot ===
    iters = list(range(1, max_iter + 1))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(iters, success_probs, marker='o')
    plt.title("Success Probability vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Pr[ancilla = 0]")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(iters, fidelities, marker='s', color='orange')
    plt.title("Fidelity with Ground State vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fidelity")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

