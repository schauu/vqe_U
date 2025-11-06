import numpy as np
from scipy.linalg import expm
import pennylane as qml
import matplotlib.pyplot as plt

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# ===  A = 1/sqrt(2) * (Z + X) ===
P_terms = [Z, X]
A = sum(P_terms) * 1/np.sqrt(2)


coeffs = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
weights = np.sqrt(coeffs) / np.linalg.norm(np.sqrt(coeffs))  # (4,1) vector → 2 ancilla qubits


U_list = P_terms
U_ops = [qml.QubitUnitary(U, wires=[1]) for U in U_list]

# === Prepare |0>_anc ⊗ |+>_sys ===
def prepare_initial_state():
    psi_sys = np.array([1, 1]) / np.sqrt(2)
    psi_anc = np.array([1, 0])
    return np.kron(psi_anc, psi_sys)

# === Reflection operator: R = (2|0⟩⟨0| ⊗ I_sys）-I ===
def construct_reflection_operator():
    I_sys = np.eye(2)
    P0_anc = np.array([[1, 0], [0, 0]])
    R = 2*np.kron(P0_anc, I_sys) - np.eye(4)
    return R

# === Run OAA iterations and record success probabilities ===
def simulate_oaa_success_only(W, W_dagger, R, psi0, max_iter):
    success_probs = []
    psi = psi0.copy()
    for _ in range(max_iter):
        psi = R @ psi
        psi = W_dagger @ psi
        psi = R @ psi
        psi = -W @ psi
        reshaped = psi.reshape((2, 2))  # [ancilla, system]
        system_unnorm = reshaped[0, :]  # ancilla = 0
        success_prob = np.sum(np.abs(system_unnorm)**2)
        success_probs.append(success_prob)
        print('[psi is]', psi)
    return success_probs

dev = qml.device("default.qubit")
@qml.qnode(dev)
def W_circuit_select():
    qml.StatePrep(weights, wires=[0])
    qml.Select(U_ops, control=[0])
    qml.adjoint(qml.StatePrep)(weights, wires=[0])
    return qml.state()
# === Main ===
W = qml.matrix(W_circuit_select, wire_order=[0,1])()  # ancilla first
W_dagger = W.conj().T
R = construct_reflection_operator()

psi_init = prepare_initial_state()
psi0 = W @ psi_init  # initial state = W |0>_anc ⊗ |+>_sys

# Baseline success probability (before OAA)
reshaped = psi0.reshape((2, 2))  # [ancilla, system]
system_unnorm = reshaped[0, :]
baseline = np.sum(np.abs(system_unnorm)**2)

# OAA
success_probs = simulate_oaa_success_only(W, W_dagger, R, psi0.copy(), max_iter=10)

# === Plot ===
fig, ax = plt.subplots(figsize=(8, 6), dpi=90)

ax.plot(range(1, 11), success_probs, marker='o', markersize=4, linewidth=1,
        label='OAA')
ax.hlines(baseline, 1, 10, colors='r', linestyles='--', linewidth=1,
          label='Baseline (no OAA)')

ax.set_title("Success Probability vs Iteration", fontsize=10)
ax.set_xlabel("Iteration", fontsize=9)
ax.set_ylabel("Pr[ancilla = 0]", fontsize=9)

ax.grid(True, linewidth=0.5)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
