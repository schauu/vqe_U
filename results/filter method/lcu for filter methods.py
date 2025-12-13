#!/usr/bin/env python3

import numpy as np
import pennylane as qml
from scipy.linalg import expm
import matplotlib.pyplot as plt
import math


n = 4
dim = 2 ** n

def get_ground_state(H):
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs[:, np.argmin(eigvals)]

def shift_and_normalize_H(H):
    eigvals = np.linalg.eigvalsh(H)
    λ_min, λ_max = np.min(eigvals), np.max(eigvals)
    H_norm = (H - λ_min * np.eye(len(H))) / (λ_max - λ_min) * np.pi /2
    λ0 = np.min(np.linalg.eigvalsh(H_norm))
    H_shifted = H_norm - λ0 * np.eye(len(H))
    return H_shifted

def shift_and_normalize_H1(H):
    eigvals = np.linalg.eigvalsh(H)
    λ_min, λ_max = np.min(eigvals), np.max(eigvals)
    H_norm = (H - λ_min * np.eye(len(H))) / (λ_max - λ_min) * np.pi
    λ0 = np.min(np.linalg.eigvalsh(H_norm))
    H_shifted = H_norm - λ0 * np.eye(len(H))
    return H_shifted

def kron_n(op_list):
    result = np.array([[1]])
    for op in op_list:
        result = np.kron(result, op)
    return result

# TFIM Hamiltonian
def tfim_model(n):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I = np.eye(2)
    for i in range(n):
        Zi = [I] * n
        Zi[i] = Z
        Zi[(i + 1) % n] = Z
        H -= kron_n(Zi)
        Xi = [I] * n
        Xi[i] = X
        H -= kron_n(Xi)
    return H



# 归一化并 shift
H = tfim_model(n)
H_shifted = shift_and_normalize_H(H)
ground_state = get_ground_state(H_shifted)
init_state = np.ones(dim) / np.sqrt(dim)

H_shifted1 = shift_and_normalize_H1(H)
def dirichlet_block_encoding(max_N):
    k_vals_full = np.arange(-max_N, max_N + 1)
    alpha_k = np.array([1/(2* max_N + 1) for k in k_vals_full])

    pad_len = 2 ** int(np.ceil(np.log2(len(alpha_k))))
    alpha_padded = np.zeros(pad_len)
    alpha_padded[:len(alpha_k)] = alpha_k
    weights = np.sqrt(alpha_padded / np.sum(alpha_padded))
    #weights = alpha_padded
    #print('weights:', weights)
    n_anc = int(np.log2(pad_len))
    ancilla_wires = list(range(n_anc))
    system_wires = list(range(n_anc, n_anc + n))
    U_list = [expm(-1.0j * k * H_shifted1) for k in k_vals_full]
    U_list += [np.eye(dim)] * (pad_len - len(U_list))
    U_ops = [qml.QubitUnitary(U, wires=range(n_anc, n_anc + n)) for U in U_list]
    dev = qml.device("default.qubit", wires=n + n_anc)

    @qml.qnode(dev)
    def block_encoding():
        for i in system_wires:
            qml.Hadamard(i)
        # for i in range(n_anc, n+n_anc, 2):
        #     qml.X(i)
        qml.StatePrep(weights, wires=range(n_anc))
        qml.Select(U_ops, control=range(n_anc))
        qml.adjoint(qml.StatePrep)(weights, wires=range(n_anc))
        return qml.state()

    # U_be = qml.matrix(block_encoding, wire_order=list(range(n_anc)) + list(range(n_anc, n_anc + n)))()
    # A_block = U_be[:dim, :dim]
    # be_state = A_block @ init_state
    full_state = block_encoding()
    reshaped = full_state.reshape((2 ** n_anc, 2 ** n))
    system_state = reshaped[0, :]
    be_state = system_state

    success_prob = np.sum(np.abs(be_state) ** 2)
    be_state /= np.linalg.norm(be_state)
    fidelity = np.abs(np.vdot(be_state, ground_state)) ** 2
    return fidelity, success_prob


def fidelity_block_encoding(m):
    k_vals_full = np.arange(-m, m + 1)
    alpha_k_full = 2 ** (-2 * m) * np.array([math.comb(2 * m, m + k) for k in k_vals_full])
    keep = [i for i in range(len(alpha_k_full)) if alpha_k_full[i] > 1e-4]
    alpha_k = alpha_k_full[keep]
    k_vals = k_vals_full[keep]
    alpha_k /= np.sum(alpha_k)
    pad_len = 2 ** int(np.ceil(np.log2(len(alpha_k))))
    alpha_padded = np.zeros(pad_len)
    alpha_padded[:len(alpha_k)] = alpha_k
    weights = np.sqrt(alpha_padded / np.sum(alpha_padded))
    n_anc = int(np.log2(pad_len))
    ancilla_wires = list(range(n_anc))
    system_wires = list(range(n_anc, n_anc + n))
    U_list = [expm(2j * k * H_shifted) for k in k_vals]
    U_list += [np.eye(dim)] * (pad_len - len(U_list))
    U_ops = [qml.QubitUnitary(U, wires=range(n_anc, n_anc + n)) for U in U_list]
    dev = qml.device("default.qubit", wires=n + n_anc)

    @qml.qnode(dev)
    def block_encoding():
        for i in system_wires:
            qml.Hadamard(i)
        # for i in range(n_anc, n+n_anc, 2):
        #     qml.X(i)
        qml.StatePrep(weights, wires=range(n_anc))
        qml.Select(U_ops, control=range(n_anc))
        qml.adjoint(qml.StatePrep)(weights, wires=range(n_anc))
        return qml.state()

    # U_be = qml.matrix(block_encoding, wire_order=list(range(n_anc)) + list(range(n_anc, n_anc + n)))()
    # A_block = U_be[:dim, :dim]
    # be_state = A_block @ init_state
    full_state = block_encoding()
    reshaped = full_state.reshape((2 ** n_anc, 2 ** n))
    system_state = reshaped[0, :]
    be_state = system_state

    success_prob = np.sum(np.abs(be_state) ** 2)
    be_state /= np.linalg.norm(be_state)
    fidelity = np.abs(np.vdot(be_state, ground_state)) ** 2
    return fidelity, success_prob

def cosine_filter_matrix(H, m):
    U = expm(1j * H)
    U_dag = expm(-1j * H)
    cosH = 0.5 * (U + U_dag)
    return np.linalg.matrix_power(cosH, 2*m)

def custom_filter_matrix(H, N):
    return 1/(2*N+1) * sum(expm(-1j * H * i) for i in range(-N, N + 1)) #/ N  # corrected j = N+1 to 2N

def evaluate_fidelity(filter_matrix, init_state, H_shifted):
    psi = filter_matrix @ init_state
    psi /= np.linalg.norm(psi)
    return np.abs(np.vdot(psi, ground_state)) ** 2
    #return np.linalg.norm(psi-ground_state)**2
    #return float(np.real_if_close(np.vdot(psi, H_shifted@psi)))



# 扫描 m
max_N = 20
m_vals = list(range(1, max_N + 1))
fidelity_cosine, success_cosine = zip(*[fidelity_block_encoding(m) for m in list(range(1, max_N+1))])#list(range(1, max_N//2 + 1))])

N_list = list(range(1, max_N + 1))
fidelity_be, success_be = zip(*[dirichlet_block_encoding(N) for N in N_list])


# Fidelity for custom filter as N varies
fidelity_list = []
for N_iter in range(1, max_N + 1):
    H_shifted = shift_and_normalize_H1(H)
    fH = custom_filter_matrix(H_shifted, N_iter)
    fidelity_custom = evaluate_fidelity(fH, init_state, H)
    fidelity_list.append(fidelity_custom)

fidelity_cos_list = []
for m_vals in range(1,max_N+1):
    H_shifted = shift_and_normalize_H(H)
    cos_filter = cosine_filter_matrix(H_shifted, m_vals)
    fidelity_cos = evaluate_fidelity(cos_filter, init_state, H)
    fidelity_cos_list.append(fidelity_cos)


plt.plot(N_list, fidelity_be, marker='s', label='Dirichlet kernel QC')
#plt.plot(list(range(1, max_N + 1, 2)), fidelity_cosine, marker='s', label='Cosine filter')
plt.plot(list(range(1, max_N + 1)), fidelity_cosine, marker='s', label='Cosine filter QC')

plt.plot(range(1, max_N + 1), fidelity_list, label="Dirichlet kernel numerical")
#plt.axhline(y=fidelity_cos, color='r', linestyle='--', label="cos^{2m}(H) Fidelity")
plt.plot(list(range(1, max_N+1)), fidelity_cos_list, label='Cosine kernel numerical')

plt.xlabel("M")
plt.xticks(list(range(0, max_N + 1)))
plt.ylabel("Fidelity with Ground State")
#plt.title("Fidelity vs. m")
plt.legend()
#plt.grid(True)
plt.savefig("cosine compare with dirichlet n6.png")
plt.show()
