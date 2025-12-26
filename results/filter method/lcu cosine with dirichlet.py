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

# init_state = np.ones(dim) / np.sqrt(dim)

init_state = np.zeros(dim)
init_state[0] = 1

# init_state = np.random.randn(dim) + 1j * np.random.randn(dim)
# init_state /= np.linalg.norm(init_state)

print('overlap init and gs', np.abs(np.vdot(init_state, ground_state))**2)
eigvals, eigvecs = np.linalg.eigh(H)

    # compute overlaps |<E_k | psi_0>|^2
overlaps = np.abs(eigvecs.conj().T @ init_state)**2
H_shifted1 = shift_and_normalize_H(H)
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
    U_list = [expm(-2.0j * k * H_shifted1) for k in k_vals_full]
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

def cosine_filter_matrix1(H, m):
    U = expm(1j * H)
    U_dag = expm(-1j * H)
    cosH = 0.5 * (U + U_dag)
    return np.linalg.matrix_power(cosH, 2*m)

def cosine_filter_matrix(H, m):
    U = expm(1j * H)
    U_dag = expm(-1j * H)
    cosH = 0.5 * (U + U_dag)
    return np.linalg.matrix_power(cosH, 4*m)

def custom_filter_matrix1(H, N):
    return 1/(2*N+1) * sum(expm(-2j * H * i) for i in range(-N, N + 1))

def custom_filter_matrix(H, N):
    return 1/(2*N+1) * sum(expm(-2j * H * i) for i in range(-2*N, 2*N + 1))

def evaluate_fidelity(filter_matrix, init_state, H_shifted):
    psi = filter_matrix @ init_state
    psi /= np.linalg.norm(psi)
    return np.abs(np.vdot(psi, ground_state)) ** 2
    #return np.linalg.norm(psi-ground_state)**2
    #return float(np.real_if_close(np.vdot(psi, H_shifted@psi)))

def cosine_dirichlet_numerical(H, N1, N2):
    cosine_matrix = cosine_filter_matrix1(H, N1)
    dirichlet_matrix = custom_filter_matrix1(H, N2)
    filter_matrix = cosine_matrix @ dirichlet_matrix
    return filter_matrix



def cosine_dirichlet_qc(H_shifted, N1, N2):
    k_vals_full = np.arange(-N1, N1 + 1)
    alpha_k_full = 2 ** (-2 * N1) * np.array([math.comb(2 * N1, N1 + k) for k in k_vals_full])
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



    j_vals_full = np.arange(-N2, N2 + 1)
    beta_j = 1/(2*N2+1)*np.ones(len(j_vals_full))
    pad_len_beta = 2 ** int(np.ceil(np.log2(len(beta_j))))
    beta_padded = np.zeros(pad_len_beta)
    beta_padded[:len(beta_j)] = beta_j
    weights1 = np.sqrt(beta_padded / np.sum(beta_padded))

    weights = np.kron(weights1, weights)
    n_anc2 = int(np.log2(pad_len_beta))

    U_list = [expm(2j * k * H_shifted) for k in k_vals]
    U_list += [np.eye(dim)] * (pad_len - len(U_list))
    U_ops = [qml.QubitUnitary(U, wires=range(n_anc+n_anc2, n_anc + n_anc2+ n)) for U in U_list]

    U_list_diri = [expm(2j * i * H_shifted) for i in j_vals_full]
    U_list_diri += [np.eye(dim)] * (pad_len_beta - len(U_list_diri))
    U_ops_diri = [qml.QubitUnitary(U, wires=range(n_anc+n_anc2, n_anc + n_anc2+ n)) for U in U_list_diri]
    dev = qml.device("default.qubit", wires=n + n_anc + n_anc2)
    @qml.qnode(dev)
    def block_encoding():
        # for i in range(n_anc+n_anc2, n+n_anc+n_anc2):
        #     qml.Hadamard(i)
        qml.StatePrep(weights, wires=range(n_anc+n_anc2))
        qml.Select(U_ops_diri, control=range(n_anc2))
        qml.Select(U_ops, control=range(n_anc2,n_anc2+n_anc))
        qml.adjoint(qml.StatePrep)(weights, wires=range(n_anc+n_anc2))
        return qml.state()

    # U_be = qml.matrix(block_encoding, wire_order=list(range(n_anc+ n_anc2 + n)))()
    # A_block = U_be[:dim, :dim]
    # be_state = A_block @ init_state
    full_state = block_encoding()
    reshaped = full_state.reshape((2 ** (n_anc+n_anc2), 2 ** n))
    system_state = reshaped[0, :]
    be_state = system_state

    #success_prob = np.sum(np.abs(be_state) ** 2)
    be_state /= np.linalg.norm(be_state)
    fidelity = np.abs(np.vdot(be_state, ground_state)) ** 2
    # fig, ax = qml.draw_mpl(block_encoding)()
    # fig.show()
    # return

    return fidelity


def dirichlet_reduced(H_shifted1, max_N):
    k_vals_full = np.arange(-max_N, max_N + 1)
    alpha_k = np.array([1 / (2 * max_N + 1) for k in np.arange(-max_N, 1)])
    beta_k = np.array([1 / (2 * max_N + 1) for k in np.arange(1, max_N + 1)])

    pad_len = 2 ** int(np.ceil(np.log2(len(k_vals_full))))


    alpha_padded = np.zeros(pad_len)
    alpha_padded[:max_N + 1] = alpha_k
    alpha_padded[int(pad_len / 2 + 1):int(pad_len / 2 + 1 + max_N)] = beta_k

    weights = np.sqrt(alpha_padded / np.sum(alpha_padded))

    # #print('weights:', weights)
    n_anc = int(np.log2(pad_len))
    ancilla_wires = list(range(n_anc))
    system_wires = list(range(n_anc, n_anc + n))
    dev = qml.device("default.qubit", wires=n + n_anc)

    @qml.qnode(dev)
    def block_encoding():
        # for i in system_wires:
        #     qml.Hadamard(i)
        # for i in range(n_anc, n+n_anc, 2):
        #     qml.X(i)
        qml.StatePrep(weights, wires=range(n_anc))
        for l in range(1, n_anc):
            qml.ControlledQubitUnitary(
                expm(2.0j * 2 ** l * H_shifted1),
                wires=[0, l] + system_wires,
                control_values=[0, 1],  # sigma=0, anc_l=1

            )

            # --- 施加 Controlled-U^(-k) (对应图中 sigma=1, 即实心点) ---
            # 控制条件: sigma=1 AND anc_l=1
            qml.ControlledQubitUnitary(
                expm(-2.0j * 2 ** l * H_shifted1),
                wires=[0, l] + system_wires,
                control_values=[1, 1],  # sigma=1, anc_l=1

            )
        qml.adjoint(qml.StatePrep)(weights, wires=range(n_anc))
        return qml.state()

    full_state = block_encoding()
    reshaped = full_state.reshape((2 ** n_anc, 2 ** n))
    system_state = reshaped[0, :]
    be_state = system_state

    be_state /= np.linalg.norm(be_state)
    fidelity = np.abs(np.vdot(be_state, ground_state)) ** 2
    return fidelity



# # 扫描 m
max_N = 10
# m_vals = list(range(1, max_N + 1))
# fidelity_cosine, success_cosine = zip(*[fidelity_block_encoding(m) for m in list(range(1, max_N+1))])#list(range(1, max_N//2 + 1))])
#
# N_list = list(range(1, max_N + 1))
# fidelity_be, success_be = zip(*[dirichlet_block_encoding(N) for N in N_list])
#
#
# Fidelity for custom filter as N varies
fidelity_list = []
for N_iter in range(1, max_N + 1):
    H_shifted = shift_and_normalize_H(H)
    fH = custom_filter_matrix(H_shifted, N_iter)
    fidelity_custom = evaluate_fidelity(fH, init_state, H)
    fidelity_list.append(fidelity_custom)

fidelity_cos_list = []
for m_vals in range(1,max_N+1):
    H_shifted = shift_and_normalize_H(H)
    cos_filter = cosine_filter_matrix(H_shifted, m_vals)
    fidelity_cos = evaluate_fidelity(cos_filter, init_state, H)
    fidelity_cos_list.append(fidelity_cos)

fidelity_both_list = []
for m_vals in range(1,max_N+1):
    H_shifted = shift_and_normalize_H(H)
    fH = cosine_dirichlet_numerical(H_shifted, m_vals, m_vals)
    fidelity_custom = evaluate_fidelity(fH, init_state, H)
    fidelity_both_list.append(fidelity_custom)
#

# fidelity for cosine+dirichlet quantum circuit
fidelity_both_qc = []
for m_vals in range(1, max_N+1):
    H_shifted = shift_and_normalize_H(H)
    fidelity = cosine_dirichlet_qc(H_shifted, m_vals, m_vals)
    fidelity_both_qc.append(fidelity)

fidelity_reduced = []
H_shifted = shift_and_normalize_H(H)
for m_vals in range(1, max_N+1):
    fidelity = dirichlet_reduced(H_shifted, m_vals)
    fidelity_reduced.append(fidelity)
#cosine_dirichlet_qc(H, 30, 30)
# #plt.plot(N_list, fidelity_be, marker='s', label='Dirichlet kernel QC')
#
# #plt.plot(list(range(1, max_N + 1)), fidelity_cosine, marker='s', label='Cosine filter QC')
#
plt.plot(range(1, max_N + 1), fidelity_list, label="Dirichlet kernel numerical")

plt.plot(list(range(1, max_N+1)), fidelity_cos_list, label='Cosine numerical')

plt.plot(range(1, max_N+1), fidelity_both_list, label='Dirichlet + Cosine numerical')

plt.plot(range(1, max_N+1), fidelity_both_qc, label='Dirichlet + Cosine QC')

plt.plot(range(1, max_N+1), fidelity_reduced, label='Dirichlet reduced qc')
#
plt.xlabel("M")
plt.xticks(list(range(0, max_N + 1)))
plt.ylabel("Fidelity with Ground State")
#plt.title("Fidelity vs. m")
plt.legend()
# #plt.grid(True)
# #plt.savefig("cosine compare with dirichlet n6.png")
# #plt.show()
#
plt.figure(figsize=(8,4))
plt.stem(eigvals, overlaps, basefmt=" ")
plt.xlabel("Eigenvalue λₖ")
plt.ylabel(r"$|\langle E_k | \psi_0\rangle|^2$")
plt.title("Overlap of initial state with eigenvectors of H")
plt.show()

