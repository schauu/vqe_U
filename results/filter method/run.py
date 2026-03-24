#!/usr/bin/env python3
from itertools import zip_longest
import numpy as np
import pennylane as qml
from scipy.linalg import expm
import matplotlib.pyplot as plt
import math
import time

n = 8
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


start_time = time.time()
# 归一化并 shift
H = tfim_model(n)
H_shifted = shift_and_normalize_H(H)
ground_state = get_ground_state(H_shifted)

init_state = np.ones(dim) / np.sqrt(dim)

# init_state = np.zeros(dim)
# init_state[0] = 1

# init_state = np.random.randn(dim) + 1j * np.random.randn(dim)
# init_state /= np.linalg.norm(init_state)

print('overlap init and gs', np.abs(np.vdot(init_state, ground_state))**2)
eigvals, eigvecs = np.linalg.eigh(H)

    # compute overlaps |<E_k | psi_0>|^2
overlaps = np.abs(eigvecs.conj().T @ init_state)**2
H_shifted1 = shift_and_normalize_H(H)
eigvals, eigvecs = np.linalg.eigh(H_shifted1)

def fast_expm(k):
    # 利用公式：exp(i * k * VΛV†) = V * exp(i * k * Λ) * V†
    # 其中 Λ 是对角阵，计算指数只需对角元素求指数，极快且精确
    phase = np.exp(2.0j * k * eigvals)
    return (eigvecs * phase) @ eigvecs.conj().T

# def cosine_filter_matrix1(H, m):
#     U = expm(1j * H)
#     U_dag = expm(-1j * H)
#     cosH = 0.5 * (U + U_dag)
#     return np.linalg.matrix_power(cosH, 2*m)

def cosine_filter_matrix1(H, m):
    U = fast_expm(0.5)
    U_dag = fast_expm(-0.5)
    cosH = 0.5 * (U + U_dag)
    return np.linalg.matrix_power(cosH, 2*m)



# def custom_filter_matrix1(H, N):
#     return 1/(2*N+1) * sum(expm(-2j * H * i) for i in range(-N, N + 1))


def custom_filter_matrix1(H, N):
    return 1/(2*N+1) * sum(fast_expm(-i) for i in range(-N, N + 1))


def evaluate_fidelity(filter_matrix, init_state, H_shifted):
    psi = filter_matrix @ init_state
    psi /= np.linalg.norm(psi)
    return np.abs(np.vdot(psi, ground_state)) ** 2
    #return np.linalg.norm(psi-ground_state)**2
    #return float(np.real_if_close(np.vdot(psi, H_shifted@psi)))
def dirichlet_reduced(H_shifted1, max_N):
    n_anc = int(np.ceil(np.log2(max_N+1))) + 1
    pad_len = 2 ** n_anc

    alpha_padded = np.zeros(pad_len)

    # NOTE: bit #0 is the sigma bit
    # NOTE: bit #1 to bit #n_anc-1 is the index

    # slower version but clearer
    for i in range(max_N+1):
        alpha_padded[2*i+0] = 1 / (2 * max_N + 1)
        if i != 0:
            alpha_padded[2*i+1] = 1 / (2 * max_N + 1)

    # faster version
    # alpha_padded[:2*(max_N+1)] = 1 / (2 * max_N+1)
    # alpha_padded[1] = 0

    weights = np.sqrt(alpha_padded / np.sum(alpha_padded))
    ancilla_wires = list(range(n_anc))
    system_wires = list(range(n_anc, n_anc + n))
    dev = qml.device("default.qubit", wires=n + n_anc)

    @qml.qnode(dev)
    def block_encoding():
        for i in system_wires:
            qml.Hadamard(i)
        qml.StatePrep(weights, wires=range(n_anc))
        # NOTE: the sigma, which is at bit #0, is now at qubit n_anc-1
        # NOTE: the index, which is at bit #1 to #n_anc-1, is now at qubit #n_anc-2 to 0 (backwards)

        # also the l here is confusing, I change it to mean the 2**l part in exmp
        for l in range(0, n_anc-1):
            qml.ControlledQubitUnitary(
                #expm(2.0j * 2 ** l * H_shifted1),
                fast_expm(2**l),
                wires=[n_anc-l-2, n_anc-1] + system_wires,
                control_values=[1, 0],  # sigma=0, anc_l=1
            )
            qml.ControlledQubitUnitary(
                #expm(-2.0j * 2 ** l * H_shifted1),
                fast_expm(-2 ** l),
                wires=[n_anc-l-2, n_anc-1] + system_wires,
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

    # fig, ax = qml.draw_mpl(block_encoding)()
    # fig.savefig(f"circuit_reduced_{max_N}.png", dpi=600)

    return fidelity

def cosine_reduced(H_shifted1, max_N):
    n_anc = int(np.ceil(np.log2(max_N+1))) + 1
    pad_len = 2 ** n_anc

    alpha_padded = np.zeros(pad_len)

    # NOTE: bit #0 is the sigma bit
    # NOTE: bit #1 to bit #n_anc-1 is the index

    # slower version but clearer
    for i in range(max_N+1):
        alpha_padded[2*i+0] = 2 ** (-2 * max_N) * math.comb(2 * max_N, max_N + i)
        if i != 0:
            alpha_padded[2*i+1] = 2 ** (-2 * max_N) * math.comb(2 * max_N, max_N + i)

    # faster version
    # alpha_padded[:2*(max_N+1)] = 1 / (2 * max_N+1)
    # alpha_padded[1] = 0

    weights = np.sqrt(alpha_padded / np.sum(alpha_padded))
    ancilla_wires = list(range(n_anc))
    system_wires = list(range(n_anc, n_anc + n))
    dev = qml.device("default.qubit", wires=n + n_anc)

    @qml.qnode(dev)
    def block_encoding():
        for i in system_wires:
            qml.Hadamard(i)
        qml.StatePrep(weights, wires=range(n_anc))
        # NOTE: the sigma, which is at bit #0, is now at qubit n_anc-1
        # NOTE: the index, which is at bit #1 to #n_anc-1, is now at qubit #n_anc-2 to 0 (backwards)

        # also the l here is confusing, I change it to mean the 2**l part in exmp
        for l in range(0, n_anc-1):
            qml.ControlledQubitUnitary(
                #expm(2.0j * 2 ** l * H_shifted1),
                fast_expm(2**l),
                wires=[n_anc-l-2, n_anc-1] + system_wires,
                control_values=[1, 0],  # sigma=0, anc_l=1
            )
            qml.ControlledQubitUnitary(
                #expm(-2.0j * 2 ** l * H_shifted1),
                fast_expm(-2 ** l),
                wires=[n_anc-l-2, n_anc-1] + system_wires,
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

    # fig, ax = qml.draw_mpl(block_encoding)()
    # fig.savefig(f"circuit_reduced_{max_N}.png", dpi=600)

    return fidelity
def cosine_dirichlet_numerical(H, N1, N2):
    cosine_matrix = cosine_filter_matrix1(H, N1)
    dirichlet_matrix = custom_filter_matrix1(H, N2)
    filter_matrix = cosine_matrix @ dirichlet_matrix
    return filter_matrix


def cosine_dirichlet_reduced(H_shifted1, N1, N2):
    n_anc = int(np.ceil(np.log2(N1+1))) + 1
    n_anc_beta = int(np.ceil(np.log2(N2+1))) + 1
    pad_len = 2 ** n_anc
    beta_len = 2 ** n_anc_beta

    alpha_padded = np.zeros(pad_len)
    beta_padded = np.zeros(beta_len)

    # NOTE: bit #0 is the sigma bit
    # NOTE: bit #1 to bit #n_anc-1 is the index

    #slower version but clearer
    for i in range(N1+1):
        alpha_padded[2*i+0] = 2 ** (-2 * N1) * math.comb(2 * N1, N1 + i)
        if i != 0:
            alpha_padded[2*i+1] = 2 ** (-2 * N1) * math.comb(2 * N1, N1 + i)

    for i in range(N2+1):
        beta_padded[2*i+0] = 1 / (2 * N2 + 1)
        if i != 0:
            beta_padded[2*i+1] = 1 / (2 * N2 + 1)

    # for i in range(N1+1):
    #     alpha_padded[2*i+0] = 2 ** (-2 * N2) * math.comb(2 * N2, N2 + i)
    #     if i != 0:
    #         alpha_padded[2*i+1] = 2 ** (-2 * N2) * math.comb(2 * N2, N2 + i)
    #
    # for i in range(N2+1):
    #     beta_padded[2*i+0] = 1 / (2 * N1 + 1)
    #     if i != 0:
    #         beta_padded[2*i+1] = 1 / (2 * N1 + 1)

    # faster version
    # alpha_padded[:2*(max_N+1)] = 1 / (2 * max_N+1)
    # alpha_padded[1] = 0

    weights = np.sqrt(alpha_padded / np.sum(alpha_padded))
    weights1 = np.sqrt(beta_padded / np.sum(beta_padded))

    #weights = np.kron(weights1, weights)
    weights = np.kron(weights, weights1)
    ancilla_wires = list(range(n_anc+n_anc_beta))
    system_wires = list(range(n_anc+n_anc_beta, n_anc +n_anc_beta+ n))
    dev = qml.device("default.qubit", wires=n + n_anc+n_anc_beta)

    @qml.qnode(dev)
    def block_encoding():
        for i in system_wires:
            qml.Hadamard(i)
        qml.StatePrep(weights, wires=range(n_anc+n_anc_beta))
        # NOTE: the sigma, which is at bit #0, is now at qubit n_anc-1
        # NOTE: the index, which is at bit #1 to #n_anc-1, is now at qubit #n_anc-2 to 0 (backwards)

        # also the l here is confusing, I change it to mean the 2**l part in exmp
        sigma_wire_alpha = n_anc - 1
        for l in range(0, n_anc-1):
            anc_l_wire_alpha = n_anc - l - 2
            qml.ControlledQubitUnitary(
                #expm(2.0j * 2 ** l * H_shifted1),
                fast_expm(2**l),
                wires=[anc_l_wire_alpha, sigma_wire_alpha] + system_wires,
                control_values=[1, 0],  # sigma=0, anc_l=1
            )
            qml.ControlledQubitUnitary(
                #expm(-2.0j * 2 ** l * H_shifted1),
                fast_expm(-2 ** l),
                wires=[anc_l_wire_alpha, sigma_wire_alpha] + system_wires,
                control_values=[1, 1],  # sigma=1, anc_l=1
            )
        sigma_wire_beta = n_anc+n_anc_beta - 1
        for l in range(0, n_anc_beta-1):
            anc_l_wire_beta = n_anc+n_anc_beta - l - 2
            qml.ControlledQubitUnitary(
                #expm(2.0j * 2 ** l * H_shifted1),
                fast_expm(2**l),
                wires=[anc_l_wire_beta, sigma_wire_beta] + system_wires,
                control_values=[1, 0],  # sigma=0, anc_l=1
            )
            qml.ControlledQubitUnitary(
                #expm(-2.0j * 2 ** l * H_shifted1),
                fast_expm(-2 ** l),
                wires=[anc_l_wire_beta, sigma_wire_beta] + system_wires,
                control_values=[1, 1],  # sigma=1, anc_l=1
            )

        qml.adjoint(qml.StatePrep)(weights, wires=range(n_anc+n_anc_beta))
        return qml.state()

    full_state = block_encoding()
    reshaped = full_state.reshape((2 ** (n_anc+n_anc_beta), 2 ** n))
    system_state = reshaped[0, :]
    be_state = system_state

    be_state /= np.linalg.norm(be_state)
    fidelity = np.abs(np.vdot(be_state, ground_state)) ** 2

    # fig, ax = qml.draw_mpl(block_encoding)()
    # fig.savefig(f"circuit_reduced_{max_N}.png", dpi=600)

    return fidelity
# # 扫描 m

# # 扫描 m
max_N = 200 # dirichlet steps
Delta = eigvals[1]
N2 = int(np.ceil(np.pi/Delta-1)*0.5)

fidelity_both_reduced = []
for m_vals in range(0,max_N+1,2):
    H_shifted = shift_and_normalize_H(H)
    if m_vals <= N2:
        fidelity_custom = cosine_dirichlet_reduced(H_shifted, 0, m_vals) # second argument is cosine step
        fidelity_both_reduced.append(fidelity_custom)
    #fidelity_custom = cosine_dirichlet_reduced(H_shifted, int(m_vals*0.1), int(m_vals*0.9))
    else:
        fidelity_custom = cosine_dirichlet_reduced(H_shifted, m_vals-N2, N2)
        fidelity_both_reduced.append(fidelity_custom)

fidelity_list = []

for m_vals in range(0, max_N+1, 2):
    H_shifted = shift_and_normalize_H(H)
    if m_vals <= N2:
        fH = custom_filter_matrix1(H_shifted, m_vals)
        psi = fH @ init_state
        psi /= np.linalg.norm(psi)
    #fH = cosine_dirichlet_numerical(H_shifted,int(N_iter*0.9), int(N_iter*0.1))
        fidelity_custom = np.abs(np.vdot(psi, ground_state)) ** 2
        fidelity_list.append(fidelity_custom)
    else:
        fH = custom_filter_matrix1(H_shifted, N2)
        psi = fH @ init_state
        psi /= np.linalg.norm(psi)
        fH1 = cosine_filter_matrix1(H_shifted, m_vals-N2)
        psi1 = fH1 @ psi
        psi1 /= np.linalg.norm(psi1)
        fidelity_cos = np.abs(np.vdot(psi1, ground_state)) ** 2
        fidelity_list.append(fidelity_cos)


fidelity_cos_num = []
for N_iter in range(0, max_N + 1,2):
    H_shifted = shift_and_normalize_H(H)
    fH = cosine_filter_matrix1(H_shifted, N_iter)
    fidelity_custom = evaluate_fidelity(fH, init_state, H)
    fidelity_cos_num.append(fidelity_custom)

fidelity_diri_num = []
for N_iter in range(0, max_N + 1,2):
    H_shifted = shift_and_normalize_H(H)
    fH = custom_filter_matrix1(H_shifted, N_iter)
    fidelity_custom = evaluate_fidelity(fH, init_state, H)
    fidelity_diri_num.append(fidelity_custom)

fidelity_cos_reduced = []
for m_vals in range(0,max_N+1,2):
    H_shifted = shift_and_normalize_H(H)
    fidelity_cos = cosine_reduced(H_shifted, m_vals)
    fidelity_cos_reduced.append(fidelity_cos)

fidelity_reduced = []
H_shifted = shift_and_normalize_H(H)
for m_vals in range(0, max_N+1,2):
    fidelity = dirichlet_reduced(H_shifted, m_vals)
    fidelity_reduced.append(fidelity)
plt.plot(list(range(0, max_N + 1,2)), fidelity_list,marker='o', ms=2,linestyle='-',label='Dirichlet + Cosine numerical')
plt.plot(list(range(0, max_N + 1,2)), fidelity_both_reduced,marker='x', ms=2,linestyle='-',label='Dirichlet + Cosine reduced qc')
plt.plot(range(0, max_N+1, 2), fidelity_cos_num, marker='x', linestyle='-',label="cosine filter numerical")
plt.plot(range(0, max_N + 1,2), fidelity_diri_num, marker='x', linestyle='-',label="Dirichlet kernel numerical")
plt.plot(range(0, max_N+1, 2), fidelity_cos_reduced, marker='x', linestyle='-',label="cosine filter qc")
plt.plot(range(0, max_N + 1,2), fidelity_reduced, marker='x', linestyle='-',label="Dirichlet kernel qc")

#
plt.xlabel("Time step")
plt.xticks(list(range(0, max_N + 1,10)))
plt.ylabel("Fidelity with Ground State")
#plt.title("Fidelity vs. m")
plt.legend()
# #plt.grid(True)
plt.savefig("cosine_with_dirichlet_n8.png")
# #plt.show()
#
plt.figure(figsize=(8,4))
plt.stem(eigvals, overlaps, basefmt=" ")
plt.xlabel("Eigenvalues λ")
plt.ylabel(r"$|\langle E_k | \psi_0\rangle|^2$")
plt.title("Overlap of initial state with eigenvectors of H")
#plt.show()
plt.savefig("overlap_n8.png")
headers = ("Dirichlet + Cosine numerical", "Dirichlet + Cosine reduced qc", "cosine filter numerical", "Dirichlet kernel numerical")

all_lists = [fidelity_list, fidelity_both_reduced, fidelity_cos_reduced, fidelity_reduced]

file_name = "fidelity_data.txt"

with open(file_name, "w", encoding="utf-8") as f:

    f.write("\t".join(headers) + "\n")

    for row in zip_longest(*all_lists, fillvalue=""):

        line = "\t".join(map(str, row))
        f.write(line + "\n")

print("data has been stored")

