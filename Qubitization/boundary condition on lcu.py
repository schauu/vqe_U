import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pennylane as qml

def build_tfim_matrix(n, pbc=False):
    coeffs, ops = [], []
    for i in range(n - 1):
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    if pbc:
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(n - 1) @ qml.PauliZ(0))
    for i in range(n):
        coeffs.append(-1.0)
        ops.append(qml.PauliX(i))
    H = qml.Hamiltonian(coeffs, ops)
    H_mat = qml.matrix(H)
    eigvals = np.linalg.eigvalsh(H_mat)
    H_mat = (H_mat - np.min(eigvals) * np.eye(2 ** n)) / (np.max(eigvals) - np.min(eigvals))
    eigvals = np.linalg.eigvalsh(H_mat)
    H_mat -= np.min(eigvals) * np.eye(2 ** n)
    return H_mat, eigvals

def run_lcu_circuit(n, m, H_mat, ground_state):
    k_vals = np.arange(-m, m + 1)
    alpha_k = 2 ** (-2 * m) * np.array([math.comb(2 * m, m + k) for k in k_vals])
    keep = np.where(alpha_k > 1e-4)[0]
    alpha_k = alpha_k[keep]
    k_vals = k_vals[keep]
    alpha_k /= np.sum(alpha_k)

    pad_len = 2 ** int(np.ceil(np.log2(len(alpha_k))))
    alpha_padded = np.zeros(pad_len)
    alpha_padded[:len(alpha_k)] = alpha_k
    weights = np.sqrt(alpha_padded / np.sum(alpha_padded))

    n_anc = int(np.log2(pad_len))
    dev = qml.device("default.qubit", wires=n + n_anc)

    U_list = [expm(2j * k * H_mat) for k in k_vals]
    U_list += [np.eye(2 ** n)] * (pad_len - len(U_list))

    @qml.qnode(dev)
    def circuit():
        for i in range(n):
            qml.Hadamard(i)
        qml.MottonenStatePreparation(weights, wires=range(n, n + n_anc))
        for idx, U in enumerate(U_list):
            bin_str = format(idx, f"0{n_anc}b")
            ctrl_wires = [n + i for i, b in enumerate(bin_str) if b == '1']
            qml.ControlledQubitUnitary(U, control_wires=ctrl_wires, wires=range(n))
        qml.adjoint(qml.MottonenStatePreparation)(weights, wires=range(n, n + n_anc))
        return qml.state()

    full_state = circuit()
    reshaped = full_state.reshape((2 ** n, 2 ** n_anc))
    system_state = reshaped[:, 0]
    normed_system = system_state / np.linalg.norm(system_state)
    fidelity = np.abs(np.vdot(normed_system, ground_state)) ** 2
    success_prob = np.sum(np.abs(system_state) ** 2)
    return fidelity, success_prob

def compare_obc_pbc(n=4, m_vals=range(1, 13)):
    H_obc, eigs_obc = build_tfim_matrix(n, pbc=False)
    ground_obc = np.linalg.eigh(H_obc)[1][:, 0]
    H_pbc, eigs_pbc = build_tfim_matrix(n, pbc=True)
    ground_pbc =np.linalg.eigh(H_pbc)[1][:, 0]

    fids_obc, succs_obc = [], []
    fids_pbc, succs_pbc = [], []

    for m in m_vals:
        fid_o, suc_o = run_lcu_circuit(n, m, H_obc, ground_obc)
        fid_p, suc_p = run_lcu_circuit(n, m, H_pbc, ground_pbc)
        fids_obc.append(fid_o)
        succs_obc.append(suc_o)
        fids_pbc.append(fid_p)
        succs_pbc.append(suc_p)

    return m_vals, fids_obc, succs_obc, fids_pbc, succs_pbc

import pennylane as qml
import numpy as np

def compute_tfim_spectral_gap(n: int = 8, use_pbc: bool = False) -> float:
    """计算 n 量子比特 TFIM 哈密顿量的谱间隙 Δ = λ1 - λ0

    参数:
        n: 系统比特数
        use_pbc: 是否使用周期边界条件（True 为 PBC，False 为 OBC）

    返回:
        spectral_gap: 谱间隙
    """
    coeffs, ops = [], []
    # Z_i Z_{i+1} 项
    for i in range(n - 1):
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    if use_pbc:
        # 加上 Z_n Z_0
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(n - 1) @ qml.PauliZ(0))
    # X_i 项
    for i in range(n):
        coeffs.append(-1.0)
        ops.append(qml.PauliX(i))

    H = qml.Hamiltonian(coeffs, ops)
    H_mat = qml.matrix(H)
    eigvals = np.linalg.eigvalsh(H_mat)
    gap = eigvals[1] - eigvals[0]
    return gap

# === 运行并绘图 ===
m_vals, fids_obc, succs_obc, fids_pbc, succs_pbc = compare_obc_pbc()
gap_obc = compute_tfim_spectral_gap(n=4, use_pbc=False)
gap_pbc = compute_tfim_spectral_gap(n=4, use_pbc=True)

print("OBC gap:", gap_obc)
print("PBC gap:", gap_pbc)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(m_vals, fids_obc, label='OBC Fidelity', marker='o')
plt.plot(m_vals, fids_pbc, label='PBC Fidelity', marker='s')
plt.xlabel("m")
plt.ylabel("Fidelity")
plt.title("Fidelity vs. m")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(m_vals, succs_obc, label='OBC Success Prob.', marker='o')
plt.plot(m_vals, succs_pbc, label='PBC Success Prob.', marker='s')
plt.xlabel("m")
plt.ylabel("Success Probability")
plt.title("Post-selection Success Probability vs. m")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
