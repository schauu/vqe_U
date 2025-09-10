import numpy as np
import math
import pennylane as qml
from scipy.linalg import expm
import matplotlib.pyplot as plt


n =  4
dim = 2 ** n

init_state = np.zeros(dim)
init_state[int('0101', 2)] = 1.0  # 手动设为 |0101⟩

#init_state = np.ones(dim) / np.sqrt(dim)

# init_state = np.zeros(dim)
# init_state[0] = 1

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
I = np.eye(2)

def heisenberg_xyz_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0):
    dim = 2 ** n
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    def kron_n(op_list):
        result = np.array([[1]])
        for op in op_list:
            result = np.kron(result, op)
        return result

    H = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(n):
        for op, J in zip([X, Y, Z], [Jx, Jy, Jz]):
            ops = [I] * n
            ops[i] = op
            ops[(i + 1) % n] = op  # 周期性边界条件
            H += J * kron_n(ops)

    return H

def kron_n(op_list):
    result = np.array([[1]])
    for op in op_list:
        result = np.kron(result, op)
    return result

# 构造 TFIM H
H = np.zeros((dim, dim), dtype=np.complex128)
for i in range(n):
    Zi = [I] * n
    Zi[i] = Z
    Zi[(i + 1) % n] = Z
    H -= kron_n(Zi)
    Xi = [I] * n
    Xi[i] = X
    H -= kron_n(Xi)


H = heisenberg_xyz_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)
eigvals, eigvecs = np.linalg.eigh(H)
λ_min, λ_max = np.min(eigvals), np.max(eigvals)
H_norm = (H - λ_min * np.eye(dim)) / (λ_max - λ_min)
λ0 = np.min(np.linalg.eigvalsh(H_norm))
H_shifted = H_norm - λ0 * np.eye(dim)
ground_state = np.linalg.eigh(H_shifted)[1][:, 0]
H_shifted1 = H_norm * 10 - λ0 * np.eye(dim)


def fidelity_lcu_circuit(m):
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
    dev = qml.device("default.qubit", wires=n + n_anc)
    U_list = [expm(2j * k * H_shifted1) for k in k_vals]
    U_list += [np.eye(dim)] * (pad_len - len(U_list))

    @qml.qnode(dev)
    def circuit():
        # for i in system_wires:
        #     qml.Hadamard(i)
        for i in range(n_anc, n+n_anc, 2):
            qml.X(i)
        qml.MottonenStatePreparation(weights, wires=ancilla_wires)
        for idx, U in enumerate(U_list):
            bin_str = format(idx, f"0{n_anc}b")
            ctrl_wires = [ancilla_wires[i] for i, b in enumerate(bin_str) if b == '1']
            qml.ControlledQubitUnitary(U, control_wires=ctrl_wires, wires=system_wires)
        qml.adjoint(qml.MottonenStatePreparation)(weights, wires=ancilla_wires)
        return qml.state()

    # U_circuit = qml.matrix(circuit, wire_order=list(range(n_anc)) + list(range(n_anc, n_anc + n)))()
    # A_block = U_circuit[:dim, :dim]
    # system_state = A_block @ init_state

    full_state = circuit()
    reshaped = full_state.reshape((2 ** n_anc, 2 ** n))
    system_state = reshaped[0, :]

    success_prob = np.sum(np.abs(system_state) ** 2)
    normalized_system = system_state / np.linalg.norm(system_state)
    fidelity = np.abs(np.vdot(normalized_system, ground_state)) ** 2
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
        # for i in system_wires:
        #     qml.Hadamard(i)
        for i in range(n_anc, n+n_anc, 2):
            qml.X(i)
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

def cosm(A):
    return 0.5 * (expm(1j * A) + expm(-1j * A))

def fidelity_cos_exact(m):
    cosH = cosm(H_shifted)
    cosH_powered = np.linalg.matrix_power(cosH, 2 * m)
    final_state = cosH_powered @ init_state
    success_prob = np.sum(np.abs(final_state) ** 2)
    final_state /= np.linalg.norm(final_state)
    fidelity = np.abs(np.vdot(final_state, ground_state)) ** 2
    return fidelity, success_prob


# 扫描 m
m_vals = list(range(1, 20))
fidelity_lcu, success_lcu = zip(*[fidelity_lcu_circuit(m) for m in m_vals])
#print('[fidelity lcu]', fidelity_lcu)
fidelity_be, success_be = zip(*[fidelity_block_encoding(m) for m in m_vals])
#print('[fidelity be]', fidelity_be)
fidelity_cos, success_cos = zip(*[fidelity_cos_exact(m) for m in m_vals])
#print('[fidelity cosm]', fidelity_cos)
#
# # 绘图
# # 绘图并保存
plt.figure()
plt.plot(m_vals, fidelity_lcu, marker='o', label='LCU Circuit')
plt.plot(m_vals, fidelity_be, marker='s', label='Block-Encoded')
plt.plot(m_vals, fidelity_cos, marker='^', label='Direct Matrix')
plt.xlabel("m (cos^2m(H))")
plt.xticks(m_vals)
plt.ylabel("Fidelity with Ground State")
plt.title("Fidelity vs. m")
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("fidelity_vs_m.png", dpi=600)
#
# plt.figure()
# plt.plot(m_vals, success_lcu, marker='o', label='LCU Circuit')
# plt.plot(m_vals, success_be, marker='s', label='Block-Encoded')
# plt.plot(m_vals, success_cos, marker='^', label='Direct Matrix')
# plt.xlabel("m (cos^2m(H))")
# plt.xticks(m_vals)
# plt.ylabel("Post-selection Success Probability")
# plt.title("Success Probability vs. m")
# plt.legend()
# plt.grid(True)
# plt.savefig("success_probability_vs_m.png", dpi=600)
#
#
# # 验证初始态与基态几乎正交
# overlap = np.abs(np.vdot(init_state, ground_state))**2
# print("Initial overlap with ground state:", overlap)




