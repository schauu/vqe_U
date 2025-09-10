import numpy as np
import math
import pennylane as qml
from scipy.linalg import expm
import matplotlib.pyplot as plt


# 设置系统大小
n = 4
dim = 2 ** n
init_state = np.ones(dim) / np.sqrt(dim)

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
I = np.eye(2)

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

eigvals, eigvecs = np.linalg.eigh(H)
λ_min, λ_max = np.min(eigvals), np.max(eigvals)
H_norm = (H - λ_min * np.eye(dim)) / (λ_max - λ_min)
λ0 = np.min(np.linalg.eigvalsh(H_norm))
H_shifted = H_norm - λ0 * np.eye(dim)
ground_state = np.linalg.eigh(H_shifted)[1][:, 0]

def compare(m):
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
    U_list = [expm(2j * k * H_shifted) for k in k_vals]
    U_list += [np.eye(dim)] * (pad_len - len(U_list))
    ancilla_wires = list(range(n_anc))
    system_wires = list(range(n_anc, n_anc + n))
    U_ops = [qml.QubitUnitary(U, wires=range(n_anc, n_anc + n)) for U in U_list]

    dev = qml.device("default.qubit", wires=n + n_anc)
    dev1 = qml.device("default.qubit", wires=n + n_anc)

    @qml.qnode(dev)
    def circuit_precise():
        qml.StatePrep(weights, wires=ancilla_wires)
        for idx, U in enumerate(U_ops):
            bin_str = format(idx, f"0{n_anc}b")

            # ctrl_wires = [ancilla_wires[i] for i, b in enumerate(bin_str) if b == '1']
            # print('[CRTL Wires]', ctrl_wires)
            # qml.ControlledQubitUnitary(U, control_wires=ctrl_wires, wires=system_wires)
            ctrl_values = [int(b) for b in bin_str]
            qml.ControlledQubitUnitary(
                U,
                control_wires=ancilla_wires,
                control_values=ctrl_values,
                wires=system_wires
            )

        qml.adjoint(qml.StatePrep)(weights, wires=ancilla_wires)
        return qml.state()

    @qml.qnode(dev1)
    def block_encoding():
        qml.StatePrep(weights, wires=ancilla_wires)
        qml.Select(U_ops, control=range(n_anc))
        qml.adjoint(qml.StatePrep)(weights, wires=ancilla_wires)
        return qml.state()

    H_be = qml.matrix(circuit_precise, wire_order=ancilla_wires + system_wires)()
    A_block = H_be[:dim, :dim]
    be_state = A_block @ init_state
    be_state /= np.linalg.norm(be_state)
    fidelity1 = np.abs(np.vdot(be_state, ground_state)) ** 2
    print('fidelity1', fidelity1)

    H_lcu = qml.matrix(block_encoding, wire_order=ancilla_wires + system_wires)()
    A_block = H_lcu[:dim, :dim]
    be_state = A_block @ init_state
    be_state /= np.linalg.norm(be_state)
    fidelity2 = np.abs(np.vdot(be_state, ground_state)) ** 2
    print('fidelity2', fidelity2)
    return None


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
    U_list = [expm(2j * k * H_shifted) for k in k_vals]
    U_list += [np.eye(dim)] * (pad_len - len(U_list))
    ancilla_wires = list(range(n_anc))
    system_wires = list(range(n_anc, n_anc + n))
    U_ops = [qml.QubitUnitary(U, wires=range(n_anc, n_anc + n)) for U in U_list]
    dev = qml.device("default.qubit", wires=n + n_anc)
    # print('[U_be]', U_list)
    print('[Weights]', weights)

    @qml.qnode(dev)
    def block_encoding():
        #qml.StatePrep(weights, wires=range(n_anc))
        qml.MottonenStatePreparation(weights, wires=ancilla_wires)
        qml.Select(U_ops, control=range(n_anc))
        # qml.adjoint(qml.StatePrep)(weights, wires=range(n_anc))
        return qml.state()

    # circuit_diagram = qml.draw_mpl(block_encoding, style='pennylane')
    # print(circuit_diagram())

    # U_be = qml.matrix(block_encoding, wire_order=list(range(n_anc)) + list(range(n_anc, n_anc + n)))()
    # print('[U_be]', U_be)
    # A_block = U_be[:dim, :dim]
    # be_state = A_block @ init_state
    # success_prob = np.sum(np.abs(be_state) ** 2)
    # be_state /= np.linalg.norm(be_state)
    # fidelity = np.abs(np.vdot(be_state, ground_state)) ** 2
    state = block_encoding()
    state = state / np.linalg.norm(state)
    return state

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
    U_list = [expm(2j * k * H_shifted) for k in k_vals]
    U_list += [np.eye(dim)] * (pad_len - len(U_list))
    ancilla_wires = list(range(n_anc))
    system_wires = list(range(n_anc, n_anc + n))
    U_ops = [qml.QubitUnitary(U, wires=range(n_anc, n_anc + n)) for U in U_list]
    dev = qml.device("default.qubit", wires=n + n_anc)

    @qml.qnode(dev)
    def circuit_precise():
        qml.MottonenStatePreparation(weights, wires=ancilla_wires)
        for idx, U in enumerate(U_ops):
            bin_str = format(idx, f"0{n_anc}b")
            ctrl_values = [int(b) for b in bin_str]
            qml.ControlledQubitUnitary(
                U,
                control_wires=ancilla_wires,
                control_values=ctrl_values,
                wires=system_wires
            )
        qml.adjoint(qml.MottonenStatePreparation(weights, wires=ancilla_wires))
        return qml.state()
    circuit_diagram = qml.draw_mpl(circuit_precise, style='pennylane')
    print(circuit_diagram())
    return None


#compare(m=6)
