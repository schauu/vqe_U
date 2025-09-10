import numpy as np
import math
import pennylane as qml
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict


def lcu_cos2m_approx(H, m, alpha_thresh=1e-4):
    """
    构造通过 LCU 方法近似 cos^{2m}(H) 的矩阵

    参数:
        H: numpy array，目标 Hamiltonian
        m: int，余弦次数
        alpha_thresh: float，忽略小权重的阈值

    返回:
        U_lcu: numpy array，LCU 线性组合矩阵
    """
    k_vals_full = np.arange(-m, m + 1)
    alpha_k_full = 2 ** (-2 * m) * np.array([math.comb(2 * m, m + k) for k in k_vals_full])

    keep = [i for i in range(len(alpha_k_full)) if alpha_k_full[i] > alpha_thresh]
    alpha_k = alpha_k_full[keep]
    k_vals = k_vals_full[keep]
    alpha_k /= np.sum(alpha_k)  # 归一化

    U_lcu = sum(a * expm(2j * k * H) for a, k in zip(alpha_k, k_vals))
    return U_lcu


def true_cos2m_matrix(H, m):
    """
    构造真正的 cos^{2m}(H) 矩阵

    参数:
        H: numpy array，Hamiltonian
        m: int，幂次数

    返回:
        cos2m: numpy array，真实矩阵
    """
    H_cos = expm(1.0j * H)
    H_cos_dag = expm(-1.0j * H)
    cos2H = 0.5 * (H_cos + H_cos_dag)
    return np.linalg.matrix_power(cos2H, 2 * m)

def get_cos2m_via_controlled_product(H, m, threshold=1e-4):
    """用 activated(b) 的乘积逻辑计算近似矩阵"""
    import itertools
    k_vals_full = np.arange(-m, m + 1)
    alpha_k_full = 2 ** (-2 * m) * np.array([math.comb(2 * m, m + k) for k in k_vals_full])
    keep = [i for i in range(len(alpha_k_full)) if alpha_k_full[i] > threshold]
    alpha_k = alpha_k_full[keep]
    k_vals = k_vals_full[keep]
    alpha_k /= np.sum(alpha_k)

    # Padding到最近的2的幂
    pad_len = 2 ** int(np.ceil(np.log2(len(alpha_k))))
    weights = np.zeros(pad_len)
    weights[:len(alpha_k)] = alpha_k
    weights /= np.sum(weights)

    # 构造 U_list：用 e^{2ikH}
    U_list = [expm(2j * k * H) for k in k_vals]
    U_list += [np.eye(H.shape[0])] * (pad_len - len(U_list))

    # 激活逻辑
    def active_U_indices(b, dim):
        indices = []
        for i in range(dim):
            bin_i = format(i, f"0{len(b)}b")
            if all(b[j] == 1 if bit == '1' else True for j, bit in enumerate(bin_i)):
                indices.append(i)
        return indices

    def compute_weighted_sum(U_list, weights):
        dim = len(U_list)
        dim1 = U_list[0].shape[0]
        A = np.zeros((dim1, dim1), dtype=complex)
        for i, b in enumerate(itertools.product([0, 1], repeat=int(np.log2(dim)))):
            indices = active_U_indices(b, dim)
            U_total = np.eye(dim1, dtype=complex)
            for idx in indices:
                U_total = U_list[idx] @ U_total
            A += weights[i] * U_total
        return A

    return compute_weighted_sum(U_list, weights)

def nlcu_circuit(H, m):
    n = 2
    dim = 2 ** n

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
    U_list = [expm(2j * k * H) for k in k_vals]
    U_list += [np.eye(dim)] * (pad_len - len(U_list))

    @qml.qnode(dev)
    def circuit():
        qml.MottonenStatePreparation(weights, wires=ancilla_wires)
        for idx, U in enumerate(U_list):
            bin_str = format(idx, f"0{n_anc}b")
            ctrl_wires = [ancilla_wires[i] for i, b in enumerate(bin_str) if b == '1']
            qml.ControlledQubitUnitary(U, control_wires=ctrl_wires, wires=system_wires)
        qml.adjoint(qml.MottonenStatePreparation)(weights, wires=ancilla_wires)
        return qml.state()



    U_circuit = qml.matrix(circuit, wire_order=list(range(n_anc)) + list(range(n_anc, n_anc + n)))()
    A_block = U_circuit[:dim, :dim]
    # system_state = A_block @ init_state

    # full_state = circuit()
    # reshaped = full_state.reshape((2 ** n_anc, 2 ** n))
    # system_state = reshaped[0, :]
    #
    # success_prob = np.sum(np.abs(system_state) ** 2)
    # normalized_system = system_state / np.linalg.norm(system_state)
    # fidelity = np.abs(np.vdot(normalized_system, ground_state)) ** 2
    return A_block

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


def alpha_k_vals(m, threshold=1e-4):
    k_vals = np.arange(-m, m + 1)
    alpha_k = 2 ** (-2 * m) * np.array([math.comb(2 * m, m + k) for k in k_vals])
    keep = alpha_k > threshold
    return k_vals[keep], alpha_k[keep] / np.sum(alpha_k[keep])

def method1_direct_sum(x, m):
    k_vals, alpha_k = alpha_k_vals(m)
    return np.sum([a * np.exp(2j * k * x) for a, k in zip(alpha_k, k_vals)])

def method2_cos2m_truth(x, m):
    return np.cos(x) ** (2 * m)

def method3_activated_product(x, m):
    k_vals, alpha_k = alpha_k_vals(m)
    pad_len = 2 ** int(np.ceil(np.log2(len(alpha_k))))
    weights = np.zeros(pad_len)
    weights[:len(alpha_k)] = alpha_k
    weights /= np.sum(weights)
    n = int(np.log2(pad_len))
    k_list = list(k_vals) + [0] * (pad_len - len(k_vals))

    def active_indices(b, dim):
        indices = []
        for i in range(dim):
            bin_i = format(i, f"0{len(b)}b")
            if all(b[j] == 1 if bit == '1' else True for j, bit in enumerate(bin_i)):
                indices.append(i)
        return indices

    dim = 2 ** n
    A = 0.0 + 0.0j
    for i, b in enumerate(itertools.product([0, 1], repeat=n)):
        indices = active_indices(b, dim)
        #print('[indices] is ', indices)
        prod = 1.0 + 0.0j
        for idx in indices:
            prod *= np.exp(2j * k_list[idx] * x)
        A += weights[i] * prod
    return A

# m = 6
# x_vals = np.linspace(-5, 5, 400)
#
# with open("method1_direct_sum.txt", "w") as f1, \
#      open("method2_cos2m_truth.txt", "w") as f2, \
#      open("method3_activated_product.txt", "w") as f3:
#
#     for x in x_vals:
#         val1 = method1_direct_sum(x, m)
#         val2 = method2_cos2m_truth(x, m)
#         val3 = method3_activated_product(x, m)
#
#         f1.write(f"{x:.8f}, {val1.real:.8f}, {val1.imag:.8f}\n")
#         f2.write(f"{x:.8f}, {val2:.8f}, 0.00000000\n")  # real only
#         f3.write(f"{x:.8f}, {val3.real:.8f}, {val3.imag:.8f}\n")
# def get_method3_spectrum(m):
#     # Step 1: alpha_k and k_vals
#     k_vals, alpha_k = alpha_k_vals(m)
#     pad_len = len(alpha_k)
#     weights = alpha_k  # no padding
#     n = int(np.ceil(np.log2(pad_len)))
#     k_list = list(k_vals)
#
#     # Step 2: Iterate over all ancilla bitstrings b
#     dim = 2 ** n
#     freq_dict = defaultdict(complex)
#
#     def active_indices(b, dim):
#         indices = []
#         for i in range(dim):
#             bin_i = format(i, f"0{len(b)}b")
#             if all(b[j] == 1 if bit == '1' else True for j, bit in enumerate(bin_i)):
#                 indices.append(i)
#         return indices
#
#     for i, b in enumerate(itertools.product([0, 1], repeat=n)):
#         if i >= len(weights):  # skip padded part
#             continue
#         indices = active_indices(b, dim)
#         total_k = sum(k_list[idx] for idx in indices if idx < len(k_list))
#         freq_dict[total_k] += weights[i]
#
#     return freq_dict
#
# # ====== Main Execution ======
#
# # Sort and plot
# # sorted_freqs = sorted(freq_dict.items())
# # ks, amps = zip(*sorted_freqs)
# #
# # plt.figure(figsize=(8, 4))
# # plt.bar(ks, np.real(amps), width=0.5, label="Re[amplitude]")
# # plt.bar(ks, np.imag(amps), width=0.5, bottom=np.real(amps), color='orange', label="Im[amplitude]")
# # plt.xlabel("Effective total frequency k")
# # plt.ylabel("Amplitude (weight)")
# # plt.title(f"Spectrum of method3_activated_product (m={m})")
# # plt.grid(True)
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
#
def alpha_k_vals(m, threshold=1e-4):
    k_vals = np.arange(-m, m + 1)
    alpha_k = 2 ** (-2 * m) * np.array([math.comb(2 * m, m + k) for k in k_vals])
    keep = alpha_k > threshold
    return k_vals[keep], alpha_k[keep] / np.sum(alpha_k[keep])

def get_cos2m_spectrum(m):
    k_vals, alpha_k = alpha_k_vals(m)
    freq_dict = defaultdict(complex)
    for k, a in zip(k_vals, alpha_k):
        freq_dict[2 * k] = a  # 注意是 2k，因为 cos^{2m}(x) 展开后是 e^{2ikx}
    return freq_dict

def get_method3_spectrum(m):
    # 获取 k 值和 alpha_k
    k_vals, alpha_k = alpha_k_vals(m)
    #print('[alpha_k] is', alpha_k)
    #print('[k_vals] is', k_vals)
    num_terms = len(alpha_k)
    weights = alpha_k / np.sum(alpha_k)
    n = int(np.ceil(np.log2(num_terms)))
    dim = 2 ** n

    # 填充 k_list 和 weight
    k_list = list(k_vals) + [0] * (dim - num_terms)
    weights = np.concatenate([weights, np.zeros(dim - num_terms)])

    # 枚举所有比特串 b
    binary_strings = list(itertools.product([0, 1], repeat=n))

    def active_indices(b, dim):
        indices = []
        for i in range(dim):
            bin_i = format(i, f"0{len(b)}b")
            if all(b[j] == 1 if bit == '1' else True for j, bit in enumerate(bin_i)):
                indices.append(i)
        return indices

    # 构造频谱
    freq_dict = defaultdict(float)
    for b, w in zip(binary_strings, weights):
        indices = active_indices(b, dim)
        #print('[indices] is ', indices)
        total_k = sum(k_list[idx] for idx in indices)
        freq_dict[total_k] += w

    return freq_dict

def plot_spectrum_comparison(freq_dict_cos2m, freq_dict_method3, m):
    all_k = sorted(set(freq_dict_cos2m) | set(freq_dict_method3))
    re1 = [freq_dict_cos2m.get(k, 0).real for k in all_k]
    re2 = [freq_dict_method3.get(k, 0) for k in all_k]

    plt.figure(figsize=(10, 5))
    bar1 = plt.bar([k - 0.2 for k in all_k], re1, width=0.4, label='cos^{2m}(x)', color='gray')
    bar2 = plt.bar([k + 0.2 for k in all_k], re2, width=0.4, label='nlcu', color='royalblue')


    plt.xlabel("Effective total frequency k")
    plt.ylabel("Amplitude (weight)")
    plt.title(f"Spectrum comparison (m = {m})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==== 主程序 ====
m = 12
freq_cos2m = get_cos2m_spectrum(m)
freq_method3 = get_method3_spectrum(m)
plot_spectrum_comparison(freq_cos2m, freq_method3, m)

