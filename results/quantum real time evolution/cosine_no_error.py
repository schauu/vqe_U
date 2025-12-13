#!/usr/bin/env python3

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from pennylane_qiskit import load_pauli_op
from qiskit.quantum_info import Operator, SparsePauliOp




# ---------------------------------------------------------
# Build Hamiltonian (same as your get_hamiltonian)
# ---------------------------------------------------------
def get_hamiltonian_pl(nq, J):
    coeffs = []
    ops = []
    for i in range(nq - 1):
        coeffs.append(J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    coeffs.append(J)
    ops.append(qml.PauliZ(nq - 1) @ qml.PauliZ(0))
    for i in range(nq):
        coeffs.append(J)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)

# def build_H_cosine(H_system):
#     """
#     Given a PennyLane Hamiltonian H_system acting on wires [0..n-1],
#     return Y(0) ⊗ H_system acting on [0..n].
#     """
#
#     new_coeffs = []
#     new_ops = []
#
#     # 构造 wire_map，例如系统有 4 个 qubit: {0:1, 1:2, 2:3, 3:4}
#     max_wire = max([int(w) for op in H_system.ops for w in op.wires])
#     wire_map = {w: w + 1 for w in range(max_wire + 1)}
#
#     for c, op in zip(H_system.coeffs, H_system.ops):
#         # shift wires by +1
#         shifted_op = op.map_wires(wire_map)
#
#         # attach Y on new ancilla 0
#         full_op = qml.PauliY(0) @ shifted_op
#
#         new_coeffs.append(c)
#         new_ops.append(full_op)
#
#     return qml.Hamiltonian(new_coeffs, new_ops)
#
# # ---------------------------------------------------------
# # Construct evolution U = exp(-i(Y ⊗ H)t)
# # ---------------------------------------------------------
# def apply_cosine_evolution(H, time):
#     H_cos = build_H_cosine(H)#H_cos = H#
#     qml.TrotterProduct(H_cos, -time, n=10, order=2)  # apply exp(-i H_cos t)
#
#
# # ---------------------------------------------------------
# # Device (noiseless)
# # ---------------------------------------------------------
# def select_dev(error_rate, nqubits):
#     # If need noise, we can add qml.DepolarizingChannel etc.
#     if error_rate == 0:
#         return qml.device("default.qubit", wires=nqubits + 1)
#     else:
#         # simple noise model
#         return qml.device("default.mixed", wires=nqubits + 1)
#
#
# # ---------------------------------------------------------
# # Cosine filtering loop
# # ---------------------------------------------------------
# def cosine_filtering_pl(H, time, nqubits, error_rate,
#                         step, threshold):
#
#     dev = select_dev(error_rate, nqubits)
#
#     # Pennylane QNode to apply filtering
#     @qml.qnode(dev, interface="numpy")
#     def cosine_step(state):
#         qml.StatePrep(state, wires=range(1, nqubits + 1))
#
#         apply_cosine_evolution(H, time)
#
#         return qml.state()
#
#     # For computing ⟨H⟩
#     @qml.qnode(dev, interface="numpy")
#     def energy_eval(state):
#         qml.StatePrep(state, wires=range(nqubits))
#         return qml.expval(H)
#
#     # initial state
#     state = 1/np.sqrt(2**nqubits) * np.ones(2**nqubits)
#     #state[0] = 1.0
#
#     expectation_list = []
#     probability_list = []
#
#     for i in range(step):
#
#         # 1. evolve
#         full_state = cosine_step(state)
#
#         # 2. post-select ancilla = 0
#         # full_state shape = (2^(nqubits+1), )
#         post_state = full_state[:2**nqubits]
#         prob = np.linalg.norm(post_state)**2
#
#         if prob < 1e-12:
#             raise ValueError("Post-selection probability=0")
#
#         post_state = post_state / np.sqrt(prob)
#
#         # 3. compute energy
#         expectation = energy_eval(post_state)
#
#         if expectation <= threshold:
#             break
#
#         # 4. accumulate probability
#         if i == 0:
#             probability_list.append(prob)
#         else:
#             probability_list.append(probability_list[-1] * prob)
#
#         expectation_list.append(expectation)
#         state = post_state
#
#     return np.array(expectation_list), np.array(probability_list)
#
# def plot_cosine_filter(expectations, probabilities):
#     """
#     expectations: numpy array of <H> per iteration
#     probabilities: cumulative success probability per iteration
#     """
#
#     steps = np.arange(1, len(expectations) + 1)
#
#     fig, ax1 = plt.subplots(figsize=(8, 5))
#
#     # -------------------------
#     # 绘制能量下降曲线（左轴）
#     # -------------------------
#     color1 = "tab:blue"
#     ax1.set_xlabel("Iteration")
#     ax1.set_ylabel("Energy  <H>", color=color1)
#     ax1.plot(steps, expectations, "-o", color=color1, label="Energy")
#     ax1.tick_params(axis="y", labelcolor=color1)
#     #ax1.grid(True, linestyle="--", alpha=0.3)
#
#     # -------------------------
#     # 绘制成功概率曲线（右轴）
#     # -------------------------
#     ax2 = ax1.twinx()
#     color2 = "tab:red"
#     ax2.set_ylabel("Success Probability", color=color2)
#     ax2.plot(steps, probabilities, "-s", color=color2, label="Probability")
#     ax2.tick_params(axis="y", labelcolor=color2)
#
#     # -------------------------
#     # 标题
#     # -------------------------
#     plt.title("Cosine Filtering: Energy Decrease & Success Probability")
#
#     plt.tight_layout()
#     plt.show()
#
# H = get_hamiltonian_pl(4, 1)
# H_array = qml.matrix(H)
# time = np.pi/2
# nqubits = 4
# error_rate = 0
# step = 15
# threshold = 1e-2
# # eval, _ = np.linalg.eigh(H_array)
# # emin = eval[0]
# # emax = eval[-1]
# # H_array = (H_array - emin * np.eye(2**nqubits)) / (emax - emin)  ## scale H
# # H_cosine = SparsePauliOp.from_operator(Operator(H_array))
# # H_cosine = SparsePauliOp(["Y"]) ^ H_cosine
# # #H_cosine = H_cosine^ SparsePauliOp(["Y"])
# # H_cosine = load_pauli_op(H_cosine)
# expectation_list, probability_list = cosine_filtering_pl(
#         H, time, nqubits, error_rate, step, threshold
#     )
# plot_cosine_filter(expectation_list, probability_list)

# # ====== 计算 trotter_product 的矩阵 ======
# def trotter_matrix(H_cos, t, n_steps):
#     op = qml.TrotterProduct(H_cos, -t, n=n_steps, order=2)
#     return qml.matrix(op)
#
#
#
# # ====== 精确矩阵（scipy） ======
# def exact_matrix(H_cos, t):
#     H_mat = qml.matrix(H_cos)
#     return sc.linalg.expm(-1j * H_mat * t)
#
#
#
# # ====== 误差曲线 ======
# def plot_trotter_error(H_cos):
#     ts = np.linspace(0, np.pi/2, 40)
#
#     exact_vals = []
#     trotter_vals = []
#     errors = []
#
#     for t in ts:
#         U_exact = exact_matrix(H_cos, t)
#         U_trot = trotter_matrix(H_cos, t)
#
#         # 这里选 Frobenius norm
#         err = np.linalg.norm(U_exact - U_trot)
#
#         exact_vals.append(np.linalg.norm(U_exact))
#         trotter_vals.append(np.linalg.norm(U_trot))
#         errors.append(err)
#
#     plt.figure(figsize=(8, 5))
#     plt.plot(ts, errors, label="||U_exact - U_trot||", color="red")
#     plt.xlabel("t")
#     plt.ylabel("Error")
#     plt.title("TrotterProduct Approximation Error")
#     plt.grid(linestyle="--", alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_error_vs_trotter_steps(H_cos, t):
#
#     trotter_steps = np.arange(1, 30)   # n = 1 ... 29
#     errors = []
#
#     U_exact = exact_matrix(H_cos, t)
#
#     for n in trotter_steps:
#         U_trot = trotter_matrix(H_cos, t, n)
#         err = np.linalg.norm(U_exact - U_trot)
#         errors.append(err)
#
#     plt.figure(figsize=(8,5))
#     plt.plot(trotter_steps, errors, '-o')
#     plt.xlabel("Trotter Steps n")
#     plt.ylabel("Error  ||U_exact - U_trot(n)||")
#     plt.title("Trotter Error vs Number of Trotter Steps")
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#
#
#
# # =================== 运行 ===========================
# nqubits = 4
# H_sys = get_hamiltonian_pl(nqubits, 1.0)
# H_cos = build_H_cosine(H_sys)
#
# t = np.pi/2
# plot_error_vs_trotter_steps(H_cos, t)
nqubits = 4
dev = qml.device("default.qubit", wires=nqubits+1)
state = 1/np.sqrt(2**4) * np.ones(2**4)
H = get_hamiltonian_pl(4,1)
H_array = qml.matrix(H)
eval, _ = np.linalg.eigh(H_array)
emin = eval[0]
emax = eval[-1]
H_array = (H_array - emin * np.eye(2**nqubits)) / (emax - emin)
H = SparsePauliOp.from_operator(Operator(H_array))
H_cos = load_pauli_op(SparsePauliOp(["Y"])^H)

def trotter_matrix(H_cos, t, n_steps):
    op = qml.TrotterProduct(H_cos, -t, n=n_steps, order=2)
    return qml.matrix(op)

# ====== 精确矩阵（scipy） ======
def exact_matrix(H_cos, t):
    H_mat = qml.matrix(H_cos)
    return sc.linalg.expm(-1j * H_mat * t)

# ====== 误差曲线 ======
def plot_trotter_error(H_cos):
    ts = np.linspace(0, np.pi/2, 40)

    exact_vals = []
    trotter_vals = []
    errors = []

    for t in ts:
        U_exact = exact_matrix(H_cos, t)
        U_trot = trotter_matrix(H_cos, t)

        # 这里选 Frobenius norm
        err = np.linalg.norm(U_exact - U_trot)

        exact_vals.append(np.linalg.norm(U_exact))
        trotter_vals.append(np.linalg.norm(U_trot))
        errors.append(err)

    plt.figure(figsize=(8, 5))
    plt.plot(ts, errors, label="||U_exact - U_trot||", color="red")
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.title("TrotterProduct Approximation Error")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_vs_trotter_steps(H_cos, t):

    trotter_steps = np.arange(1, 30)   # n = 1 ... 29
    errors = []

    U_exact = exact_matrix(H_cos, t)

    for n in trotter_steps:
        U_trot = trotter_matrix(H_cos, t, n)
        err = np.linalg.norm(U_exact - U_trot)
        errors.append(err)

    plt.figure(figsize=(8,5))
    plt.plot(trotter_steps, errors, '-o')
    plt.xlabel("Trotter Steps n")
    plt.ylabel("Error  ||U_exact - U_trot(n)||")
    plt.title("Trotter Error vs Number of Trotter Steps")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# t = np.pi/2
# plot_error_vs_trotter_steps(H_cos, t)
# ---------------------------------------------------------
# Device (noiseless)
# ---------------------------------------------------------
def select_dev(error_rate, nqubits):
    # If need noise, we can add qml.DepolarizingChannel etc.
    if error_rate == 0:
        return qml.device("default.qubit", wires=nqubits + 1)
    else:
        # simple noise model
        return qml.device("default.mixed", wires=nqubits + 1)


# ---------------------------------------------------------
# Cosine filtering loop
# ---------------------------------------------------------
def cosine_filtering_pl(H, H_cos, time, nqubits, error_rate,
                        step, threshold):

    dev = select_dev(error_rate, nqubits)

    # Pennylane QNode to apply filtering
    @qml.qnode(dev, interface="numpy")
    def cosine_step(state):
        qml.StatePrep(state, wires=range(1, nqubits + 1))

        qml.TrotterProduct(H_cos, -time, n=10, order=2)


        return qml.state()

    # For computing ⟨H⟩
    @qml.qnode(dev, interface="numpy")
    def energy_eval(state):
        qml.StatePrep(state, wires=range(nqubits))
        return qml.expval(H)

    # initial state
    state = 1/np.sqrt(2**nqubits) * np.ones(2**nqubits)
    #state[0] = 1.0

    expectation_list = []
    probability_list = []

    for i in range(step):

        # 1. evolve
        full_state = cosine_step(state)

        # 2. post-select ancilla = 0
        # full_state shape = (2^(nqubits+1), )
        reshaped = full_state.reshape((2, 2**nqubits))
        post_state = reshaped[0, :]
        #post_state = full_state[:2**nqubits]
        prob = np.linalg.norm(post_state)**2

        if prob < 1e-12:
            raise ValueError("Post-selection probability=0")

        post_state = post_state / np.sqrt(prob)

        # 3. compute energy
        expectation = energy_eval(post_state)

        if expectation <= threshold:
            break

        # 4. accumulate probability
        if i == 0:
            probability_list.append(prob)
        else:
            probability_list.append(probability_list[-1] * prob)

        expectation_list.append(expectation)
        state = post_state

    return np.array(expectation_list), np.array(probability_list)

def plot_cosine_filter(expectations, probabilities):
    """
    expectations: numpy array of <H> per iteration
    probabilities: cumulative success probability per iteration
    """

    steps = np.arange(1, len(expectations) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # -------------------------
    # 绘制能量下降曲线（左轴）
    # -------------------------
    color1 = "tab:blue"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Energy  <H>", color=color1)
    ax1.plot(steps, expectations, "-o", color=color1, label="Energy")
    ax1.tick_params(axis="y", labelcolor=color1)
    #ax1.grid(True, linestyle="--", alpha=0.3)

    # -------------------------
    # 绘制成功概率曲线（右轴）
    # -------------------------
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Success Probability", color=color2)
    ax2.plot(steps, probabilities, "-s", color=color2, label="Probability")
    ax2.tick_params(axis="y", labelcolor=color2)

    # -------------------------
    # 标题
    # -------------------------
    plt.title("Cosine Filtering: Energy Decrease & Success Probability")

    plt.tight_layout()
    plt.show()

H = SparsePauliOp.from_operator(Operator(H_array))
H_op = load_pauli_op(H)
#H_cos = load_pauli_op(SparsePauliOp(["Y"])^H)
H_cos = load_pauli_op(H^SparsePauliOp(["Y"]))
time = np.pi/2
nqubits = 4
error_rate = 0
step = 30
threshold = 1e-4

expectation_list, probability_list = cosine_filtering_pl(
        H_op, H_cos, time, nqubits, error_rate, step, threshold
    )
print('success probability at 0 step', probability_list[0])
plot_cosine_filter(expectation_list, probability_list)