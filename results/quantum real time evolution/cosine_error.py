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
        return qml.device("default.mixed", wires=nqubits + 1)
    else:
        # simple noise model
        return qml.device("default.mixed", wires=nqubits + 1)


# ---------------------------------------------------------
# Cosine filtering loop
# ---------------------------------------------------------
def cosine_filtering_pl(H, H_cos, time, nqubits, error_rate,
                        step, threshold):

    dev = select_dev(error_rate, nqubits)

    # evolution + noise
    @qml.qnode(dev, interface="numpy")
    def cosine_step(rho_sys):
        """rho_sys 是 system 的 4-qubit density matrix (16x16)"""

        dim_sys = 2 ** nqubits

        # 构造 ancilla=|0><0| ⊗ rho_sys 作为 5-qubit 初始态
        proj0 = np.array([[1.0, 0.0],
                          [0.0, 0.0]])
        rho_total = np.kron(proj0, rho_sys)  # 维度 (32, 32)

        # 在所有 5 个比特上放这个密度矩阵
        qml.QubitDensityMatrix(rho_total, wires=range(nqubits + 1))

        # 时间演化
        qml.TrotterProduct(H_cos, -time, n=10, order=2)

        # 噪声：对所有 5 个比特加 depolarizing
        if error_rate > 0:
            for w in range(nqubits + 1):
                qml.DepolarizingChannel(error_rate, wires=w)

        # 返回完整的 5-qubit 密度矩阵
        return qml.state()

    # energy measurement：只看系统的 n 个比特
    @qml.qnode(dev, interface="numpy")
    def energy_eval(rho_sys):
        qml.QubitDensityMatrix(rho_sys, wires=range(nqubits))
        return qml.expval(H)

    # ------- 初始纯态 → system density matrix -------
    #psi = 1 / np.sqrt(2 ** nqubits) * np.ones(2 ** nqubits)
    psi = np.zeros(2 ** nqubits)
    psi[0] = 1
    rho = np.outer(psi, psi.conj())         # (16,16) system density matrix

    expectation_list = []
    probability_list = []

    for i in range(step):

        # 1. 演化到 5-qubit 密度矩阵 ρ_full (32×32)
        rho_full = cosine_step(rho)

        # 2. 对 ancilla=0 做 post-selection
        dim_sys = 2 ** nqubits
        # ρ_full 的块结构：
        # [ ρ00  ρ01 ]
        # [ ρ10  ρ11 ]
        # ancilla=0 对应左上角 block
        rho00 = rho_full[0:dim_sys, 0:dim_sys]

        prob = np.trace(rho00).real

        if prob < 1e-12:
            raise ValueError("Post-selection probability = 0")

        rho_post = rho00 / prob

        # 3. 计算能量
        expectation = energy_eval(rho_post)

        # 如果已经达到阈值就停
        if expectation <= threshold:
            break

        # 4. 累积成功概率（乘法）
        if i == 0:
            probability_list.append(prob)
        else:
            probability_list.append(probability_list[-1] * prob)

        expectation_list.append(expectation)

        # 下一轮用 post-selection 后的系统密度矩阵
        rho = rho_post

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
#error_rate = 1e-1
step = 30
threshold = 1e-4
#
# expectation_list, probability_list = cosine_filtering_pl(
#         H_op, H_cos, time, nqubits, error_rate, step, threshold
#     )
# print('success probability at 0 step', probability_list[0])
# plot_cosine_filter(expectation_list, probability_list)

def run_for_noise_levels(noise_list, H_op, H_cos, time, nqubits, steps, threshold):
    results = {}

    for err in noise_list:
        print(f"Running error_rate = {err} ...")
        exp_list, prob_list = cosine_filtering_pl(
            H_op, H_cos, time, nqubits, err, steps, threshold
        )
        results[err] = (exp_list, prob_list)

    return results

def plot_all_energy(results):
    plt.figure(figsize=(8, 5))

    for err, (exp_list, _) in results.items():
        steps = np.arange(1, len(exp_list) + 1)
        plt.plot(steps, exp_list, '-o', label=f"error = {err}")

    plt.xlabel("Iteration")
    plt.ylabel("Energy <H>")
    plt.title("Energy Convergence Under Different Noise Levels")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("energy_error.png")
    plt.show()

def plot_all_probability(results):
    plt.figure(figsize=(8, 5))

    for err, (_, prob_list) in results.items():
        steps = np.arange(1, len(prob_list) + 1)
        plt.plot(steps, prob_list, '-s', label=f"error = {err}")

    plt.xlabel("Iteration")
    plt.ylabel("Success Probability (cumulative)")
    plt.title("Post-selection Success Probability Under Noise")
    plt.ylim(0, 0.2)
    plt.legend()
    #plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("probability_error.png")
    plt.show()

noise_levels = [0, 1e-3, 1e-2, 1e-1]

results = run_for_noise_levels(
    noise_levels,
    H_op, H_cos,
    time=np.pi/2,
    nqubits=4,
    steps=30,
    threshold=1e-4
)

plot_all_energy(results)
plot_all_probability(results)
