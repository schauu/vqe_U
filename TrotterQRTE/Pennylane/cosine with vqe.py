#!/usr/bin/env python3

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from pennylane_qiskit import load_pauli_op
from qiskit.quantum_info import Operator, SparsePauliOp

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

def ansatz_vqe(params, nqubits, depth):
    """PennyLane version of the given Qiskit ansatz_vqe."""
    t = 0

    # Apply H on system qubits
    for i in range(nqubits):
        qml.Hadamard(wires=i)

    # Layers
    for _ in range(depth):

        # --- RZZ ring ---
        for i in range(nqubits):
            j = i + 1 if i < (nqubits - 1) else 0
            qml.IsingZZ(params[t], wires=[i, j])
            t += 1

        # (barrier ignored)

        # --- RX ring ---
        for i in range(nqubits):
            j = i + 1 if i < (nqubits - 1) else 0
            qml.RX(params[t], wires=i)
            t += 1

        # --- RY on ancilla ---
        qml.RY(params[t], wires=nqubits)   # ancilla is last wire
        t += 1

def select_dev(error_rate, nqubits):
    """代替原来的 select_sim，返回一个 PennyLane device。"""
    if error_rate == 0:
        return qml.device("default.qubit", wires=nqubits + 1)
    else:
        # 简单噪声模型，用 mixed device；需要的话可以在电路里插入噪声通道
        return qml.device("default.mixed", wires=nqubits + 1)

def apply_trotter_evolution(H_cos, time, time_step):
    n_steps = max(1, int(round(time / time_step)))
    qml.TrotterProduct(H_cos, -time, n=n_steps, order=2)


class VQECallback:
    def __init__(self, dev, nqubits, reference, H_matrix, threshold,
                 H_cosine, time, time_step, vqe_depth, init_state):
        """
        dev: PennyLane device
        nqubits: 系统比特数（不含 ancilla）
        reference: 参考基态能量（你的 reference）
        H_matrix: 系统哈密顿量的矩阵 (2^n x 2^n)
        threshold: 约束中的 threshold
        H_cosine: 用 build_H_cosine 构造的 Hamiltonian
        time, time_step: Trotter 演化参数
        vqe_depth: ansatz 深度
        init_state: 当前 step 的系统态向量 (2^n,) —— 即外层 loop 里的 statevector
        """
        self.dev = dev
        self.nqubits = nqubits
        self.reference = reference
        self.H_matrix = H_matrix
        self.threshold = threshold
        self.H_cosine = H_cosine
        self.time = time
        self.time_step = time_step
        self.vqe_depth = vqe_depth
        self.init_state = init_state  # 系统态，只作用在 0..nqubits-1

        @qml.qnode(self.dev)
        def vqe_circuit(params):
            # 初始化系统态到前 nqubits 上，ancilla 保持 |0>
            qml.StatePrep(self.init_state, wires=range(self.nqubits))
            # 对 H_cosine 做 Trotter 演化（作用在 0..nqubits 上）
            apply_trotter_evolution(self.H_cosine, self.time, self.time_step)
            # 叠加 ansatz（注意 ansatz_vqe 的 wires 是 0..nqubits）
            ansatz_vqe(params, self.nqubits, self.vqe_depth)
            return qml.state()

        self.qnode = vqe_circuit

    # 原来的 run_sim：返回 statevector
    def run_sim(self, params):
        state = self.qnode(params)
        # 转成 numpy 数组方便后续 np.linalg 操作
        return np.array(state, dtype=complex)

    # 原来的 constraint_func：保留
    def constraint_func(self, params):
        statevector = self.run_sim(params)
        sys_state = statevector[:2 ** self.nqubits]
        sys_state /= np.linalg.norm(sys_state)
        expectation = sys_state.conj().T @ (self.H_matrix @ sys_state)
        expectation = expectation.real
        return -(expectation - self.reference - self.threshold)
        # 或者用你原来注释的那一版：
        # return -np.abs(expectation - self.reference - self.threshold)

def cosine_filtering_vqe(H, time, nqubits, order, time_step,
                         error_rate, step, threshold):
    """
    PennyLane 版本的 cosine_filtering_vqe。
    H: qml.Hamiltonian（系统哈密顿量，作用在 0..nqubits-1）
    其它参数和你原来的函数一致。
    """
    dev = select_dev(error_rate, nqubits)

    expectation_before_list = []
    probability_before_list = []
    expectation_after_list = []
    probability_after_list = []

    # 初始态 |000...0>
    statevector = np.zeros(2 ** nqubits, dtype=complex)
    statevector[0] = 1.0

    # 构造 H_cosine 和矩阵形式 H_matrix
    H_cosine = build_H_cosine(H, nqubits)
    H_matrix = np.array(qml.matrix(H), dtype=complex)

    vqe_depth = 3
    vqe_nparams = vqe_depth * ((nqubits + 1) * 2 - 1)  # 和你原来的公式一致

    # 1) cosine filter 用的 QNode（对应原来的 qc + construct_trotter_circuit + simulator.run）
    @qml.qnode(dev)
    def cosine_step_qnode(state_in):
        # state_in 是 (2^n,) 的系统态，作用在 0..nqubits-1 上
        qml.StatePrep(state_in, wires=range(nqubits))
        # ancilla = |0>
        apply_trotter_evolution(H_cosine, time, time_step, order)
        return qml.state()

    for i in range(step):
        print(f"Running {i}", flush=True)

        # ------------------------------
        # cosine filter 部分（对应你原来的 qc + qc.save_statevector）
        # ------------------------------
        full_state = cosine_step_qnode(statevector)
        full_state = np.array(full_state, dtype=complex)

        # 后选 ancilla = 0 子空间
        postselect_state = full_state[:2 ** nqubits]
        probability = np.linalg.norm(postselect_state)
        postselect_state /= probability

        # 这里 expectation 是「参考能量」（对应原来 statevector->postselect_state）
        expectation = postselect_state.conj().T @ (H_matrix @ postselect_state)

        if i == 0:
            # 第一轮不累乘
            probability_total_before = probability
            expectation_ref = expectation
        else:
            probability_total_before = probability * probability_before_list[-1]
            # 参考能量仍然用上一轮 VQE 之后的 expectation_after
            expectation_ref = expectation_after_list[-1]

        probability_before_list.append(probability_total_before)
        expectation_before_list.append(expectation_ref)

        # 更新当前系统态，作为下一步的输入
        statevector = postselect_state

        # ------------------------------
        # VQE 部分（保持 SLSQP + constraint_func）
        # ------------------------------
        result_intermediate = []

        # VQECallback 里会再做一遍「cosine + ansatz」
        vqe_callback = VQECallback(
            dev=dev,
            nqubits=nqubits,
            reference=expectation_ref.real,
            H_matrix=H_matrix,
            threshold=-1e-2,        # 和你原来的 -1e-2 一致
            H_cosine=H_cosine,
            time=time,
            time_step=time_step,
            vqe_depth=vqe_depth,
            init_state=statevector,  # 当前 step 的系统态
        )

        def callback(xk):
            # 保留你原来 callback 的逻辑：记录每轮的概率
            sv = vqe_callback.run_sim(xk)
            prob = np.linalg.norm(sv[:2 ** nqubits])
            result_intermediate.append(prob)

        np.random.seed(42)
        init_params = np.random.random(vqe_nparams)

        vqe_result = sc.minimize(
            vqe_callback.cost_func,
            init_params,
            method="SLSQP",
            constraints={"type": "ineq", "fun": vqe_callback.constraint_func},
            options={"maxiter": 1000},
            callback=callback,
        )
        print(vqe_result)

        # 用优化结果跑一次，得到后选态
        post_vqe_state_full = vqe_callback.run_sim(vqe_result.x)
        post_vqe_sys = post_vqe_state_full[:2 ** nqubits]
        probability_vqe = np.linalg.norm(post_vqe_sys)
        print(f"Probability at this step is {probability_vqe}")
        post_vqe_sys /= probability_vqe

        expectation_after = post_vqe_sys.conj().T @ (H_matrix @ post_vqe_sys)

        # 终止条件
        if expectation_after <= threshold:
            break

        if i == 0:
            probability_total_after = probability_vqe
        else:
            probability_total_after = probability_vqe * probability_after_list[-1]

        print(f"Probability consider previous step is {probability_total_after}")

        probability_after_list.append(probability_total_after)
        expectation_after_list.append(expectation_after)
        statevector = post_vqe_sys

    expectation_before_list = np.array(expectation_before_list)
    probability_before_list = np.array(probability_before_list)
    expectation_after_list = np.array(expectation_after_list)
    probability_after_list = np.array(probability_after_list)

    # 保留你原来的 sanity check
    assert np.max(np.abs(np.imag(expectation_before_list))) < 1e-6
    assert np.max(np.abs(np.imag(expectation_after_list))) < 1e-6

    return (
        expectation_before_list.real,
        probability_before_list,
        expectation_after_list.real,
        probability_after_list,
        # result_intermediate 你如果需要也可以 return 出去
    )


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
