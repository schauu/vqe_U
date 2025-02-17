#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import Estimator
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter, MatrixExponential
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE


def get_hamiltonian(nq, J):
    op_list = [("ZZ", [i, i + 1], J) for i in range(nq - 1)]
    op_list.append(("ZZ", [nq - 1, 0], J))
    op_list += [("X", [i], J) for i in range(nq)]
    return SparsePauliOp.from_sparse_list(op_list, num_qubits=nq).simplify()


def ansatz_vqe(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits + 1)
    t = 0
    for i in range(nqubits):
        circuit.h(i)
    for idepth in range(depth):
        for i in range(nqubits):
            if i < (nqubits - 1):
                j = i + 1
            else:
                j = 0
            circuit.rzz(params[t], i, j)
            t += 1
        circuit.barrier()
        for i in range(nqubits):
            if i < (nqubits - 1):
                j = i + 1
            else:
                j = 0
            circuit.rx(params[t], i)
            t += 1
        circuit.ry(params[t], nqubits)
        t += 1
    return circuit


def ansatz_hamiltonian(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits)
    t = 0
    for i in range(nqubits):
        circuit.h(i)
    for _ in range(depth):
        for i in range(nqubits):
            if i < (nqubits - 1):
                j = i + 1
            else:
                j = 0
            circuit.rzz(params[t], i, j)
            t += 1
        circuit.barrier()
        for i in range(nqubits):
            if i < (nqubits - 1):
                j = i + 1
            else:
                j = 0
            circuit.rx(params[t], i)
            t += 1
    return circuit


def ansatz_hea(nqubits, depth, params):
    circuit = QuantumCircuit(nqubits)
    t = 0
    for _ in range(depth):
        for i in range(nqubits):
            circuit.ry(params[t], i)
            t += 1
        circuit.barrier()
        for i in range(nqubits - 1):
            circuit.cx(i, i + 1)
    return circuit


def construct_trotter_circuit(H, time, nqubits, order, time_step):
    if order == 0:
        formular = MatrixExponential()
    elif order == 1:
        formular = LieTrotter(reps=time_step)
    else:
        formular = SuzukiTrotter(order=order, reps=time_step)
    trotter_step_first_order = PauliEvolutionGate(H, time, synthesis=formular)
    circuit = QuantumCircuit(nqubits + 1)
    circuit.append(trotter_step_first_order, range(nqubits + 1))
    # circuit = circuit.decompose(reps=2)
    return circuit


def select_sim(error_rate):
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
    error1 = depolarizing_error(error_rate * 10, 2)
    noise_model.add_all_qubit_quantum_error(error1, "cx")
    sim_d = AerSimulator(noise_model=noise_model)
    if error_rate == 0:
        simulator = AerSimulator()
    else:
        simulator = sim_d
    return simulator


def cosine_filtering(H, time, nqubits, order, time_step, error_rate, step):
    simulator = select_sim(error_rate)
    expectation_list = []
    probability_list = []
    initial_state = Statevector.from_label("0000")
    statevector = initial_state.data

    H_cosine = SparsePauliOp(["Y"]) ^ H
    H_matrix = H.to_matrix()

    for i in range(step):
        ## cosine filter part
        qc = QuantumCircuit(nqubits + 1)
        qc.initialize(statevector, range(nqubits))
        circuit_temp = construct_trotter_circuit(
            H_cosine, time, nqubits, order, time_step
        )
        qc = qc.compose(circuit_temp, range(nqubits + 1))
        qc1 = qc.copy()
        qc.save_statevector()
        circuit = transpile(qc, simulator)
        result = simulator.run(circuit).result().data(0)["statevector"]
        new_state = result.data
        postselect_state = new_state[: 2**nqubits]
        probability = np.linalg.norm(postselect_state)
        postselect_state /= probability
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state))
        if i == 0:
            probability = probability
        else:
            probability *= probability_list[-1]
        probability_list.append(probability)
        expectation_list.append(expectation)
        statevector = postselect_state

    expectation_list = np.array(expectation_list)
    probability_list = np.array(probability_list)

    assert np.max(np.abs(np.imag(expectation_list))) < 1e-6

    return expectation_list.real, probability_list


class VQECallback:
    def __init__(self, sim, circuit, nqubits, reference, H_matrix, threshold, param):
        self.sim = sim
        self.circuit = circuit
        self.nqubits = nqubits
        self.reference = reference
        self.H_matrix = H_matrix
        self.threshold = threshold
        self.param = param

    def run_sim(self, params):
        new_circuit = self.circuit.assign_parameters(params)
        result = self.sim.run(new_circuit).result()
        statevector = result.data(0)["statevector"].data
        return statevector

    def cost_func(self, params):
        statevector = self.run_sim(params)
        probability = np.linalg.norm(statevector[: 2**self.nqubits])
        return -probability

    def cost_func_3(self, params):
        # for cosine_filtering_vqe_3
        statevector = self.run_sim(params)
        statevector = statevector[: 2**self.nqubits]
        probability = np.linalg.norm(statevector[: 2**self.nqubits])
        statevector /= probability
        expectation = statevector.conj().dot(self.H_matrix.dot(statevector)).real
        return -probability + 1 * (expectation - self.reference)

    def constraint_func(self, params):
        statevector = self.run_sim(params)
        statevector = statevector[: 2**self.nqubits]
        statevector /= np.linalg.norm(statevector)

        expectation = statevector.conj().T.dot(self.H_matrix.dot(statevector)).real
        return -(expectation - self.reference - self.threshold)
        #return -np.abs(expectation - self.reference - self.threshold)


def cosine_filtering_vqe(H, time, nqubits, order, time_step, error_rate, step):
    simulator = select_sim(error_rate)
    expectation_before_list = []
    probability_before_list = []
    expectation_after_list = []
    probability_after_list = []
    initial_state = Statevector.from_label("0000")
    statevector = initial_state.data

    H_cosine = SparsePauliOp(["Y"]) ^ H
    H_matrix = H.to_matrix()

    vqe_depth = 3
    vqe_nparams = vqe_depth * (2 * nqubits + 1)
    vqe_params = ParameterVector("θ", vqe_nparams)
    #parameter = np.random.random(vqe_nparams)

    for i in range(step):
        print(f"Running {i}", flush=True)
        ## cosine filter part
        qc = QuantumCircuit(nqubits + 1)
        qc.initialize(statevector, range(nqubits))
        circuit_temp = construct_trotter_circuit(
            H_cosine, time, nqubits, order, time_step
        )
        qc = qc.compose(circuit_temp, range(nqubits + 1))
        qc1 = qc.copy()
        qc.save_statevector()
        circuit = transpile(qc, simulator)
        result = simulator.run(circuit).result().data(0)["statevector"]
        new_state = result.data
        postselect_state = new_state[: 2**nqubits]
        probability = np.linalg.norm(postselect_state)
        postselect_state /= probability
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state))
        if i == 0:
            probability = probability
        else:
            probability *= probability_before_list[-1]
        probability_before_list.append(probability)
        expectation_before_list.append(expectation)
        statevector = postselect_state
        ## vqe part
        vqe_qc = qc1
        vqe_ansatz = ansatz_vqe(nqubits, vqe_depth, vqe_params)
        vqe_qc.compose(vqe_ansatz, range(nqubits + 1), inplace=True)
        vqe_qc.save_statevector()
        vqe_qc = transpile(vqe_qc, simulator)
        vqe_callback = VQECallback(
            simulator, vqe_qc, nqubits, expectation.real, H_matrix, 1e-1, vqe_params
        )

        np.random.seed(42)
        parameter = np.random.random(vqe_nparams)
        # parameter = np.zeros(vqe_nparams, dtype=np.float64)
        vqe_result = sp.optimize.minimize(
            vqe_callback.cost_func,
            parameter,
            method="SLSQP",
            constraints={"type": "ineq", "fun": vqe_callback.constraint_func},
            options={"maxiter" : 1000},
        )
        print(vqe_result)
        #parameter = vqe_result.x
        postselect_state = vqe_callback.run_sim(vqe_result.x)[: 2**nqubits]
        probability = np.linalg.norm(postselect_state)
        print(f"Probability at this step is {probability}")
        postselect_state /= probability
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state))
        if i == 0:
            probability = probability
        else:
            probability *= probability_after_list[-1]
        print(f"Probability consider previous step is {probability}")
        probability_after_list.append(probability)
        expectation_after_list.append(expectation)
        statevector = postselect_state

    expectation_before_list = np.array(expectation_before_list)
    probability_before_list = np.array(probability_before_list)
    expectation_after_list = np.array(expectation_after_list)
    probability_after_list = np.array(probability_after_list)

    assert np.max(np.abs(np.imag(expectation_before_list))) < 1e-6
    assert np.max(np.abs(np.imag(expectation_after_list))) < 1e-6

    return (
        expectation_before_list.real,
        probability_before_list,
        expectation_after_list.real,
        probability_after_list,
    )


def cosine_filtering_vqe_2(
    H, time, nqubits, order, time_step, error_rate, step, reject_if_fail=False
):
    # INSTEAD OF HAVING THE COSINE FILTERING CIRCUIT DURING THE VQE CALCULATION
    # WE JUST SAVE THE COSINE FILTERING STATEVECTOR AND THEN INIITIALIZE THE
    # VQE CIRCUIT WITH THE SAVED STATEVECTOR
    # MAYBE THAT CAN GO FASTER?
    simulator = select_sim(error_rate)
    expectation_before_list = []
    probability_before_list = []
    expectation_after_list = []
    probability_after_list = []
    initial_state = Statevector.from_label("0000")
    statevector = initial_state.data

    H_cosine = SparsePauliOp(["Y"]) ^ H
    H_matrix = H.to_matrix()

    vqe_depth = 3
    vqe_nparams = vqe_depth * (2 * nqubits + 1)
    vqe_params = ParameterVector("θ", vqe_nparams)
    parameter = np.random.random(vqe_nparams)

    for i in range(step):
        print(f"Running {i}", flush=True)
        ## cosine filter part
        qc = QuantumCircuit(nqubits + 1)
        qc.initialize(statevector, range(nqubits))
        circuit_temp = construct_trotter_circuit(
            H_cosine, time, nqubits, order, time_step
        )
        qc = qc.compose(circuit_temp, range(nqubits + 1))
        qc1 = qc.copy()
        qc.save_statevector()
        circuit = transpile(qc, simulator)
        result = simulator.run(circuit).result().data(0)["statevector"]
        new_state = result.data
        postselect_state = np.copy(new_state[: 2**nqubits])
        probability = np.linalg.norm(postselect_state)
        postselect_state /= probability
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state))
        if i == 0:
            probability = probability
        else:
            probability *= probability_before_list[-1]
        probability_before_list.append(probability)
        expectation_before_list.append(expectation)
        statevector = postselect_state
        temp_state = statevector
        temp_pro = probability
        temp_e = expectation

        old_expectation = expectation.real
        ## vqe part
        vqe_qc = QuantumCircuit(nqubits + 1)
        vqe_qc.initialize(new_state, range(nqubits + 1)) #####
        vqe_ansatz = ansatz_vqe(nqubits, vqe_depth, vqe_params)
        vqe_qc.compose(vqe_ansatz, range(nqubits + 1), inplace=True)
        vqe_qc.save_statevector()
        vqe_qc = transpile(vqe_qc, simulator)
        vqe_callback = VQECallback(
            simulator, vqe_qc, nqubits, expectation.real, H_matrix, 1e-1, vqe_params
        )
        np.random.seed(42)
        #parameter = np.random.random(vqe_nparams)
        # parameter = np.zeros(vqe_nparams, dtype=np.float64)
        vqe_result = sp.optimize.minimize(
            vqe_callback.cost_func,
            parameter,
            method="SLSQP",
            constraints={"type": "ineq", "fun": vqe_callback.constraint_func},
        )
        print(vqe_result)
        parameter = vqe_result.x
        postselect_state = vqe_callback.run_sim(vqe_result.x)[: 2**nqubits]
        probability = np.linalg.norm(postselect_state)
        postselect_state /= probability
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state))
        new_expectation = expectation.real
        print(f"    {old_expectation} {new_expectation} ", end="")
        if -(new_expectation - old_expectation - 1e-1) >= 0:
        #if np.abs(new_expectation - old_expectation)<= 1e-1:
            print(f"CONSTRAINT SATISFIED")
            statevector = postselect_state
            if i == 0:
                probability = probability
            else:
                probability *= probability_after_list[-1]
            probability_after_list.append(probability)
            expectation_after_list.append(expectation)
        else:
            if reject_if_fail:
                print(f"CONSTRAINT UNSATISFIED")
            else:
                print("NOT REJECTING")
                statevector = temp_state
                probability_after_list.append(temp_pro)
                expectation_after_list.append(temp_e)
    expectation_before_list = np.array(expectation_before_list)
    probability_before_list = np.array(probability_before_list)
    expectation_after_list = np.array(expectation_after_list)
    probability_after_list = np.array(probability_after_list)

    assert np.max(np.abs(np.imag(expectation_before_list))) < 1e-6
    assert np.max(np.abs(np.imag(expectation_after_list))) < 1e-6

    return (
        expectation_before_list.real,
        probability_before_list,
        expectation_after_list.real,
        probability_after_list,
    )


def cosine_filtering_vqe_3(H, time, nqubits, order, time_step, error_rate, step):
    # MODIFY HOW THE CONSTRAINT IS USED
    # NOT SURE ABOUT THIS ONE
    # THIS IS JUST SOLVING VQE
    # MAYBE WE CAN JUST USE A LOW DEPTH VQE THAT CONVERGE QUICKLY
    # THEN USE THE COSINE FILTERING TO DO THE ACTUAL GROUND STATE EVOLUTION
    simulator = select_sim(error_rate)
    expectation_before_list = []
    probability_before_list = []
    expectation_after_list = []
    probability_after_list = []
    initial_state = Statevector.from_label("0000")
    statevector = initial_state.data

    H_cosine = SparsePauliOp(["Y"]) ^ H
    H_matrix = H.to_matrix()

    vqe_depth = 1
    vqe_nparams = vqe_depth * (2 * nqubits + 1)
    vqe_params = ParameterVector("θ", vqe_nparams)

    for i in range(step):
        print(f"Running {i}", flush=True)
        ## cosine filter part
        qc = QuantumCircuit(nqubits + 1)
        qc.initialize(statevector, range(nqubits))
        circuit_temp = construct_trotter_circuit(
            H_cosine, time, nqubits, order, time_step
        )
        qc = qc.compose(circuit_temp, range(nqubits + 1))
        qc1 = qc.copy()
        qc.save_statevector()
        circuit = transpile(qc, simulator)
        result = simulator.run(circuit).result().data(0)["statevector"]
        new_state = result.data
        postselect_state = np.copy(new_state[: 2**nqubits])
        probability = np.linalg.norm(postselect_state)
        postselect_state /= probability
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state))
        #if i == 0:
        #    probability = probability
        #else:
        #    probability *= probability_before_list[-1]
        probability_before_list.append(probability)
        expectation_before_list.append(expectation)
        statevector = postselect_state

        old_expectation = expectation.real

        vqe_qc = QuantumCircuit(nqubits + 1)
        vqe_qc.initialize(new_state, range(nqubits + 1))
        vqe_ansatz = ansatz_vqe(nqubits, vqe_depth, vqe_params)
        vqe_qc.compose(vqe_ansatz, range(nqubits + 1), inplace=True)
        vqe_qc.save_statevector()
        vqe_qc = transpile(vqe_qc, simulator)
        vqe_callback = VQECallback(
            simulator, vqe_qc, nqubits, expectation.real, H_matrix, 1e-2, vqe_params
        )

        np.random.seed(42)
        parameter = np.random.random(vqe_nparams)
        # parameter = np.zeros(vqe_nparams, dtype=np.float64)
        vqe_result = sp.optimize.minimize(
            vqe_callback.cost_func_3,
            parameter,
            method="SLSQP",
        )
        print(vqe_result)

        postselect_state = vqe_callback.run_sim(vqe_result.x)[: 2**nqubits]
        probability = np.linalg.norm(postselect_state)
        postselect_state /= probability
        expectation = postselect_state.conj().T.dot(H_matrix.dot(postselect_state))
        #if i == 0:
        #    probability = probability
        #else:
        #    probability *= probability_after_list[-1]
        probability_after_list.append(probability)
        expectation_after_list.append(expectation)

        new_expectation = expectation.real

        statevector = postselect_state

    expectation_before_list = np.array(expectation_before_list)
    probability_before_list = np.array(probability_before_list)
    expectation_after_list = np.array(expectation_after_list)
    probability_after_list = np.array(probability_after_list)

    assert np.max(np.abs(np.imag(expectation_before_list))) < 1e-6
    assert np.max(np.abs(np.imag(expectation_after_list))) < 1e-6

    return (
        expectation_before_list.real,
        probability_before_list,
        expectation_after_list.real,
        probability_after_list,
    )


def plot_data(
    filename, nstep, f_e, f_p, fv_before_e, fv_before_p, fv_after_e, fv_after_p
):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    step = list(range(nstep))

    colors = ["darkmagenta", "limegreen", "k"]
    markers = ["*", "x", "v", ":"]

    ax1.plot(
        step,
        f_e,
        label="Filter Expect",
        marker=markers[0],
        c=colors[0],
        ls="-",
        lw=0.8,
    )
    ax1.plot(
        step,
        fv_before_e,
        label="Filter+VQE Expect before",
        marker=markers[1],
        c=colors[0],
        ls="-",
        lw=0.8,
    )
    ax1.plot(
        step,
        fv_after_e,
        label="Filter+VQE Expect after",
        marker=markers[2],
        c=colors[0],
        ls="-",
        lw=0.8,
    )
    ax1.axhline(
        y=np.min([f_e, fv_before_e, fv_after_e]),
        color="r",
        linestyle="--",
        label="Minimum expectation",
    )
    # ax1.plot(step, expectation_exact, c="k", ls=":", label="Expectation exact")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Expectation", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    ax2 = ax1.twinx()

    ax2.plot(
        step,
        f_p,
        label="Filter Prob",
        marker=markers[0],
        c=colors[1],
        ls="-",
        lw=0.8,
    )
    ax2.plot(
        step,
        fv_before_p,
        label="Filter+VQE Prob before",
        marker=markers[1],
        c=colors[1],
        ls="-",
        lw=0.8,
    )
    ax2.plot(
        step,
        fv_after_p,
        label="Filter+VQE Prob after",
        marker=markers[2],
        c=colors[1],
        ls="-",
        lw=0.8,
    )

    ax2.set_ylabel("Probability", color=colors[1])
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor=colors[1])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    legend = fig.legend(
        all_handles,
        all_labels,
        bbox_to_anchor=(0.80, 0.5),
        loc="center left",
        # framealpha=0.5,
    )

    ax1.set_title("Comparison of Expectations and Probabilities over Steps")

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    fig.savefig(filename, dpi=600)


if __name__ == "__main__":
    nqubits = 4
    J = 1.0 / np.sqrt(2)
    hamiltonian = get_hamiltonian(nqubits, J)
    H_array = hamiltonian.to_matrix()
    eval, _ = np.linalg.eigh(H_array)
    # print(eval)
    emin = eval[0]
    emax = eval[-1]
    H_array = (H_array - emin * np.eye(2**nqubits)) / (emax - emin)  ## scale H
    final_time = np.pi / 2  # time
    time = final_time
    time_step = 3  # time_step
    initial_state = Statevector.from_label("00000")
    error_rate = 0
    order = 2
    nstep = 20

    H = SparsePauliOp.from_operator(Operator(H_array))

    f_e, f_p = cosine_filtering(H, time, nqubits, order, time_step, error_rate, nstep)
    f_before_e, f_before_p, f_after_e, f_after_p = cosine_filtering_vqe(
        H, time, nqubits, order, time_step, error_rate, nstep
    )
    print(f_e)
    print(f_p)
    print(f_before_e)
    print(f_before_p)
    print(f_after_e)
    print(f_after_p)
    plot_data("data_1.png", nstep, f_e, f_p, f_before_e, f_before_p, f_after_e, f_after_p)
