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
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE
import pennylane as qml

def get_hamiltonian(nq, J):
    op_list = [("ZZ", [i, i + 1], J) for i in range(nq - 1)]
    op_list.append(("ZZ", [nq - 1, 0], J))
    op_list += [("X", [i], J) for i in range(nq)]
    return SparsePauliOp.from_sparse_list(op_list, num_qubits=nq).simplify()

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


def cosine_filtering(H, time, nqubits, order, time_step, error_rate, step, threshold):
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
        if expectation <= threshold:
            break
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


def plot_data(
    filename, nstep, f_e, f_p, f_e1, f_p1, f_e2, f_p2, f_e4, f_p4):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    step = list(range(nstep))
    colors = ["darkmagenta", "limegreen", "k", 'b']
    markers = ["+", "x", "v", "*"]
    length = len(f_e)
    length1 = len(f_e1)
    length2 = len(f_e2)
    length4 = len(f_e4)
    ax1.plot(
        step[:length2],
        f_e2,
        label="Expectations for Second order trotterization",
        marker=markers[2],
        c=colors[0],
        ls="-",
        lw=0.8,
    )
    ax1.plot(
        step[:length1],
        f_e1,
        label="Expectations for First order trotterization",
        marker=markers[0],
        c=colors[0],
        ls="-",
        lw=0.8,
    )
    ax1.plot(
        step[:length],
        f_e,
        label="Expectations for Exact matrix exponential",
        marker=markers[3],
        c=colors[0],
        ls="-",
        lw=0.8,
    )
    ax1.plot(
        step[:length4],
        f_e4,
        label="Expectations for Forth order trotterization",
        marker=markers[1],
        c=colors[0],
        ls="-",
        lw=0.8,
    )
    ax1.axhline(
        y=np.min(np.concatenate([f_e1, f_e, f_e2, f_e4])),
        color="r",
        linestyle="--",
        label="Minimum expectation",
    )
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Expectation", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax2 = ax1.twinx()
    ax2.plot(
        step[:length2],
        f_p2,
        label="Successful probability for Second order trotterization",
        marker=markers[2],
        c=colors[1],
        ls="-",
        lw=0.8,
    )
    ax2.plot(
        step[:length1],
        f_p1,
        label="Successful probability for First order trotterization",
        marker=markers[0],
        c=colors[1],
        ls="-",
        lw=0.8,
    )
    ax2.plot(
        step[:length],
        f_p,
        label="Successful probability for Exact matrix exponential",
        marker=markers[3],
        c=colors[1],
        ls="-",
        lw=0.8,
    )
    ax2.plot(
        step[:length4],
        f_p4,
        label="Successful probability for Fourth order trotterization",
        marker=markers[1],
        c=colors[1],
        ls="-",
        lw=0.8,
    )
    ax2.set_ylabel("Successful Probability", color=colors[1])
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor=colors[1])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    legend = fig.legend(
        all_handles,
        all_labels,
        bbox_to_anchor=(0, 0, 1, 1),
        loc="upper right",
        #ncol=3,
        borderaxespad=0,
        fontsize=8,
        bbox_transform=ax1.transAxes
    )

    #ax1.set_title("Comparison of Expectations and Probabilities over Steps")

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
    time_step = 3  # trotter step
    initial_state = Statevector.from_label("00000")
    error_rate = 0
    order = 2
    nstep = 25
    threshold = 1e-6

    H = SparsePauliOp.from_operator(Operator(H_array))

    depth = 3
    # Pure Cosine filter
    order = 0
    f_e, f_p = cosine_filtering(H, time, nqubits, order, time_step, error_rate, nstep, threshold)
    order = 1
    f_e1, f_p1 = cosine_filtering(H, time, nqubits, order, time_step, error_rate, nstep, threshold)
    order = 2
    f_e2, f_p2 = cosine_filtering(H, time, nqubits, order, time_step, error_rate, nstep, threshold)
    order = 4
    f_e4, f_p4 = cosine_filtering(H, time, nqubits, order, time_step, error_rate, nstep, threshold)

    plot_data('pure Cosine filter with Trotterization', nstep, f_e, f_p, f_e1, f_p1, f_e2, f_p2, f_e4, f_p4)
    with open('pure_cosine.txt', 'w', encoding='utf-8') as f:
       for i in range(nstep):
           f.write(f"{f_e[i]}\t{f_e1[i]}\t{f_e2[i]}\t{f_e4[i]}\t{f_p[i]}\t{f_p1[i]}\t{f_p2[i]}\t{f_p4[i]}\n")
       print('Result has been stored')
